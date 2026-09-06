//! The bounded numeric recorder.
//!
//! The recorder watches the interpreter execute one iteration of a hot loop
//! and writes it down as a [`TraceIr`]. It never executes a Lua operation. The
//! interpreter calls [`Recorder::step`] before each guest instruction,
//! either from the fast loop or from the general dispatch loop, and then runs
//! the instruction as usual.
//!
//! The hook sees the instruction at `pc` and the registers as they are before
//! it runs. The recorder specialises on the types it sees and computes branch
//! directions from the actual values. It also predicts the next pc and
//! checks that the next hook agrees, so a divergence between the recorder's
//! model and the interpreter aborts the recording instead of producing a
//! wrong trace.
//!
//! # What it records
//!
//! Integer, float, boolean and nil values:
//! `Call`, `CallOne`, `CallOpen`, `TailCall`, `Return` and `ReturnOpen` of Lua
//! closures (see below),
//! `Move`, `LoadConstant`, `LoadNil`, `ClearRange`, `Not`, `Jump`, `GetTable`,
//! `SetTable`, `GetField`, `SetField`, `Length`,
//! `JumpIfFalse`, `Negate`, `Add`, `Subtract`, `Multiply`, `Divide`,
//! `Modulo`, `Power`, `FloorDivide`, the bitwise operations and shifts,
//! `Equal`, `LessThan`, `LessEqual`, `NumericForPrep` and `NumericForLoop`,
//! plus `GetUpvalue`, `SetUpvalue` and `GetGlobal`, and `LoadConstant` of a
//! string. Any other opcode or operand type aborts the recording with a
//! reason. Arithmetic on a numeric string aborts: the interpreter converts it.
//!
//! A branch on a boolean (`JumpIfFalse`, or a compare-and-branch test) whose
//! two arms are pure and join a few instructions later records no guard:
//! both arms are recorded and each register they write takes a value select
//! (see the `select` module and "Value selects" in [`super`]). The
//! interpreter then runs the arm it takes while the recorder follows its
//! pcs without recording them again.
//!
//! `LuaJIT`'s 64-bit integer cdata (`int64_t` and `uint64_t`, the `LL` and `ULL`
//! values) are typed lanes of their own ([`TraceType::Int64`] and
//! [`TraceType::UInt64`]), guarded by type at entry like any other lane. An
//! arithmetic instruction or comparison with a cdata operand converts both
//! operands to their common 64-bit type, as the interpreter does (a number
//! truncates, unsigned wins over signed), and then wraps. The operations are
//! total (a zero divisor has a result), so none needs a guard. `tonumber` and
//! the `bit` library functions `band`, `bor`, `bxor`, `bnot`, `tobit`,
//! `lshift`, `rshift`, `arshift`, `rol` and `ror` record as trace operations
//! behind the callee guard when an argument is a cdata. A float shifted by a
//! cdata count, three or more `bit` operands, arrays and fields that hold
//! cdata, and `^` abort the recording.
//!
//! A string is an opaque value like a table, known by identity. It moves
//! between registers, cells and globals and rides to snapshots. Its bytes are
//! readable when it is an entry input no iteration writes: `#s`,
//! `string.byte (s, i)` and `string.sub (s, i, i)` read them in place (see
//! [`oximond::jit_ir::trace::BindingKind::Bytes`]). The one-byte string that
//! `sub` makes is never made: it is the open result of the call, which only
//! `string.byte` takes, so `string.byte (s:sub (i, i))` is one byte read.
//! Two strings never compare natively, since equal bytes may sit in two
//! allocations.
//!
//! A string an array read (`keys[i]`) finds in a bound table's slot is not
//! given a lane: it is borrowed ([`oximond::jit_ir::trace::TraceType::StringRef`]),
//! the address of the slot. The owner is a table the entry bound, so it stays
//! rooted for the run. A borrow moves between registers and rides to
//! snapshots, where the restore clones the string out of the slot. Whatever
//! could replace or move the slot ends the borrows (see
//! [`oximond::jit_ir::trace::TraceOp::ends_borrows`]): the recorder rejects a
//! recording that records one while any register still holds a borrow
//! ([`TraceAbortReason::BorrowOutlivesEffect`]), since it has no operation
//! that owns the string first. A register that is live across the loop-back
//! and receives a borrow cannot be a typed parameter, so such a loop aborts as
//! type-unstable.
//!
//! Tables and callables (Lua closures, native functions and the `ipairs`
//! iterator) are opaque values. They move between registers, cells and
//! globals, compare for identity and test for truthiness. A table is also the
//! receiver of the table operations below. A callable can be read out of a
//! table field (a method, `math.sin`) as a constant the entry checks (see
//! [`oximond::jit_ir::trace::BindingKind::Callee`]). Nothing reads a table out
//! of a table, so in Lua 5.2 and later, where a global is a field of `_ENV`,
//! `math.sin` is native only through a local. Two tables compare natively
//! only when the recording saw the same table twice, behind a guard that
//! leaves before the comparison otherwise, so the interpreter can run `__eq`.
//! Values of different types compare unequal without a metamethod (`__eq`
//! needs two tables), so the entry types make that comparison a constant.
//!
//! `GetTable`, `SetTable` and `Length` on a table that never changes between
//! iterations record as table operations (see
//! [`oximond::jit_ir::trace::TableBinding`] and the "Tables" section of
//! [`super`]): a number key that names a slot of the array part, a constant
//! string key (in the table or, through table-valued `__index`, in a fallback
//! table), and the length. The recorder reads the live slot to learn its
//! type and refuses (with `UnsupportedOperand`) whatever native code could
//! not repeat: a key outside the array, an absent or nil slot, a value of a
//! type with no payload, a store that would change a slot's type, a table
//! the loop rewrites. A string key is the bytes the interpreter loaded into
//! the key register, or the constant a field instruction names, so it does
//! not matter which chunk's constant they came from.
//!
//! A read of a key the table lacks, a write of a new key and `#t` run the
//! function the metatable names as `__index`, `__newindex` and (in the
//! profiles that ask for it) `__len`. When that function is a Lua closure the
//! recorder inlines it as a call (see `inline_meta`): a helper walks the
//! metatables as the interpreter does and checks that the raw handler at the
//! end is the pinned closure, so every table of the class passes. A store
//! under a string key into a table the trace does not bind, and one a
//! table-valued `__newindex` redirects, is a helper too
//! ([`oximond::jit_ir::trace::HelperId::TABLE_SET_STRING`]).
//!
//! A table that does change is a table known by lane: the array read gives it
//! one, and `t.name` and `t.name = v` on it record as field operations that
//! find the slot through the lane (see [`oximond::jit_ir::trace::TableField`]).
//! A table read out of a bound array always is. A table in a register the
//! loops around this loop write is too, when a trace tree could run those
//! loops (see [`super::nest::can_cover`]): the tree's outer sides write the
//! register, so the entry cannot bind the table's fields once. Such a table
//! takes no `#` through its lane, and its number keys inside the array part
//! read the array through the table's own words (a lane array view,
//! [`oximond::jit_ir::trace::TraceOp::LaneArrayLoad`]): the row holds the
//! table's address, and the load reads the array pointer and length there.
//! A key in the hash part, a string or closure element and an `__index` walk
//! stay lookup calls. An entry-bound table keeps its bound array and length.
//! A side whose parent's snapshot holds a table loaded inside the tree in a
//! register (`SideSeed::lanes`) starts with that register as a lane, so its
//! first recording reads the way the root does. Because these reads abort
//! nothing, the coverage gate accepts outer parts that index, measure or
//! store into a table read out of a table (`#l.z`, `l.y[i]`), and a short
//! `while` or generic-for root in such a nest is kept as a stepping stone
//! while its outer sides land (see "Nested loops" in [`super`]). Once it is
//! one, it keeps its branch guards in the loop (`TraceIr::keep_branch_guards`),
//! and an outer side may read the nil that ends an `ipairs` loop.
//!
//! An upvalue cell or a global the iteration touches becomes an extra window
//! register (see [`oximond::jit_ir::trace::ExternalSlot`]). Its entry type is
//! what it holds when first used, which equals what it held at the loop
//! header because nothing in the iteration wrote it before. Globals are only
//! read as window registers when the environment holds a present raw value;
//! a missing one runs `__index`, which is a call when it is a Lua closure.
//! `_ENV` and two names for one cell abort.
//!
//! # Virtual tables
//!
//! A `NewTable` in the root frame or in an inlined callee makes no table. The
//! register names a virtual table (`Sym::Virtual`), whose array and fields
//! are recorded values, and a `PhantomTable` operation stands for the
//! constructor: native code counts it, and the VM charges the collector for
//! the table at the end of the run, as `charge_elided_table` does. The
//! constructor's stores, reads of constant keys, a `#` with one border, and
//! moves between registers all resolve in the recorder with the
//! interpreter's table semantics (numeric keys normalized, a duplicate key
//! overwriting in place, nil deleting).
//!
//! `setmetatable (t, mt)` of a virtual table keeps it virtual: the recipe
//! carries `mt` and an exit sets it after the contents. A helper checks `mt`
//! each time the call runs (no `__gc`, no `__mode`, and none of `__index`,
//! `__newindex` and `__len` that the recording did without), and an access
//! that one of those would answer rejects the recording.
//!
//! The exception is `error(tbl)` under a `pcall` of the trace, and a failed
//! `assert (false, tbl)` in the profiles that raise the message unchanged
//! (Lua 5.3 and later, and `LuaJIT`): the table lands in the caller's result
//! register, still virtual, and the recording goes on through the caught
//! path, since no side trace can start from a table that exists only in the
//! trace (see [`inline_caught`]). Lua 5.1 and 5.2 build an argument error
//! for a message that is no string, which the recorder does not build, so
//! that `assert` rejects the recording.
//!
//! A table stored in a virtual table stays virtual: `{b1, {b1 + b2}}` makes
//! two constructors whose recipes nest, and reading the inner table back
//! (`t[2][1]`) names it by a register again. A snapshot lists the inner
//! recipe beside the outer one, and an exit builds every table it names
//! before it fills any, so the nest comes out whole. A table that would hold
//! itself is made real instead, and the closing store goes through
//! [`HelperId::TABLE_INIT`]. Later stores into either table are calls of
//! [`HelperId::TABLE_INIT`] too, as long as nothing else names the tables
//! ([`Recorder::record_escaped_set`]), and the model takes them, so the
//! recording goes on after the cycle closes.
//!
//! A table that escapes becomes real at that point. A store into a real
//! table (an array slot, a field, a global), a compare, a truth test, an
//! argument of a call (an inlined Lua callee or a native other than
//! `setmetatable` and the error forms) and a returned value, and any other
//! read of the register as a value, makes the table and its contents with calls of
//! [`HelperId::TABLE_NEW`] and [`HelperId::TABLE_INIT`], and the recorder
//! goes on with the lane of the new table as the value
//! ([`Recorder::materialize`]). Tables the table holds are made with it, all
//! before any is filled. The run's
//! bank roots them and the loop-back release drops the lanes, so a loop that
//! makes a table per iteration does not grow the bank. The recorded contents
//! stay as a model of the real table: reads of constant keys go on resolving
//! in the recorder for as long as no write could have reached the table (see
//! [`Recorder::shadow_is_current`]), because a table met by lane takes no
//! number key natively. A write to the table after it escaped rejects the
//! recording, and so does an escape of a table with a metatable. No
//! phantom charge stands for a table that escaped: the real table counts by
//! itself, and a snapshot taken before the escape builds the table from its
//! recipe, which counts the same way.
//!
//! What still rejects the recording with [`TraceAbortReason::VirtualEscape`]
//! (or an abort of the operation's own): a table captured by a closure or
//! built into a captured local, a read of a number key of a table the
//! callee got by lane, a key the
//! trace computes, a deletion that would leave a hole, and an access that a
//! metamethod of the table's metatable would answer. A table in a register
//! the loop-back needs
//! (named locals in scope count, as for every register) rejects too, made or
//! not. A register that the loop writes before it reads, which still holds
//! last iteration's made table, does not.
//!
//! A snapshot lists the tables its live registers name as recipes
//! ([`oximond::jit_ir::trace::VirtualTable`]) with the contents as they stood
//! when it was taken, and a register whose table is dead at the resume point
//! restores nil. Each recipe records the entry count its phantom charge used
//! (the table's peak size), so building the table at an exit takes that
//! charge back and the table is counted once.
//!
//! # Calls
//!
//! A `Call` or `CallOne` of a Lua closure with fixed arguments and results
//! is inlined. The interpreter makes the real call and pushes the callee's
//! frame, and the recorder follows it: the next hook is the callee's first
//! instruction one frame deeper, and the callee's `Return` leads back to the
//! caller's next instruction. The recorder keeps a symbolic register file per
//! inlined frame (a fresh activation holds its arguments and nil) and an IR
//! frame per call, so a guard inside a callee has a snapshot that lists the
//! root frame stopped at its call and every callee above it.
//!
//! The callee is pinned by identity. A guard compares the live closure with
//! the recorded one (a lane of the trace's pinned closures, see
//! [`oximond::jit_ir::trace::TraceIr::pinned_closures`]) before the call, with
//! a `Before` snapshot, because two closures of one prototype capture other
//! cells and a rebound callee must run in the interpreter. A callee's upvalues
//! and globals are window slots like the root's, read through the pinned
//! closure. A cell that a callee shares with the root frame's captured local,
//! or with another callee, has one register: the recorder aliases the names
//! and the VM checks the promise at entry.
//!
//! A call site whose callee is not one closure (the register the callee
//! sits in is written by the loop, as when locals rotate, or the callee is
//! read out of an array) is guarded on the callee's prototype instead
//! ([`oximond::jit_ir::trace::HelperId::CLOSURE_SAME_PROTO`]), and the frame
//! is `dynamic` ([`oximond::jit_ir::trace::InlinedFrame::dynamic`]): it runs
//! whichever closure of the prototype the call had, which its snapshots carry
//! as a register after the chunk's. Its upvalues are no window slots, for
//! another closure has other cells. `GetUpvalue` and `SetUpvalue` go through
//! the live closure
//! ([`oximond::jit_ir::trace::HelperId::CELL_GET_INTEGER`] and its siblings,
//! [`oximond::jit_ir::trace::HelperId::CELL_SET`]), which native code does
//! itself through the closures' rows where the cell holds a number or a
//! boolean. The cell is open or closed, as the runtime has it, so a write
//! reaches the enclosing frame's variable at once and a table stored into a
//! cell reaches the collector's write barrier. A callee that reads a global
//! or an environment field in such a frame aborts the recording: another
//! closure of the prototype may have another environment. The root's captured
//! locals may not be touched by a trace with such a frame (see
//! [`SideSeed::captured_busy`]), and a run refuses to call through the
//! prototype guard a closure that shares a cell with a window slot (see
//! `RunTables::held_cells`).
//!
//! A site that takes turns between more callees than one trace holds is a
//! chain: the root pins the first callee, and the guard of each side pins the
//! next, up to [`policy::MAX_SIDE_DEPTH`] sides deep (sixteen, so a table of
//! sixteen functions runs natively; with more, the targets past the chain
//! leave through the last guard). Callees made by `load` are closures like
//! any other: their chunk is their prototype, pinned by identity, and a
//! second closure of the same loaded chunk shares a prototype guard.
//!
//! A side may name callee cells the root did not (and write them): see the
//! notes on cells in the parent module. In a frame that runs whichever
//! closure of a prototype the call had, both go through the live closure.
//!
//! A call of a native function with a pure numeric form (`math.sin`,
//! `math.floor`, `math.max`, `tonumber` of a float) becomes a call of that
//! function from native code, behind the same identity guard. A step of an
//! `ipairs` loop becomes an array read.
//!
//! A call of `coroutine.resume`, or of a function `coroutine.wrap` made, in the
//! root frame becomes a call of a helper that runs the thread through the VM
//! from native code (see `inline_resume`). The recorder sleeps through the
//! call, like a self call's, to learn from the result registers which values
//! the thread gave, and the helper misses when a later call gives others. A
//! thread is a lane of the run's bank (`TraceType::Thread`), an array read
//! gives one, and the trace holds no cell, global, field or lane the thread
//! could change without reading it again after each call. A loop that calls
//! `coroutine.yield` on every iteration is rejected at its first recording:
//! the thread leaves it in the middle of an iteration.
//!
//! A `TailCall` of a Lua closure inside an inlined call replaces that call's
//! frame, as the interpreter does. The rebuilt frame of a snapshot counts the
//! tail calls and takes the name the last one gives it (see
//! [`oximond::jit_ir::trace::InlinedFrame::tail_calls`]). A `TailCall` of a
//! native function is a call with open results, which the `ReturnOpen` after
//! it returns.
//!
//! A `TailCall` of a Lua closure in the root frame replaces the root frame
//! (see "Loops through tail calls" below).
//!
//! A call whose result count is open (`C = 0`) leaves its values for the
//! `CallOpen`, `TailCallOpen` or `ReturnOpen` that takes them, after at most
//! some moves. The recorder knows how many there are: along the recorded path
//! the callee is pinned and so is every branch. The interpreter keeps the
//! values in a slot of the frame that a snapshot cannot rebuild, so no guard
//! may leave between the two calls: the second call's callee is guarded before
//! the first runs (see [`Recorder::hoist_callee`]). The note of that guard
//! is one of a stack, found by the consumer's frame depth and pc: the
//! producer may be a Lua call that is inlined, and its body holds producer and
//! consumer pairs of its own (`tonumber (band (x, y))`), which note and take
//! theirs before the outer consumer runs. A native that can miss
//! (a library helper) takes open results anyway, if neither call stored
//! anything: its guards leave through the snapshot before the first call, and
//! the interpreter makes both (see [`Recorder::open_region_for`]).
//!
//! A callee with varargs is inlined with the values past its parameters as
//! symbolic values. Its frame has storage a snapshot cannot describe, so every
//! guard inside the call leaves through the snapshot before the call and the
//! interpreter makes the call itself (see [`Recorder::reexec`]). A tail call
//! of the varargs (`return f(...)`) is followed there too, and a table store
//! inside such a call aborts the recording, since starting the call over would
//! store twice. Any other call that has stored nothing yet lets a branch guard
//! leave the same way (see [`Recorder::call_site`]), which is what gives the
//! other arm a side trace that starts at the call. Once the call wrote an
//! upvalue cell or ran more than [`BEFORE_CALL_MAX_PREFIX`] instructions, and
//! the frames are ones a side can rebuild ([`Recorder::frames_seedable`]),
//! the guard leaves through the call's frames instead: the interpreter would
//! otherwise redo the whole prefix of the call at every exit, and a side
//! starts inside the frames (see [`SideSeed::frames`] and "Sides from exits
//! inside inlined frames" in the trace module docs).
//!
//! Anything else aborts the recording: a native without a numeric form or
//! with varargs, open results that no call takes next, a chain of calls that
//! each take and leave open results, a callee already on the stack of inlined
//! calls (recursion), a loop inside a callee, a nesting deeper than a snapshot
//! can describe, and any error (except one a protected call of the trace
//! catches, see below), yield or hook. A root recording that meets a
//! loop inside a call starts again at the next iteration, a few times: the
//! iteration it started on may be the rare one that calls it.
//! While a recording runs, the call shortcuts that elide a frame (native
//! compiled calls and inline call plans) are off, so every call shows its
//! frame.
//!
//! An `error`, or an `assert` whose condition fails (also as a tail call),
//! inside a `pcall` or `xpcall` the root frame made is a caught exit, and
//! inside a `pcall` an inlined function made it is a caught path: the
//! recording lets the
//! interpreter unwind (the VM does not abort it for the error) and follows it
//! to the instruction after the protected call, with `false` and the error
//! value in the call's results. The trace does not unwind. A guard that
//! always fails, with a snapshot of the state after the call, ends the path
//! of the root frame's call, and a hot exit grows a side trace there. After
//! an inlined function's call the recording goes on in the trace. An `xpcall` handler that is a Lua
//! function is recorded as a call that starts over in the interpreter when it
//! leaves the trace. See [`inline_caught`] for the error value, which is the
//! interpreter's bit for bit, and what falls back to the old behavior.
//!
//! # Numbers per profile
//!
//! Lua 5.3 to 5.5 have integers and floats as distinct subtypes, and the
//! recorder keeps them apart: an entry guard says which one a register
//! holds. Lua 5.1, 5.2 and `LuaJIT` 2.1 have only floats, so a register that
//! somehow held an integer value aborts and an integer constant records as a
//! float. The operations follow the interpreter exactly:
//!
//! - Integer `+ - *` and negation wrap. `/` and `^` convert both operands
//!   to float. A float operand turns any other arithmetic into float
//!   arithmetic on the converted operands.
//! - Integer `//` and `%` round toward negative infinity and wrap
//!   `MIN // -1`. A zero divisor must raise the interpreter's error, so a
//!   `Before` guard on the divisor comes first and the interpreter runs the
//!   operation itself when it fails.
//! - Bitwise operations and shifts take integers. A float or string operand
//!   (which the interpreter may convert) aborts instead.
//! - An integer and a float compare by exact value, never through a lossy
//!   conversion.
//! - An integer numeric for loop stops where `checked_add` of the step
//!   overflows, which a `Before` guard on the headroom reproduces without an
//!   overflow flag. A loop whose three values are not all integers runs as a
//!   float loop, converted by `NumericForPrep` as the interpreter does.
//!
//! # Loop formation
//!
//! A root trace starts at the first hook after the interpreter takes a
//! backward branch to the header. It closes when a hook sees the same chunk,
//! depth and header pc again. A backward branch to a different header means
//! an inner loop exists, so the recording aborts and the inner loop gets its
//! own recording.
//!
//! # Loops through tail calls
//!
//! A tail call replaces its frame in place, so a function that tail calls
//! itself, or a cycle of functions that tail call each other, comes back to
//! pc 0 of the same function at the same depth: a loop whose header is pc 0.
//! The interpreter offers that pc to the policy after each tail call in
//! tracing mode (see `JitState::tail_landing`), and a recording starts there
//! with the latch at the chunk's last instruction (the loop spans the whole
//! function).
//!
//! A `TailCall` of a Lua closure in the root frame of such a recording runs
//! the callee in the root frame: a later activation of it ([`Activation`],
//! an IR frame with [`oximond::jit_ir::trace::RootActivation`]). The window
//! stays the root's. Its guest registers hold the callee's fresh activation
//! (the arguments, then nil, as moves and a constant at the call), and the
//! slots after them stay the trace's cells and globals, which the callee
//! reads through its pinned closure, as an inlined callee does. The callee is
//! pinned and guarded before the call. Each such call adds one to the frame's
//! tail call count, a window slot of its own
//! ([`oximond::jit_ir::trace::ExternalSlot::TailCalls`]).
//!
//! A tail call back into the closure the recording started with closes the
//! loop at its header ([`oximond::jit_ir::trace::TraceIr::loop_tail`]). Its
//! guard compares the callee with the closure the root frame runs at entry
//! ([`oximond::jit_ir::trace::ExternalSlot::RootClosure`]), so every closure of
//! the root chunk can run the trace with its own cells, and the guard is
//! loop-invariant whenever the callee comes from a cell or a global. The loop
//! parameters are the root chunk's registers at pc 0 (the moved arguments)
//! and the count. The poll snapshot resumes before the header with no
//! previous pc, as a tail call leaves the frame.
//!
//! The window's guest registers are the root chunk's: a callee with more
//! registers is refused ([`TraceAbortReason::RootTailCall`]). The function of
//! a cycle with the most registers can carry the whole cycle, and every
//! function of a cycle is landed on. Also refused: root tail calls in side
//! recordings or loops with another header, vararg callees, forwarded or
//! open arguments, a root frame with captured locals, and another closure of
//! the root chunk.
//!
//! # Function traces
//!
//! A function that calls itself directly is hot at its entry, not at a loop
//! header: the interpreter offers pc 0 of a frame that a self call just
//! pushed to the policy (see `JitState::call_landing`), under a site key of
//! its own ([`super::policy::FUNCTION_ENTRY_PC`]). A recording started there
//! ([`Recorder::for_function`]) records one activation from pc 0 to its
//! `Return`, which closes the trace with a
//! [`TraceTerminator::Return`]. Every parameter is an entry input, the other
//! guest registers start as nil, and the cells and globals are window slots
//! as in a loop trace.
//!
//! A `Call` or `CallOne` in the root frame of the frame's own closure, read
//! from a cell or global the function never writes, is a self call
//! ([`oximond::jit_ir::trace::TraceOp::SelfCall`]): the VM checks at entry that
//! those slots hold the frame's closure, so the call needs no identity guard.
//! Its arguments must have the entry's parameter types, so a nested
//! activation of the native code never mismatches. The recorder then sleeps
//! through the callee, which the interpreter runs, and wakes at the
//! instruction after the call, where the result register tells it the type
//! the function returns. Every return of the tree and every self call must
//! agree on it. The call's snapshot is the frame stopped before the call.
//!
//! A self call with an open result count (`f(a, f(b))`) leaves its value to
//! the self call or tail call that takes it next. Its type is what the
//! callee's frame returned, which the recorder sees at that frame's `Return`
//! while it sleeps. The taking call needs no depth check of its own: the self
//! call that made the value checked the same depth.
//!
//! A `TailCall` (or `TailCallOpen`) of the frame's own closure in the root
//! frame, `return f(args)`, closes the trace with a
//! [`TraceTerminator::TailCall`]: the frame starts over at pc 0 with the
//! arguments, as the interpreter's in-place tail call does, and native code
//! jumps back to the activation's start. The activation's tail call count is
//! a window slot ([`ExternalSlot::TailCalls`]) every snapshot names, so an
//! exit restores the count, and the name the tail call gives the frame.
//!
//! A function trace has no loop, no root activation, commits no effect (no
//! store into a table it did not make, no upvalue store), and returns exactly
//! one value: a number, a boolean, nil or a table. Anything else aborts with
//! [`TraceAbortReason::SelfCall`]. A table bound at entry from a parameter
//! must be passed on unchanged by every self call, since a nested activation
//! uses the bindings of the entered one.
//!
//! A function that passes tables it read out of other tables on to its calls
//! (`check (tree.left)`) has no such bindings: its table parameters are read
//! through their lanes, as a table loaded out of an array is
//! ([`FunctionSeed::lanes`]), so each activation reads the table it was given.
//! A field that holds a table reads as that table's lane
//! ([`oximond::jit_ir::trace::TraceOp::TableFieldLoad`] of a field of type
//! `Table`), and a field the table lacks reads as nil, from a row that says
//! the key is absent and no metatable stands behind the table (a field of
//! type `Nil`). Both leave through the load's guard when the table answers
//! another way: the guard is a branch like any other, and a side trace takes
//! the path of the other answer.
//!
//! A `NewTable` makes the table for real, where the constructor runs, by a
//! call of [`HelperId::TABLE_NEW`] with the shape the interpreter's
//! constructor gives it, and each field the constructor stores is a call of
//! [`HelperId::TABLE_INIT`]. No table stays virtual in a function trace: an
//! exit deep in a recursion would have to build one for every activation it
//! unwinds. The registers hold the table's lane, so an exit restores the
//! very table the interpreter would have made so far. The emitter stores the
//! fields of a table with a shape in the slots the helper gives it, where it
//! can, and the run counts them for the collector. A function that makes
//! closures records no constructor, since a table in a captured register is
//! no lane's to hold.
//!
//! # Loop parameters
//!
//! While recording, each register is either `Carried` (the iteration has not
//! written it, so it holds its value from the previous iteration or from
//! entry) or the result of a recorded operation. At close the recorder knows
//! which registers the iteration writes. It turns carried reads of written
//! registers into loop parameters, carried reads of unwritten registers into
//! invariant entry inputs, and carried register-map entries of unwritten
//! registers into `Unchanged`.
//!
//! # Side traces
//!
//! A recorder started with [`Recorder::new_side`] records the rest of an
//! iteration from the state a failed guard restored, for a trace that already
//! exists. It starts at the guard's resume pc instead of the loop header, takes
//! the root's window (guest registers and the slots after them) as it is, and
//! closes at the loop header as a [`TraceTerminator::Link`].
//!
//! A side shares the root's window and root bank. It may name an upvalue or
//! global the tree has not named yet, and pin a callee the tree has not
//! pinned (the tree's window and pinned list grow, see
//! [`oximond::jit_ir::trace::plan_tree`]). It may also add strings to the root
//! bank (an error value it builds): the tree takes them in when the side
//! joins, and the root's slots and every earlier side's keep their numbers.
//! It inlines a
//! call like the root, except that a callee may not touch an upvalue: a cell
//! the callee shares with the root frame is a promise only the root can
//! make. It has no loop of its own,
//! so no register is a loop parameter. Every live register the iteration
//! reads, or that a snapshot must restore, is an entry input instead, and its
//! snapshots never trust the interpreter register of one: the guard's parent
//! may have changed it since the interpreter last saw it. A register that
//! holds a string constant in the parent's state rides to the snapshots as a
//! root slot. What the loop needs at the header is a value for each of the
//! root's loop parameters, of the type the parameter has.
//!
//! When loops of the chunk enclose the root's (`SideSeed::outer`), the side
//! may leave the root's span and take the backward branch of such a loop, so
//! a side from the root's exit runs the rest of the outer iteration and links
//! back to the root's header through the code before the inner loop. That
//! recording still aborts at the backward branch of any other loop. The
//! enclosing headers it branches back to are reported in
//! [`ClosedTrace::crossed`], so the policy can see which loops the tree runs.
//!
//! # Guards and snapshots
//!
//! Every guard carries a complete register-map snapshot.
//!
//! - A conditional jump on a boolean guards the direction it took. The
//!   snapshot is the branch's post-state with the other successor, which is
//!   the same registers because a branch writes nothing.
//! - A `NumericForLoop` guards the sign of the step first, with a `Before`
//!   snapshot so the interpreter re-executes the latch. It then guards
//!   whether the loop continues, with an `After` snapshot that has the
//!   control register already advanced.
//!
//! Guards that nothing separates share a snapshot. A snapshot that equals the
//! one before it (phase, previous pc, registers, inlined frames, virtual
//! tables, counted instructions) is not pushed again, so the guards of one
//! instruction (a table read has a bounds guard and a type guard) and the
//! guards of two instructions that restore the same state use one entry, as
//! `LuaJIT`'s snapshots do. On the 512-block Mineclonia emerge this shares 148
//! of 648 snapshots in the recording of `get_structure_starts`.
//!
//! A call whose callee value the recording already compared with the same
//! pinned closure takes no guard and no snapshot
//! ([`Recorder::callee_guarded`]): the recording is one path, so the
//! earlier guard ran on every run that gets there, and an upvalue read from
//! a window slot the trace never writes is the slot's entry value at every
//! read. A write to the upvalue, or a callee read out of a table field
//! again, is another value and is guarded again. A Lua call in the root
//! frame or of a vararg function keeps its guard, since the callee's guards
//! or a region leave through its snapshot. Mineclonia's `load`-made ULL
//! helpers are upvalues of its RNG step and call the bit library through
//! upvalues too: in the first recording of `get_structure_starts`
//! (structures.lua:744) on the 512-block emerge, 426 of the guards at the
//! snapshot cap were such compares and 64 of 80 Lua callee guards repeated
//! one (2026-10-31, a count by guard kind at the cap).
//!
//! Before the overwrites that loop hit `maxsnap` (500, the `LuaJIT` default)
//! on the emerge.
//! Without the repeated callee guards its recording reaches 3331 guest
//! instructions at the cap instead of 2392, and the 499 snapshots are 238
//! array reads and 196 array stores of the ULL pairs and 66 branch guards.
//! A store needs a new snapshot for the next guard (`LuaJIT` sets `needsnap`
//! after one too), so no sharing of guard snapshots gets those recordings,
//! which run a frequency reducer or a chunk test that passes, under the
//! cap: all three end there (`random.lua:1189`, `=(load):51`) and the loop
//! is rejected. One iteration that runs a structure's `create_start` needs
//! more than the 4000 guest instructions of `maxrecord` as well. `LuaJIT`'s
//! default limits are the same, so it would stop recording that loop too
//! and trace the hot callees as functions; this VM has function traces only
//! for direct self calls. The repro
//! (`target/flow-luanti-native/repros/structures-744.lua`) records on a
//! failing chunk test of about 1090 guest instructions, which fits.
//!
//! A read of an array slot that the recording already read or stored in
//! this iteration is a move of the known value, with no guard, and an
//! in-place store into such a slot of the value's type is an overwrite
//! with no guard and no snapshot (`slot_facts`). Reads alone forwarded 144
//! reads in each `get_structure_starts` recording, which still reached the
//! cap at 3360 guest instructions: most of those reads already shared the
//! snapshot before their inlined call, and the stores' snapshots were left.
//! The cap also counted the snapshot `hoist_callee` takes before the call
//! that makes open results for a native (`tonumber (band (value, mask))`
//! twice per helper), which a guard uses only when the native's region
//! opens and a guard in it leaves; the cap now counts only snapshots some
//! guard leaves through. On the repro a failing chunk test went from 352
//! snapshots to 102 guarded ones for 1089 guest instructions; on the emerge
//! the root of loop@744 records 1782 guest instructions with 132 and
//! compiles, with a side of 3421 (2026-11-01). It then retires on the
//! callee guard of `placement.chunk_test` (below), not on a limit.
//!
//! Guards inside an inlined call that leave through the last snapshot of
//! the same frames when only register work came between ("rewound" guards,
//! one snapshot for the guards of `addull` before its first store) were
//! tried and dropped (2026-10-31): they took one failing chunk test from
//! 352 to 201-250 snapshots, but the emerge's recordings still hit the cap
//! at the same instruction, and the exits inside inlined calls grew (the
//! `mineclonia/ull_kernel` root: 41 exits, 24 inside inlined calls -> 96
//! and 90), so `ull_kernel` went from 29.4 M to 68.6 M instructions and
//! `module_calls` from 26.6 M to 63.4 M. So was giving up a side whose tree
//! is over the cache's caps on its own, instead of counting the compile as
//! refused for room: the repro went from 28.6 G to 30.7 G instructions.
//!
//! Where the loop compiles (the repro), every set calls another closure of
//! one prototype through `placement.chunk_test`, so the root leaves at that
//! callee guard on almost every entry and retires within its first
//! productivity windows ("23268 iterations in 22607 entries, 3 sides; exit
//! 333 (member 1, guard 0 Callee, frame 2 pc 9) 14029x"). `LuaJIT` would hit
//! the same callee guard; it grows one side per closure since its roots
//! never retire, and here the root with three sides of about a thousand
//! guest instructions holds 14.3 MB of the 16 MB metadata cap.
//!
//! Two more ways to compile that loop were tried and dropped (2026-10-31,
//! repro `target/flow-luanti-native/repros/structures-744.lua`, 9216 chunks
//! of the 15 overworld structure sets: interpreter 0.82 s, tracing 1.0 s).
//! The ULL helpers cost one snapshot per three guest instructions: a chunk
//! test that fails is 1090 guest instructions and 352 snapshots, one that
//! passes runs past 500. Ending the recording at a guard that always fails
//! at 6/8 of a cap, with a side for the rest of the iteration, compiled the
//! root, but the sides did not fit the tree ("a cell alias stands for a name
//! with no window slot", "the tree has no room") and every entry ran one
//! iteration (1.4 to 1.8 s). With the snapshot cap at 2000 the root compiles,
//! and then `placement.chunk_test`, a different closure of one prototype for
//! each set, fails the callee guard every iteration. Guarding such a call on
//! the prototype once it made three closures (`PROTO_CLC_POLY`) fixed that,
//! but the root still runs one iteration per entry (the sets take other
//! branches) and retires within one productivity window, before any guard
//! reaches `SIDE_EXIT_THRESHOLD`; and on a loop over five closures of one
//! prototype it cost 1.87 G instructions against 0.97 G for one side per
//! closure. The emerge stayed at 293.6 and 302.5 G (304.2 G before).
//!
//! Branches on a nil or float register need no guard: the type is already
//! fixed by the entry contract. A carried register tested this way still
//! joins the contract as an input, so the type is checked at entry.

use super::debug_env;
use super::helpers::{
    STRING_NUMBER_FLOAT_MODE, STRING_NUMBER_LUA53_MODE, STRING_NUMBER_METAMETHOD_MODE,
    method_identity, string_number,
};
use super::helpers::{Walk, walk_handlers};
use super::nest::{self, LoopSpan};
use super::stats::TraceAbortReason;
use crate::bytecode::{Chunk, Constant, Instruction, OpCode, UpvalueSource};
use crate::compat::{CompatibilityProfile, FloatArith, FloatModulo};
use crate::jit_ir::trace::{
    ArithmeticOp, BindingKind, BoxedSteps, CellAlias, CellSource, Entry, ExternalSlot, FrameId,
    FunctionTrace, HelperEffects, HelperId, HelperType, HostFunction, InlinedFrame, LoopParamInit,
    LoopRole, MAX_HOST_ARGS, MAX_INDEX_HOPS, MAX_NEW_TABLE_KEYS, MAX_TRACE_FRAME_DEPTH,
    NativeBinary, NativeUnary, NumberRegister, NumericOp, PINNED_BUILTIN_COROUTINE_RESUME,
    PINNED_BUILTIN_COROUTINE_YIELD, PINNED_BUILTIN_DEBUG_TRACEBACK, PINNED_BUILTIN_IPAIRS,
    PINNED_BUILTIN_IPAIRS_ITERATOR, PINNED_BUILTIN_NEXT, PINNED_BUILTIN_PAIRS,
    PINNED_BUILTIN_PCALL, PINNED_BUILTIN_STRING_FORMAT, PINNED_BUILTIN_STRING_GSUB,
    PINNED_BUILTIN_TOSTRING, PINNED_BUILTIN_XPCALL, Protected, REGISTER_RUN, ROOT_STRING_LANE_BASE,
    RegisterEntry, RegisterMap, RegisterUse, RootActivation, Snapshot, SnapshotFrame,
    SnapshotPhase, TABLE_ADDRESS_KEY, TableBinding, TableField, TailSite, TraceBuilder,
    TraceFrameInfo, TraceId, TraceIr, TraceOp, TraceScalar, TraceSite, TraceTerminator, TraceType,
    ValueId, VirtualAlloc, VirtualId, VirtualKey, VirtualTable, WideConversion,
    loop_types_are_stable, pinned_builtin_identity, pinned_builtin_kind, pinned_native_identity,
    pinned_native_index,
};
use crate::jit_ir::{Comparison, RegisterSet, observable_registers, register_set_contains};
use crate::stdlib::substring_range;
use crate::vm::arith::{
    call_metamethod_key, integer_to_number, numeric_scalar_result, register_form,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

mod inline;
mod inline_bit;
mod inline_caught;
mod inline_meta;
mod inline_next;
mod inline_operator;
mod inline_pattern;
mod inline_protected;
mod inline_resume;
mod inline_slice;
mod inline_table;
mod inline_text;
mod inline_vmcall;
mod linked;
mod number;
mod select;
mod slot_facts;
use inline_caught::CaughtCheck;
use inline_protected::ProtectedTarget;
pub(crate) use linked::{CallerSeed, NO_CLOSURE};

/// Per-chunk liveness, computed on first use and shared by every recording of
/// the chunk, so a large chunk pays the fixpoint once.
pub(crate) type SharedLiveness = Arc<OnceLock<Vec<RegisterSet>>>;
use crate::runtime::{
    BuiltinFunction, CellHandle, ClosureHandle, DirectNativeKind, NativeFunctionHandle, Runtime,
    TableHandle, ThreadHandle, Value,
};
use std::sync::Arc;

/// The guest instructions a call through the VM counts for, when the policy
/// weighs the work of a loop.
const HOST_CALL_GUEST_WORK: u32 = 20;
/// The register that holds the callee of a call instruction: `B` of a
/// `CallOne` (whose `A` is the destination), `A` of the other calls. `None`
/// for any other instruction.
pub(super) fn callee_register(instruction: Instruction) -> Option<u8> {
    match instruction.opcode() {
        Ok(OpCode::CallOne) => Some(instruction.b()),
        Ok(
            OpCode::Call
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen,
        ) => Some(instruction.a()),
        _ => None,
    }
}

/// Most guest instructions one trace may record (`LuaJIT`'s `maxrecord`).
pub(crate) const MAX_TRACE_OPS: usize = 4_000;
/// How far past the read of a function [`Recorder::reads_poly_callee`]
/// looks for the call that takes it.
const MAX_POLY_CALLEE_LEAD: usize = 8;
/// Backward branches of `string.gmatch` loops inside the recorded loop that a
/// recording follows, which unrolls those loops.
pub(crate) const MAX_UNROLLED_BACKEDGES: usize = 10;
/// Most iterations a numeric for loop inside the recorded loop may have left
/// at its latch for the recording to unroll it (the `loopunroll` of `LuaJIT`).
pub(crate) const MAX_UNROLLED_LOOP_TRIPS: f64 = 15.0;
/// Backward branches of numeric for loops inside the recorded loop that one
/// recording follows, over all such loops. The bodies repeat, so this bounds
/// the size of the trace as well as the guards the unrolled loops need.
pub(crate) const MAX_UNROLLED_LOOP_BACKEDGES: usize = 24;
/// Backward jumps of `while` loops that one recording follows, over all such
/// loops. A `while` loop has no trip count to look at, and a root's entry
/// costs about as much as eight trips of a small body (`TRACE_MIN_ITERATIONS_PER_ENTRY`),
/// so a loop that runs longer than this pays as a root of its own. Measured
/// on the escape-time kernel of `a_float_while_loop_with_two_exits_keeps_its_covered_outer_loop`
/// (24 trips per pixel): unrolled to the numeric limit it replaces the
/// outer loop's trace with one that leaves at the loop's test on most
/// pixels.
pub(crate) const MAX_UNROLLED_WHILE_BACKEDGES: usize = 6;
/// Guest instructions a recording that unrolled loops may hold. A nest that
/// needs more is not unrolled: its trace would take too long to compile for
/// a short run to repay, and the loops inside keep their own traces.
pub(crate) const MAX_UNROLLED_GUEST_INSTRUCTIONS: usize = 160;
/// [`MAX_UNROLLED_GUEST_INSTRUCTIONS`] for a side. A side that calls a
/// function with a ten-trip loop of 17 instructions (`digits` in
/// `mineclonia/open_results`) needs about 200. With 160 its guard kept no
/// side, and every failure cost an exit, an entry of the short loop's own
/// trace and an entry back, some 7 000 instructions (34.5M for the benchmark
/// against 14.7M with the side). A root that does not fit still has the
/// loops inside as traces of their own; a side that does not fit leaves its
/// guard an exit for good. A recording that still hits the cap wastes what
/// it recorded, so the cap stays near what that case needs. Roots keep 160:
/// with 240 the nest of `trace_return_nests.rs` unrolled into a root that
/// retired, and ran interpreted under the integer profiles.
pub(crate) const MAX_UNROLLED_SIDE_GUEST_INSTRUCTIONS: usize = 240;
/// Guest instructions an inlined call may run before a guard inside it
/// stops leaving through the snapshot before the call (see
/// [`Recorder::starts_call_over`]). A guard in a small callee keeps starting
/// the call over, which needs no frames rebuilt; one deep in a long call
/// (Mineclonia's density function inlines 27 calls) leaves where it is, and
/// its side starts there.
pub(crate) const BEFORE_CALL_MAX_PREFIX: u32 = 64;
/// Guest instructions an outer side must record past the loops around the
/// root before it may end at the frame's return (see [`Recorder::exit_at`]).
const MIN_BEYOND_WORK: u32 = 24;

/// The guest instructions a root recording needs for a trace to make its host
/// C calls itself. A call from a trace costs about 500 instructions more than
/// the interpreter's own (the state in front of the call is written back to
/// the frame, and the trace reads its bindings again afterwards), and the
/// interpreter spends about 23 instructions per guest instruction of a plain
/// arithmetic body, so a loop pays from about 22 (see "Calling host
/// functions" in the module documentation).
const VMCALL_MIN_GUEST: u32 = 20;
/// Iterations a numeric for loop beside the root's must have left at its
/// latch for an outer side to leave them to a loop side (see
/// [`SideSeed::loop_sides`]). The side records the first trip itself and the
/// loop side one more, so a loop with fewer is unrolled instead.
pub(crate) const MIN_LOOP_SIDE_TRIPS: f64 = 3.0;
/// Most IR operations one trace may hold. A numeric-for latch records five.
/// Twice `maxrecord` by default.
pub(crate) const MAX_TRACE_IR_OPS: usize = 8_000;
/// Most snapshots that guards of one trace leave through (`maxsnap`).
/// `LuaJIT`'s default is 500. Raising it to 2000 let `get_structure_starts`
/// (structures.lua:743) record past the cap, but the 512-block Mineclonia
/// emerge did not move (317.0 G instructions against 315.4 G and 325.2 G
/// for 500 on the same build: the run-to-run spread is larger than the
/// difference), so the cap stays where `LuaJIT` has it.
pub(crate) const MAX_TRACE_SNAPSHOTS: usize = 500;
/// Most upvalue cells and globals one trace may name beyond its registers.
pub(crate) const MAX_TRACE_EXTERNALS: usize = 256;
/// Most string constants one trace may keep in its root bank.
pub(crate) const MAX_TRACE_ROOTS: usize = 256;
/// Most distinct callables (closures, natives, builtins) one trace may pin by
/// identity.
pub(crate) const MAX_TRACE_PINNED: usize = 128;
/// Most fields of run-time tables one trace may name.
pub(crate) const MAX_TRACE_FIELDS: usize = 256;
/// Most table constructors one trace may keep virtual.
pub(crate) const MAX_TRACE_VIRTUALS: usize = 64;
/// Most entries (array slots and fields) one virtual table may hold.
pub(crate) const MAX_VIRTUAL_ENTRIES: usize = 64;
/// Most calls one escape of virtual tables may make: a table per table and a
/// store per entry.
const MAX_MATERIALIZE_CALLS: usize = 192;
/// Most calls one trace may inline, nested or not.
pub(crate) const MAX_TRACE_INLINED_CALLS: usize = 128;
/// The caps of a recording that `jit.opt.start` sets per VM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RecordLimits {
    /// Guest instructions (`maxrecord`).
    pub(crate) guest_ops: usize,
    /// IR operations: twice `maxrecord`.
    pub(crate) ir_ops: usize,
    /// Snapshots (`maxsnap`).
    pub(crate) snapshots: usize,
    /// Whether a trace may call a host C function itself (see
    /// `inline_vmcall`), or the recording aborts at the call.
    pub(crate) vmcall: bool,
    /// Guest instructions a trace needs to make host calls itself
    /// ([`VMCALL_MIN_GUEST`]).
    pub(crate) vmcall_work: u32,
}

impl RecordLimits {
    /// The limits for a `maxrecord` of `ops`.
    pub(crate) const fn with_record(self, ops: usize) -> Self {
        Self {
            guest_ops: ops,
            ir_ops: ops.saturating_mul(2),
            ..self
        }
    }
}

impl Default for RecordLimits {
    fn default() -> Self {
        Self {
            guest_ops: MAX_TRACE_OPS,
            ir_ops: MAX_TRACE_IR_OPS,
            snapshots: MAX_TRACE_SNAPSHOTS,
            vmcall: !crate::jit::debug_env::no_vmcall(),
            vmcall_work: crate::jit::debug_env::vmcall_min_guest().unwrap_or(VMCALL_MIN_GUEST),
        }
    }
}

/// The abort for a recording that hit the cap `what`; `OXIMOND_JIT_TRACE` names
/// the cap next to the abort.
pub(super) fn limit(what: &'static str) -> TraceAbortReason {
    if debug_env::trace() {
        eprintln!("jit-trace: recording hit the {what} cap");
    }
    TraceAbortReason::LimitExceeded
}
/// Most moves between a call with open results and the call that takes them.
const MAX_OPEN_MOVES: usize = 8;

/// What a side recording takes from the tree it continues.
#[derive(Clone, Debug)]
// The seed's mode flags are a deliberate flat shape on this record.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct SideSeed {
    /// For a side of a counted tree: the counted instructions the path ran
    /// from the start of the iteration to the guard this side continues.
    pub(crate) counted: Option<u32>,
    /// The pc the recording starts at: where the failed guard resumes.
    pub(crate) start_pc: usize,
    /// The root frame's `last_interpreted_pc` after the exit restored it.
    pub(crate) last: Option<usize>,
    /// The window slots beyond the guest registers. A side can use these and
    /// no others: a new one would need an entry load the root does not have.
    pub(crate) externals: Vec<ExternalSlot>,
    /// The type each of `externals` has at the tree's entry, where it has an
    /// entry input.
    pub(crate) external_types: Vec<Option<TraceType>>,
    /// The strings of the root bank, by slot.
    pub(crate) root_constants: Vec<Arc<[u8]>>,
    /// Registers that hold a string constant in the parent's state at the
    /// guard, by root slot. The interpreter register still holds the older
    /// value until an exit, so a side must name the constant itself.
    pub(crate) roots: Vec<(usize, u32)>,
    /// The loop parameters of the root: register and type. A side that links
    /// back must hand the loop a value of that type for each.
    pub(crate) params: Vec<(u16, TraceType)>,
    /// The control register of a `next` loop whose control value the tree
    /// keeps in a lane of the run's bank (see `TraceIr::boxed_control`).
    pub(crate) boxed: Option<usize>,
    /// The tables the tree binds at entry, in the tree's order. A side names
    /// a table by its position in this list and may add bindings after them.
    pub(crate) bindings: Vec<TableBinding>,
    /// The fields of run-time tables the tree names, in the tree's order. A
    /// side names a field by its position and may add fields after them.
    pub(crate) fields: Vec<TableField>,
    /// The closures the root pins, by identity, and the cells it holds under
    /// two names. A side starts from both and may extend them: it pins the
    /// closures of its own calls and names cells the root met once. The tree
    /// takes the extended lists over when the side joins.
    pub(crate) pinned: Vec<u64>,
    pub(crate) aliases: Vec<CellAlias>,
    /// Whether loops enclose the root's: the side may leave the root's span
    /// and run through them before it reaches the header again.
    pub(crate) outer: bool,
    /// Whether the outer side may also run through short loops beside the
    /// root's and end at the frame's return (see [`Recorder::exit_at`]).
    pub(crate) beyond: bool,
    /// The header of the enclosing loop whose backward branch the failed
    /// guard itself took, if it did: the side starts at that header, so the
    /// recording never sees the branch.
    pub(crate) entered: Option<usize>,
    /// The registers that hold a borrowed string in the parent's state at the
    /// guard. The side takes the slot's address for them, as the parent has
    /// it, and may use it until it ends borrows itself.
    pub(crate) borrowed: Vec<usize>,
    /// The registers that hold a table the parent met at run time in its
    /// state at the guard (read out of another table, or given by a lookup
    /// call). The side reads such a table by lane, as the parent does: the
    /// parent rewrites the register whenever it reads the table, so an entry
    /// binding of it would only be demoted by the plan (see
    /// [`oximond::jit_ir::trace::plan_tree`]).
    pub(crate) lanes: Vec<usize>,
    /// The registers the side may take as inputs of type `Any` (see
    /// [`Recorder::with_lazy`]): the root has such an input for each, and
    /// the parent's state at the guard leaves the register as the run's
    /// entry found it.
    pub(crate) lazy: Vec<u16>,
    /// For a side of a function trace: the function's parameter and result
    /// types and the closure whose frame the side runs in. The side closes at
    /// the activation's `Return`.
    pub(crate) function: Option<FunctionSeed>,
    /// Whether a member of the tree reads and writes a callee's cells
    /// through the live closure ([`InlinedFrame::dynamic`]), and whether a
    /// member holds a captured local of the root frame in a register. Both
    /// at once would leave the register stale after a write through the
    /// closure, so a side refuses the second kind of use when the tree has
    /// the first, and the other way round.
    pub(crate) dynamic_cells: bool,
    pub(crate) captured_busy: bool,
    /// Whether a member writes a callee's cell through the live closure
    /// ([`HelperId::CELL_SET`]), and whether a member writes a captured
    /// local of the root frame. A frame that only reads cells through the
    /// closure reads the interpreter's register, which is the variable while
    /// no member writes it.
    pub(crate) dynamic_cell_writes: bool,
    pub(crate) captured_written: bool,
    /// Whether the side is a loop side: the loop (beside the root's, or
    /// inside an outer root's) that its parent's terminal exit stands in
    /// front of. It starts
    /// at the loop's header and closes where it comes round again, like a
    /// root, but its registers come from the parent's state like any side's.
    pub(crate) looping: bool,
    /// Whether an outer side that goes beyond the nest may end in front of a
    /// loop (a numeric for, a `while` or a generic for) and leave its
    /// remaining trips to a loop side, instead of unrolling them.
    pub(crate) loop_sides: bool,
    /// Whether the tree's root ends in front of a loop inside its own (see
    /// "Outer roots" in the module docs of `trace`): its inner loops are loop
    /// sides of the tree, so a side that meets the backward branch of one
    /// ends in front of it too.
    pub(crate) inner_loops: bool,
    /// The loop sides among the guard's parent and its ancestors, innermost
    /// first. A side that comes round to the header of one closes with
    /// [`TraceTerminator::LinkLoop`] instead of ending in front of a copy of
    /// the loop.
    pub(crate) loop_headers: Vec<SeedLoop>,
    /// The frame that called the root frame, when a side that leaves the
    /// root frame by its return may run on in it (see
    /// [`CallerSeed`]).
    pub(crate) caller: Option<CallerSeed>,
    /// Whether the side ends at a `Return` of the root frame with an exit
    /// that resumes the interpreter there (see
    /// [`Recorder::returns_to_interpreter`]).
    pub(crate) return_exit: bool,
    /// The inlined calls the guard failed inside, outermost first, which the
    /// side starts inside (see "Sides from exits inside inlined frames" in
    /// the module docs). Empty for a side that starts in the root frame.
    pub(crate) frames: Vec<SeedFrame>,
    /// For a side with `frames`: the root frame's registers after the exit
    /// restored them. The first hook shows the innermost frame, so the
    /// root's entry kinds come from here.
    pub(crate) root_registers: Vec<Value>,
    /// For a side with `frames`: the root frame's cells and environment,
    /// which the root's window slots read. The first hook shows the
    /// innermost callee's. The root frame is live below the frames while
    /// the side starts, which is the only time the recorder reads these.
    pub(crate) root_cells: Vec<CellHandle>,
    pub(crate) root_environment: Option<TableHandle>,
}

/// A loop side among the ancestors of a side recording (see
/// [`SideSeed::loop_headers`]).
#[derive(Clone, Debug)]
pub(crate) struct SeedLoop {
    /// The loop side's index among the tree's sides.
    pub(crate) side: usize,
    /// The loop's header and the pc of its closing branch.
    pub(crate) header: usize,
    pub(crate) latch: usize,
    /// The loop side's parameters: register and type. A link hands the loop
    /// a value of that type for each.
    pub(crate) params: Vec<(u16, TraceType)>,
    /// The registers the loop side reads from its entry state without
    /// carrying them, which a link may not have written.
    pub(crate) invariant: Vec<u16>,
    /// The registers the sides between the loop side and the recording's
    /// parent (that one included) wrote.
    pub(crate) written: Vec<u16>,
}

/// The closing branch of a loop a recording may leave to a loop side (see
/// [`Recorder::wants_loop_side_at`]).
#[derive(Clone, Copy, Debug)]
enum LoopKind {
    /// A numeric for latch with `remaining` trips after this one, and its
    /// counters.
    Numeric {
        remaining: f64,
        counters: [Operand; 3],
    },
    /// The backward jump of a `while` loop.
    While,
    /// A generic for latch whose step gave an entry.
    Generic,
}

/// An inlined call a side recording starts inside: the frame the parent's
/// exit would have pushed for the interpreter, which the recorder rebuilds as
/// an inlined frame of its own instead.
#[derive(Clone, Debug)]
pub(crate) struct SeedFrame {
    /// The call as the parent inlined it, with `closure` a lane of
    /// [`SideSeed::pinned`] and `parent` the side's frame below this one.
    pub(crate) info: InlinedFrame,
    /// The chunk the frame runs.
    pub(crate) chunk: Arc<Chunk>,
    /// The frame's `last_interpreted_pc`: the call that made the next frame,
    /// or for the innermost frame what the exit restored.
    pub(crate) last: Option<usize>,
    /// What each register holds in the parent's state at the guard.
    pub(crate) registers: Vec<SeedRegister>,
}

/// A register of a [`SeedFrame`].
#[derive(Clone, Debug)]
pub(crate) enum SeedRegister {
    /// A typed value, which the side takes from its parent as a frame input.
    /// The value is the one the exit restored: its kind types the input.
    Live(Value),
    /// A string constant of the tree's root bank, by slot.
    Root(u32),
    /// The element of an `ipairs` loop in a call that a loop side of the
    /// loop reads again at its header (see
    /// [`Recorder::reload_ipairs_element`]): the parent holds a borrow
    /// there, which is no frame input. The side must write it before it
    /// reads it.
    Reloaded,
    /// A register the frame no longer reads: the side starts at a `return`
    /// that does not return it, in a function with no to-be-closed local,
    /// and holds nil there. A borrowed string the parent holds in it is no
    /// frame input (the element of a scan that matched, before `return i`).
    Dead,
}

/// The root frame of a side that starts inside inlined calls, as a hook in
/// it would show it: the window slots of the root read its cells, its
/// environment and its chunk's constants, not the innermost callee's.
struct RootView<'a> {
    chunk: Arc<Chunk>,
    registers: &'a [Value],
    surroundings: Surroundings<'a>,
}

impl<'a> RootView<'a> {
    fn of(
        at: &HookPoint<'a>,
        seed: &'a SideSeed,
        chunk: &Arc<Chunk>,
    ) -> Result<Self, TraceAbortReason> {
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let environment = seed.root_environment.ok_or(TraceAbortReason::Divergence)?;
        Ok(Self {
            chunk: Arc::clone(chunk),
            registers: &seed.root_registers,
            surroundings: Surroundings {
                upvalues: &seed.root_cells,
                environment,
                runtime,
                varargs: &[],
            },
        })
    }

    /// The hook point `at` as the root frame would show it.
    fn at<'b>(&'b self, at: &HookPoint<'b>) -> HookPoint<'b> {
        HookPoint {
            chunk: &self.chunk,
            depth: at.depth,
            top_level: at.top_level,
            pc: at.pc,
            registers: self.registers,
            open: &[],
            coroutine: at.coroutine,
            hooks_active: at.hooks_active,
            source: at.source,
            surroundings: Some(self.surroundings),
        }
    }
}

/// What a function-entry recording knows about its function (see "Function
/// traces" in the module docs).
#[derive(Clone, Debug)]
pub(crate) struct FunctionSeed {
    /// The closure the recorded frame runs: the callee of every self call.
    pub(crate) closure: ClosureHandle,
    /// The type of each parameter at entry. A root recording fills it at its
    /// first hook; a side takes the root's.
    pub(crate) parameters: Vec<TraceType>,
    /// The type every return gives, once a return or a self call showed it.
    pub(crate) result: Option<TraceType>,
    /// Whether the function passes tables it read out of other tables on to
    /// its calls (`check (tree.left)`). Its table parameters are then no
    /// entry bindings, which every activation would share: a field of one is
    /// read through the table's lane, so each activation reads its own table.
    pub(crate) lanes: bool,
}

/// A self call the interpreter is making while the recorder sleeps: the
/// operation that stands for it, and where the caller resumes.
#[derive(Clone, Copy, Debug)]
struct SelfCallWait {
    op: u32,
    /// The caller's call depth.
    depth: usize,
    /// The call instruction.
    pc: usize,
    /// The call's result window. No results is an open count: the value
    /// goes on to the self call or tail call that takes it next.
    base: usize,
    results: usize,
    /// What the callee's frame returned last, seen at its `Return` while the
    /// recorder slept: the value an open result count leaves in no register.
    /// `None` when that was no single value.
    returned: Option<Kind>,
}

/// Which interpreter loop called the hook.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HookSource {
    /// The fast interpreter loop.
    Fast,
    /// The general dispatch loop.
    General,
}

/// What the recorder may read besides registers: the running closure's
/// upvalue cells and the global environment. It only reads. The interpreter
/// executes every operation.
#[derive(Clone, Copy)]
pub(crate) struct Surroundings<'a> {
    pub(crate) upvalues: &'a [CellHandle],
    pub(crate) environment: TableHandle,
    pub(crate) runtime: &'a Runtime,
    /// The running frame's extra arguments (what `...` holds).
    pub(crate) varargs: &'a [Value],
}

/// What the interpreter shows the recorder before an instruction.
pub(crate) struct HookPoint<'a> {
    pub(crate) chunk: &'a Arc<Chunk>,
    /// The call-stack depth.
    pub(crate) depth: usize,
    /// Whether the running frame and every frame below it have no closure:
    /// the frame runs a chunk the host executed ([`oximond::vm::Vm::execute`]),
    /// behind at most the VM's own stub frames, which runs once. A Lua
    /// function has a closure, also when a host calls it (`lua_pcall`), and
    /// may run any number of times.
    pub(crate) top_level: bool,
    /// The pc of the instruction about to run.
    pub(crate) pc: usize,
    /// The running frame's registers. At least the chunk's register count.
    pub(crate) registers: &'a [Value],
    /// The open results the instruction before this one left in the running
    /// frame (for a `CallOpen` to take), empty when there are none.
    pub(crate) open: &'a [Value],
    pub(crate) coroutine: Option<ThreadHandle>,
    pub(crate) hooks_active: bool,
    pub(crate) source: HookSource,
    /// Upvalues and globals. `None` records no operation that needs them.
    pub(crate) surroundings: Option<Surroundings<'a>>,
}

/// What one hook call did.
pub(crate) enum Step {
    /// The hook repeats an instruction the other loop already recorded.
    Skipped,
    /// The instruction is recorded.
    Recorded,
    /// The loop closed. The trace verified.
    Closed(Box<ClosedTrace>),
    /// A scouting recorder saw the interpreter take the failure edge of a
    /// guard it watches: the side for that guard starts here, at the
    /// instruction just hooked.
    Wake(WakeCandidate),
    /// The recording stopped.
    Abort(TraceAbortReason),
}

/// A guard of a queued side that a scouting recorder watches: the guard's
/// instruction, where its failure resumes, and its origin. A `beyond` guard
/// is one that leaves the nest.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WakeCandidate {
    pub(crate) guard_pc: usize,
    pub(crate) successor: usize,
    pub(crate) origin: u32,
    pub(crate) beyond: bool,
    /// Whether the guard is the root's own: the side starts where the root's
    /// loop exits.
    pub(crate) root: bool,
}

/// What a scouting recorder watches (see [`Recorder::scouting`]).
#[derive(Clone, Debug)]
struct Scout {
    /// The root's closing branch and the root's header: the loop exits when
    /// the branch falls through to the next pc, and the side the scout follows
    /// ends when the loop comes round to its header again.
    latch: usize,
    header: usize,
    candidates: Vec<WakeCandidate>,
    /// Whether the interpreter is on the path of the queued side, which
    /// starts where the root loop exits.
    following: bool,
    /// The pc of the instruction hooked before this one, in the frame.
    previous: Option<usize>,
    budget: u32,
}

impl Scout {
    /// Observes the instruction at `pc` about to run. Returns the guard of
    /// the side the interpreter left the side's path by, when that is a
    /// guard that leaves the nest.
    fn observe(&mut self, pc: usize) -> Option<WakeCandidate> {
        let previous = self.previous.replace(pc);
        if self.following {
            // The side's path ended where it links back to the root's loop.
            if pc == self.header {
                self.following = false;
                return None;
            }
            let candidate = self.candidates.iter().find(|candidate| {
                Some(candidate.guard_pc) == previous && candidate.successor == pc
            });
            if let Some(candidate) = candidate.copied() {
                // Whichever guard the interpreter failed, it is off the
                // side's path now.
                self.following = false;
                return candidate.beyond.then_some(candidate);
            }
        }
        if previous == Some(self.latch) && pc == self.latch + 1 {
            self.following = true;
        }
        None
    }
}

/// The nest around a short numeric for loop whose root just closed, read from
/// the counters of the loops around it (see `Recorder::nest_plan`): the
/// recording that would take the short loops into one trace starts at
/// `target`, the innermost loop around the root that is not short, and
/// unrolls everything between.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NestPlan {
    /// The loop that same recording would start from.
    pub(crate) target: LoopSpan,
    /// The iterations `target` has left.
    pub(crate) target_remaining: f64,
    /// An upper estimate of the trips one run of the root's loop has.
    pub(crate) root_trips: f64,
    /// Guest instructions one iteration of `target` has with every loop
    /// below it unrolled.
    pub(crate) guest_instructions: f64,
    /// Backward branches that recording follows.
    pub(crate) backedges: f64,
}

/// A recorded loop.
#[derive(Debug)]
pub(crate) struct ClosedTrace {
    /// Guest instructions in one iteration.
    pub(crate) guest_instructions: u32,
    /// Array reads answered from slot facts (`slot_facts`).
    pub(crate) slot_reads_forwarded: u32,
    /// In-place array stores proved by slot facts (`slot_facts`).
    pub(crate) slot_overwrites: u32,
    pub(crate) header_pc: usize,
    /// The headers of enclosing loops an outer side trace runs through: the
    /// loops whose backward branch it records. While it runs, the
    /// interpreter never sees those loops' backedges.
    pub(crate) crossed: Vec<usize>,
    /// For a root trace closed at a numeric for loop's latch, the iterations
    /// left in the run the recording saw.
    pub(crate) remaining_trips: Option<f64>,
    /// For the same loop, an upper estimate of the iterations one run of it
    /// has in all (see `run_trips`), when the counters bound it.
    pub(crate) run_trips: Option<f64>,
    /// For a root trace in the bottom frame of the main thread whose loop sits
    /// inside numeric for loops only: the iterations of the loop left in the
    /// program, estimated from the loops' counters (see
    /// `Recorder::remaining_work`).
    pub(crate) remaining_work: Option<f64>,
    /// For the same root: what a recording of the whole nest from a longer
    /// loop around it would be, when the loops between are short enough to
    /// unroll (see [`NestPlan`]).
    pub(crate) nest_plan: Option<NestPlan>,
    /// For an outer side that ended in front of a numeric for loop (see
    /// [`SideSeed::loop_sides`]): the loop's header and its latch.
    pub(crate) entered_loop: Option<(usize, usize)>,
    pub(crate) ir: TraceIr,
}

/// The kind of value a register held when the recording began.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Nil,
    Boolean,
    Integer,
    Float,
    /// `LuaJIT`'s `int64_t` cdata.
    Int64,
    /// `LuaJIT`'s `uint64_t` cdata.
    UInt64,
    Table,
    /// A Lua closure or a native function.
    Closure,
    /// A coroutine, known by identity.
    Thread,
    /// A string, known by identity.
    String,
    /// A string the parent trace of a side holds as a borrow of a table
    /// slot: the side gets its address, not a lane. The interpreter holds a
    /// plain string, so it only ever matches a live string. Only a side
    /// recording's entry has it.
    Borrowed,
    /// Anything else: userdata, or an integer in a profile without them.
    Other,
}

impl Kind {
    // The coroutine-wrapper and plain-callable arms are distinct value
    // kinds that happen to share the `Closure` answer.
    #[allow(clippy::match_same_arms)]
    fn of(value: &Value, integers: bool) -> Self {
        match value {
            Value::Nil => Self::Nil,
            Value::Boolean(_) => Self::Boolean,
            Value::Integer(_) if integers => Self::Integer,
            Value::Number(_) => Self::Float,
            Value::Int64(_) => Self::Int64,
            Value::UInt64(_) => Self::UInt64,
            Value::Table(_) => Self::Table,
            Value::String(_) => Self::String,
            Value::Thread(_) => Self::Thread,
            // So is a function `coroutine.wrap` made. The call checks at run
            // time that the callable is one (`HelperId::coroutine_wrap`).
            Value::CoroutineWrapper(_) => Self::Closure,
            // A native function is a callable like a closure: the trace
            // only compares it by identity with the one it pinned.
            Value::Closure(_)
            | Value::NativeFunction(_)
            | Value::GmatchIterator(_)
            | Value::BuiltinFunction(
                BuiltinFunction::IPairsIterator
                | BuiltinFunction::Next
                | BuiltinFunction::Pairs
                | BuiltinFunction::IPairs
                | BuiltinFunction::ToString
                | BuiltinFunction::StringFormat
                | BuiltinFunction::StringGsub
                | BuiltinFunction::ProtectedCall
                | BuiltinFunction::ExtendedProtectedCall
                | BuiltinFunction::DebugTraceback
                | BuiltinFunction::CoroutineResume
                | BuiltinFunction::CoroutineYield,
            ) => Self::Closure,
            _ => Self::Other,
        }
    }

    const fn trace_type(self) -> Option<TraceType> {
        match self {
            Self::Nil => Some(TraceType::Nil),
            Self::Boolean => Some(TraceType::Boolean),
            Self::Integer => Some(TraceType::Integer),
            Self::Float => Some(TraceType::Float),
            Self::Int64 => Some(TraceType::Int64),
            Self::UInt64 => Some(TraceType::UInt64),
            Self::Table => Some(TraceType::Table),
            Self::Closure => Some(TraceType::Closure),
            Self::Thread => Some(TraceType::Thread),
            Self::String => Some(TraceType::String),
            Self::Borrowed => Some(TraceType::StringRef),
            Self::Other => None,
        }
    }

    const fn from_type(ty: TraceType) -> Self {
        match ty {
            TraceType::Nil => Self::Nil,
            TraceType::Boolean => Self::Boolean,
            TraceType::Integer => Self::Integer,
            TraceType::Float => Self::Float,
            TraceType::Int64 => Self::Int64,
            TraceType::UInt64 => Self::UInt64,
            TraceType::Table => Self::Table,
            TraceType::Closure => Self::Closure,
            TraceType::Thread => Self::Thread,
            // A userdata is held only by operations (the interpreter's
            // registers keep it as the kind no entry accepts).
            TraceType::Userdata | TraceType::Any => Self::Other,
            // A borrow is a string to the interpreter.
            TraceType::String | TraceType::StringRef => Self::String,
        }
    }
}

/// The live value of a number the interpreter holds.
#[derive(Clone, Copy, Debug)]
enum Live {
    Integer(i64),
    Float(f64),
}

/// A number operand and the value the interpreter holds for it.
#[derive(Clone, Copy, Debug)]
struct Number {
    operand: Operand,
    live: Live,
}

impl Number {
    const fn ty(self) -> TraceType {
        match self.live {
            Live::Integer(_) => TraceType::Integer,
            Live::Float(_) => TraceType::Float,
        }
    }
}

/// What a table access indexes with.
enum TableKey {
    /// A string constant the interpreter loaded into the key register.
    Field(crate::string::LuaString),
    /// A number, which names an array slot if it is integral and in range.
    Index(Number),
    /// A string the trace computes or reads at run time (a lane string or a
    /// borrow out of an array slot), with the bytes the interpreter holds.
    Hash(Operand, crate::string::LuaString),
    /// A table or function as the key (`ordinal_sturdy[shape]` in
    /// Mineclonia's `get_sturdy_faces_1`, nodeprops.lua:705), with the value
    /// the interpreter holds. Only a lookup call reads it.
    Object(Operand, Value),
}

const fn is_number(ty: TraceType) -> bool {
    matches!(ty, TraceType::Integer | TraceType::Float)
}

/// What a register holds at some point of the iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sym {
    /// The iteration has not written it yet.
    Carried,
    /// The result of recorded operation `index`.
    Op(u32),
    /// A string constant of the chunk, held by root slot `slot`. Nothing can
    /// read it, so a loop only passes it through to a snapshot.
    Root(u32),
    /// The table constructor `virtuals[index]` made. No table exists: its
    /// array and fields are SSA values, and an exit builds it from a recipe.
    /// Only the root frame holds one, and no operation reads the register but
    /// a move, a field or index access with a fixed key, and `#`.
    Virtual(u32),
}

/// A value a virtual table holds.
#[derive(Clone, Copy, Debug)]
enum VirtualItem {
    /// A number, boolean or 64-bit integer of a recorded value.
    Value(Operand, TraceType),
    /// A string constant, by root slot.
    Root(u32),
    /// Another virtual table of the recorder, by id: a table built inside
    /// the table, which stays virtual as long as neither escapes.
    Virtual(u32),
}

/// The table a virtual table became when it escaped.
#[derive(Clone, Copy, Debug)]
struct Escaped {
    /// The operation holding the table's lane ([`HelperId::TABLE_NEW`]).
    table: u32,
    /// How many operations the recording held once the table was made and
    /// filled. The state a virtual table recorded still describes the real
    /// one until an operation after this may have written it (see
    /// [`Recorder::shadow_is_current`]).
    after: usize,
}

/// The contents of a table constructor the trace runs without allocating, as
/// the interpreter's table would hold them: `array` is keys 1 to `len`, all
/// present, and `fields` the other present keys in insertion order. A nil
/// value stores nothing, so there are no holes.
#[derive(Clone, Debug)]
struct VirtualState {
    alloc: VirtualAlloc,
    array: Vec<VirtualItem>,
    fields: Vec<(VirtualKey, VirtualItem)>,
    /// The most entries the table held, which its phantom charge covers.
    peak: usize,
    /// The metatable `setmetatable` gave it.
    metatable: Option<VirtualMeta>,
    /// The real table the trace made when the virtual one escaped. The
    /// recorded contents stay as a model of it that reads can use.
    real: Option<Escaped>,
}

/// The metatable of a virtual table, and which of its metamethods the
/// recording found absent. The trace calls a helper that checks the same
/// absences each time it runs the call, so what the recording did without the
/// metamethod stays right.
#[derive(Clone, Copy, Debug)]
struct VirtualMeta {
    /// The metatable, a table operand.
    table: Operand,
    /// The metatable has no `__index`: a key the table lacks reads as nil.
    no_index: bool,
    /// The metatable has no `__newindex`: a key the table lacks is stored raw.
    no_newindex: bool,
    /// The metatable has no `__len`: `#` is the border.
    no_len: bool,
    /// The absences a later access relied on (the bits of the helper's
    /// mask: 1 `__index`, 2 `__newindex`, 4 `__len`). The helper checks
    /// only those, so a metamethod the trace never leans on may appear
    /// without sending the call to the interpreter.
    relied: u8,
}

impl VirtualMeta {
    /// The mask of the metamethods the helper must find absent.
    fn mask(self) -> i64 {
        let absent =
            |flag: bool, bit: u8| i64::from(flag && self.relied & bit != 0) * i64::from(bit);
        absent(self.no_index, 1) | absent(self.no_newindex, 2) | absent(self.no_len, 4)
    }
}

impl VirtualState {
    fn entries(&self) -> usize {
        self.array.len() + self.fields.len()
    }

    fn items(&self) -> impl Iterator<Item = &VirtualItem> {
        self.array
            .iter()
            .chain(self.fields.iter().map(|(_, item)| item))
    }
}

/// An operand of a recorded operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Operand {
    /// Register `r` as the iteration found it.
    Carried(usize),
    Op(u32),
}

/// How the recorder makes a library call: through the VM, or by calling a
/// helper.
#[derive(Clone, Copy)]
enum Plan {
    Host(HostFunction),
    Helper(HelperId),
}

#[derive(Clone, Debug)]
enum RecKind {
    Constant(TraceScalar),
    /// A register of an inlined frame a side recording starts inside (see
    /// [`SeedFrame`]): the value the parent's state at the guard holds for
    /// `register` of the side's frame `frame`, a side input
    /// ([`oximond::jit_ir::trace::FrameInput`]). It becomes no instruction.
    FrameInput {
        frame: u32,
        register: u16,
    },
    Move(Operand),
    Negate(Operand),
    IntToFloat(Operand),
    Numeric(NumericOp, Operand),
    ToWide(Operand, TraceType, WideConversion),
    NativeUnary(NativeUnary, Operand),
    NativeBinary(NativeBinary, Operand, Operand),
    Arithmetic(ArithmeticOp, Operand, Operand),
    Compare(Comparison, Operand, Operand),
    Not(Operand),
    /// The join of a pure two-arm branch: the condition, then the value the
    /// register holds after the arm the condition true takes, then the
    /// other's ([`TraceOp::Select`]).
    Select(Operand, Operand, Operand),
    GuardBranch {
        condition: Operand,
        expected: bool,
        snapshot: u32,
    },
    /// `table[key]` on the array part. The snapshot is a `Before` one.
    /// `past_end` marks the nil read of a key outside the array part
    /// ([`TraceOp::ArrayLoad`] with `past_end`).
    ArrayLoad {
        binding: u32,
        table: Operand,
        key: Operand,
        expected: TraceType,
        past_end: bool,
        snapshot: u32,
    },
    /// `table[key]` on the array part of a table met by lane
    /// ([`TraceOp::LaneArrayLoad`]), whose words come through the address
    /// word `field` of the lane's row. The snapshot is a `Before` one.
    LaneArrayLoad {
        table: Operand,
        field: u32,
        key: Operand,
        expected: TraceType,
        snapshot: u32,
    },
    /// `table[key]` on the array part for a slot that holds an integer or a
    /// float ([`TraceOp::NumberLoad`]): a float with the number's value. The
    /// snapshot is a `Before` one.
    NumberLoad {
        binding: u32,
        table: Operand,
        key: Operand,
        snapshot: u32,
    },
    /// Whether the number loaded by the [`Self::NumberLoad`] operation is an
    /// integer ([`TraceOp::NumberTag`]).
    NumberTag(Operand),
    /// The conjunction of two booleans ([`TraceOp::BoolAnd`]).
    BoolAnd(Operand, Operand),
    /// [`TraceOp::NumberModulo`] of two floats, with the float modulo to use
    /// and the integer tag.
    NumberModulo(ArithmeticOp, Operand, Operand, Operand),
    /// The step of `next` along the array part ([`TraceOp::ArrayNext`]):
    /// the index of the first slot with a value after `key`, or after the
    /// start of the array when `start`. The snapshot is a `Before` one.
    ArrayNext {
        binding: u32,
        table: Operand,
        key: Operand,
        start: bool,
        ty: TraceType,
        snapshot: u32,
    },
    /// `table[key]` for a string key known only at run time, on a table
    /// without a metatable. The snapshot is a `Before` one.
    HashLoad {
        binding: u32,
        table: Operand,
        key: Operand,
        expected: TraceType,
        snapshot: u32,
    },
    /// `table[key] = value`. With `set` a raw set that may add the key
    /// ([`TraceOp::ArraySet`]), otherwise an in-place overwrite.
    ArrayStore {
        binding: u32,
        table: Operand,
        key: Operand,
        value: Operand,
        snapshot: u32,
        set: bool,
    },
    /// An in-place `table[key] = value` that a slot fact proves cannot fail
    /// ([`TraceOp::ArrayOverwrite`]): no guard and no snapshot (see
    /// `slot_facts`).
    ArrayOverwrite {
        binding: u32,
        table: Operand,
        key: Operand,
        value: Operand,
    },
    FieldLoad {
        binding: u32,
        table: Operand,
    },
    FieldStore {
        binding: u32,
        table: Operand,
        value: Operand,
    },
    TableLength {
        binding: u32,
        table: Operand,
    },
    /// `#string` of a string whose bytes the entry bound.
    StringLength {
        binding: u32,
        string: Operand,
    },
    /// A byte of a bound string. The snapshot is a `Before` one.
    StringByte {
        binding: u32,
        string: Operand,
        key: Operand,
        snapshot: u32,
    },
    /// `table.name` on a table met at run time. The snapshot is a `Before`
    /// one.
    TableFieldLoad {
        field: u32,
        table: Operand,
        snapshot: u32,
    },
    TableFieldStore {
        field: u32,
        table: Operand,
        value: Operand,
        snapshot: u32,
    },
    /// `table[key] = value` overwriting a number or boolean in the array
    /// part of a table met by lane ([`TraceOp::LaneArrayStore`]), whose
    /// words come through the address word `field` of the lane's row. The
    /// snapshot is a `Before` one.
    LaneArrayStore {
        table: Operand,
        field: u32,
        key: Operand,
        value: Operand,
        snapshot: u32,
    },
    /// The metatable of `table` is the class binding `binding` pins
    /// ([`TraceOp::MetaGuard`]). The snapshot is a `Before` one.
    MetaGuard {
        binding: u32,
        table: Operand,
        snapshot: u32,
    },
    /// The constructor of virtual table `virt` ran: a phantom charge of its
    /// final peak size ([`TraceOp::PhantomTable`]).
    PhantomTable {
        virt: u32,
    },
    /// A call of a library function through the VM ([`TraceOp::HostCall`]).
    /// The snapshot is the `Before` one of the call.
    HostCall {
        function: HostFunction,
        args: Vec<Operand>,
        result: TraceType,
        snapshot: u32,
    },
    /// A call of a helper ([`TraceOp::CallHelper`]) that gives one result at
    /// most. The snapshot is the `Before` one of the call.
    CallHelper {
        helper: HelperId,
        args: Vec<Operand>,
        snapshot: u32,
    },
    /// Result `index` (after the first) of the helper call `call`
    /// ([`TraceOp::HelperResult`]).
    HelperResult {
        call: u32,
        index: u8,
    },
    /// A self call of a function trace ([`TraceOp::SelfCall`]). The snapshot
    /// is the `Before` one of the call.
    SelfCall {
        args: Vec<Operand>,
        snapshot: u32,
    },
}

impl RecKind {
    /// The snapshot the operation's guard leaves through, if it has one.
    const fn guard_snapshot(&self) -> Option<u32> {
        match self {
            Self::GuardBranch { snapshot, .. }
            | Self::ArrayLoad { snapshot, .. }
            | Self::LaneArrayLoad { snapshot, .. }
            | Self::NumberLoad { snapshot, .. }
            | Self::ArrayNext { snapshot, .. }
            | Self::HashLoad { snapshot, .. }
            | Self::ArrayStore { snapshot, .. }
            | Self::StringByte { snapshot, .. }
            | Self::TableFieldLoad { snapshot, .. }
            | Self::TableFieldStore { snapshot, .. }
            | Self::LaneArrayStore { snapshot, .. }
            | Self::MetaGuard { snapshot, .. }
            | Self::HostCall { snapshot, .. }
            | Self::CallHelper { snapshot, .. }
            | Self::SelfCall { snapshot, .. } => Some(*snapshot),
            _ => None,
        }
    }

    /// Whether recording this ends the lifetime of every borrowed string,
    /// like [`TraceOp::ends_borrows`] of the operation it becomes. No wildcard:
    /// a new kind must say.
    fn ends_borrows(&self) -> bool {
        match self {
            Self::ArrayStore { set, .. } => *set,
            Self::SelfCall { .. } => true,
            Self::CallHelper { helper, .. } => helper.effects().ends_borrows(),
            Self::Constant(_)
            | Self::FrameInput { .. }
            | Self::Move(_)
            | Self::Negate(_)
            | Self::IntToFloat(_)
            | Self::Numeric(..)
            | Self::ToWide(..)
            | Self::NativeUnary(..)
            | Self::NativeBinary(..)
            | Self::Arithmetic(..)
            | Self::Compare(..)
            | Self::Not(_)
            | Self::NumberTag(_)
            | Self::BoolAnd(..)
            | Self::Select(..)
            | Self::NumberModulo(..)
            | Self::GuardBranch { .. }
            | Self::ArrayLoad { .. }
            | Self::LaneArrayLoad { .. }
            | Self::NumberLoad { .. }
            | Self::ArrayNext { .. }
            | Self::HashLoad { .. }
            | Self::FieldLoad { .. }
            | Self::FieldStore { .. }
            | Self::ArrayOverwrite { .. }
            | Self::TableLength { .. }
            | Self::StringLength { .. }
            | Self::StringByte { .. }
            | Self::TableFieldLoad { .. }
            | Self::TableFieldStore { .. }
            | Self::LaneArrayStore { .. }
            | Self::MetaGuard { .. }
            | Self::HostCall { .. }
            | Self::HelperResult { .. }
            | Self::PhantomTable { .. } => false,
        }
    }

    /// Whether the operation commits an effect: a store, or a helper that
    /// writes or advances private state. A snapshot from before it would
    /// replay the effect.
    fn commits_effect(&self) -> bool {
        match self {
            Self::ArrayStore { .. }
            | Self::ArrayOverwrite { .. }
            | Self::FieldStore { .. }
            | Self::TableFieldStore { .. }
            | Self::LaneArrayStore { .. }
            | Self::SelfCall { .. } => true,
            Self::CallHelper { helper, .. } => helper.effects().commits_effect(),
            Self::Constant(_)
            | Self::FrameInput { .. }
            | Self::Move(_)
            | Self::Negate(_)
            | Self::IntToFloat(_)
            | Self::Numeric(..)
            | Self::ToWide(..)
            | Self::NativeUnary(..)
            | Self::NativeBinary(..)
            | Self::Arithmetic(..)
            | Self::Compare(..)
            | Self::Not(_)
            | Self::NumberTag(_)
            | Self::BoolAnd(..)
            | Self::Select(..)
            | Self::NumberModulo(..)
            | Self::GuardBranch { .. }
            | Self::ArrayLoad { .. }
            | Self::LaneArrayLoad { .. }
            | Self::NumberLoad { .. }
            | Self::ArrayNext { .. }
            | Self::HashLoad { .. }
            | Self::FieldLoad { .. }
            | Self::TableLength { .. }
            | Self::StringLength { .. }
            | Self::StringByte { .. }
            | Self::TableFieldLoad { .. }
            | Self::MetaGuard { .. }
            | Self::HostCall { .. }
            | Self::HelperResult { .. }
            | Self::PhantomTable { .. } => false,
        }
    }

    fn operands(&self, mut visit: impl FnMut(Operand)) {
        match self {
            Self::Constant(_) | Self::FrameInput { .. } | Self::PhantomTable { .. } => {}
            Self::HelperResult { call, .. } => visit(Operand::Op(*call)),
            Self::Move(source)
            | Self::Negate(source)
            | Self::IntToFloat(source)
            | Self::Numeric(_, source)
            | Self::ToWide(source, ..)
            | Self::NativeUnary(_, source)
            | Self::NumberTag(source)
            | Self::Not(source) => visit(*source),
            Self::Arithmetic(_, left, right)
            | Self::Compare(_, left, right)
            | Self::BoolAnd(left, right)
            | Self::NativeBinary(_, left, right) => {
                visit(*left);
                visit(*right);
            }
            Self::NumberModulo(_, left, right, integer) => {
                visit(*left);
                visit(*right);
                visit(*integer);
            }
            Self::Select(condition, if_true, if_false) => {
                visit(*condition);
                visit(*if_true);
                visit(*if_false);
            }
            Self::GuardBranch { condition, .. } => visit(*condition),
            Self::ArrayLoad { table, key, .. }
            | Self::LaneArrayLoad { table, key, .. }
            | Self::NumberLoad { table, key, .. }
            | Self::ArrayNext { table, key, .. }
            | Self::HashLoad { table, key, .. } => {
                visit(*table);
                visit(*key);
            }
            Self::ArrayStore {
                table, key, value, ..
            }
            | Self::ArrayOverwrite {
                table, key, value, ..
            }
            | Self::LaneArrayStore {
                table, key, value, ..
            } => {
                visit(*table);
                visit(*key);
                visit(*value);
            }
            Self::FieldLoad { table, .. }
            | Self::TableLength { table, .. }
            | Self::MetaGuard { table, .. }
            | Self::TableFieldLoad { table, .. } => visit(*table),
            Self::StringLength { string, .. } => visit(*string),
            Self::StringByte { string, key, .. } => {
                visit(*string);
                visit(*key);
            }
            Self::FieldStore { table, value, .. } | Self::TableFieldStore { table, value, .. } => {
                visit(*table);
                visit(*value);
            }
            Self::HostCall { args, .. }
            | Self::CallHelper { args, .. }
            | Self::SelfCall { args, .. } => {
                args.iter().copied().for_each(visit);
            }
        }
    }
}

/// One value an open-result call left for the instruction after it.
#[derive(Clone, Copy, Debug)]
enum Open {
    /// An operation's value.
    Op(u32),
    /// The one-byte string `string.sub (s, i, i)` makes, as the byte the
    /// operation holds. Nothing makes the string: only `string.byte` takes
    /// it, which returns the byte.
    Char(u32),
    /// A string constant of the trace's roots, by root slot: what `type`
    /// returns. It becomes a constant operation where an operation must
    /// take it, and stays a root where a register can.
    Root(u32),
}

/// Where the table of a table operation comes from.
#[derive(Clone, Copy, Debug)]
enum Receiver {
    /// An entry input no iteration writes, in this window register.
    Bound(usize),
    /// An entry input like [`Self::Bound`] that the code of the loops around
    /// this one writes: another table at another outer iteration.
    Around(usize),
    /// A table read out of an array by this iteration.
    Loaded(Operand),
}

/// How a field operation reaches its table.
#[derive(Clone, Copy, Debug)]
enum FieldAccess {
    /// The entry resolves the field's slot once, in this window register's
    /// table.
    Bound(usize),
    /// Native code finds the field through the table's lane at run time.
    Lane(Operand),
}

#[derive(Clone, Debug)]
struct RecOp {
    /// The logical frame the instruction ran in: 0 for the root frame.
    frame: u32,
    pc: usize,
    kind: RecKind,
    /// The type of the value the operation defines, if it defines one.
    ty: Option<TraceType>,
}

/// The previously executed guest pc a `Before` snapshot records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrevPc {
    /// Before the first instruction: the closing branch of the previous
    /// iteration, known only when the loop closes.
    Latch,
    /// Before the first instruction of an inlined call: nothing ran yet.
    Unset,
    At(usize),
}

/// One inlined frame of a snapshot, innermost last.
#[derive(Clone, Debug, PartialEq)]
struct RecSnapFrame {
    frame: u32,
    prev: PrevPc,
    regs: Vec<Sym>,
}

#[derive(Clone, Debug)]
struct RecSnapshot {
    phase: SnapshotPhase,
    /// The root frame's previously executed pc. Inside an inlined call it is
    /// the call.
    prev: PrevPc,
    /// Where the root frame resumes: the phase's own pc in the root frame,
    /// the instruction after the outermost call inside an inlined one. Decides
    /// which root registers are dead.
    root_resume: usize,
    /// The IR frame of the root frame's activation: 0, or a later one a tail
    /// call made (see [`Activation`]).
    root_frame: u32,
    /// The root frame's registers and the window slots after them.
    regs: Vec<Sym>,
    /// Inside an inlined call: the call's pc in the root frame and the
    /// registers its return writes (the first and how many). Nothing reads
    /// the old value of one that was dead at the call, whatever the resume
    /// pc after it says.
    pending: Option<(usize, usize, usize)>,
    /// Inside an inlined `..` handler: the registers of the root frame, from
    /// the first up to but not including the second, that hold the operands
    /// the chain has not folded in. The rebuilt frame's return reads them,
    /// though the instruction after the chain does not, so they are live.
    held: Option<(usize, usize)>,
    /// The inlined frames, outermost first.
    inner: Vec<RecSnapFrame>,
    /// The virtual tables the root registers name, as they stood when the
    /// snapshot was taken, by index into the recorder's `virtuals`.
    virtuals: Vec<(u32, VirtualState)>,
    /// Counted guest instructions executed since the iteration began, when
    /// the exit restores this snapshot (zero unless the recorder counts).
    counted: u32,
}

impl RecSnapshot {
    /// Whether restoring `other` gives the interpreter the same state as
    /// restoring this one.
    fn same_as(&self, other: &Self) -> bool {
        self.phase == other.phase
            && self.prev == other.prev
            && self.root_resume == other.root_resume
            && self.root_frame == other.root_frame
            && self.counted == other.counted
            && self.pending == other.pending
            && self.held == other.held
            && self.regs == other.regs
            && self.inner == other.inner
            && (self.virtuals.is_empty() && other.virtuals.is_empty()
                || format!("{:?}", self.virtuals) == format!("{:?}", other.virtuals))
    }
}

/// Where a hook must land: the call depth, the chunk (by address) and the pc.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Place {
    depth: usize,
    chunk: usize,
    pc: usize,
}

impl Place {
    fn of(depth: usize, chunk: &Arc<Chunk>, pc: usize) -> Self {
        Self {
            depth,
            chunk: Arc::as_ptr(chunk) as usize,
            pc,
        }
    }
}

/// The instruction last recorded and where the recorder expects the next hook.
#[derive(Clone, Copy, Debug)]
struct Last {
    at: Place,
    next: Place,
}

/// A call the trace is inlining: the callee's symbolic registers while it
/// runs, and what the return needs.
#[derive(Debug)]
struct Inlined {
    chunk: Arc<Chunk>,
    /// Index into the recorder's pinned closures.
    closure: u32,
    /// Index into the recorder's inlined frame table, plus one: the IR's
    /// `FrameId`.
    frame: u32,
    /// The callee's registers. Nothing here is carried: a fresh activation
    /// holds its arguments and nil.
    regs: Vec<Sym>,
    prev: PrevPc,
    /// The call instruction in the caller.
    call_pc: usize,
    result_base: usize,
    results: usize,
    /// The values the call passed beyond the callee's parameters, which
    /// `Vararg` reads. Empty for a callee without varargs.
    varargs: Vec<Sym>,
    /// Whether this frame opened the re-execution region (see
    /// [`Recorder::reexec`]), which its return closes.
    owns_reexec: bool,
    /// Tail calls that replaced the frame the call made.
    tail_calls: u32,
    /// Whether `pcall` or `xpcall` made the call: the return delivers `true`
    /// before the callee's values.
    protected: Protected,
    /// Whether this is the message handler of an `xpcall` that caught an
    /// error the trace raised: the return delivers `false` and the handler's
    /// first value.
    handler: bool,
    /// What entered the frame: a metamethod instruction returns into the
    /// register it writes, or discards the values (`__newindex`).
    entry: Entry,
    /// What the return does beyond writing the callee's values.
    after: Pending,
    /// For a call that runs whichever closure of the pinned closure's
    /// prototype it was made with (see [`InlinedFrame::dynamic`]): the
    /// operation holding the callee, which the frame's snapshots carry as a
    /// register past the chunk's and its cell accesses go through.
    dynamic: Option<u32>,
}

/// What an inlined frame's return owes its instruction beyond the values
/// (see [`inline_operator`]).
#[derive(Debug, Default)]
enum Pending {
    /// Nothing: the values go where the call's results do.
    #[default]
    Nothing,
    /// A `..` that ran a handler for one pair of operands: the rest of the
    /// chain is still to be folded into the result.
    Concat(Box<inline_operator::ConcatChain>),
    /// A `tostring` that ran `__tostring`: the return must be the string the
    /// interpreter takes it for.
    ToString,
}

/// A later activation of the root frame: a tail call in the root frame
/// replaced the root closure with this one, as the interpreter's in-place tail
/// call does (see [`oximond::jit_ir::trace::RootActivation`]). The root's
/// window stays: its guest registers hold the activation's registers (a fresh
/// call's: the arguments, then nil), and the slots after them stay the
/// trace's cells and globals. Upvalues and globals read through the pinned
/// closure, as an inlined callee's do.
#[derive(Debug)]
struct Activation {
    chunk: Arc<Chunk>,
    /// Index into the recorder's pinned closures.
    closure: u32,
    /// The IR frame that stands for the activation: index into `inlined`,
    /// plus one.
    frame: u32,
}

/// What recording an instruction decided about where execution goes next.
#[derive(Clone, Copy, Debug)]
enum Flow {
    /// The next instruction of the same frame.
    Next(usize),
    /// A call the interpreter makes: the callee starts at its pc 0.
    Enter,
    /// A tail call the interpreter makes: the callee replaces the running
    /// frame and starts at its pc 0, one frame deep where that frame was.
    Replace,
    /// A tail call in the root frame: the callee replaces the root frame (a
    /// later activation of it, or the root closure again) and starts at its
    /// pc 0.
    ReplaceRoot,
    /// A return into the caller, which resumes at `resume`: the instruction
    /// after its call.
    Leave { resume: usize },
    /// The root frame's return into its caller, which the recording follows
    /// (see [`Recorder::return_through`]): the caller resumes at `resume`.
    LeaveRoot { resume: usize },
    /// The call in the root frame's caller that makes the root frame again
    /// (see [`Recorder::call_root_again`]): the root function starts at its
    /// pc 0.
    EnterRoot,
    /// An error a protected call of the trace catches: the interpreter drops
    /// `levels` frames, and the caller of the protected call resumes at
    /// `resume`.
    Caught { resume: usize, levels: usize },
    /// An error an `xpcall` of the trace catches: the interpreter drops
    /// `levels` frames and runs the message handler, the innermost frame
    /// now, from its first instruction.
    CaughtHandler { levels: usize },
    /// A self call of a function trace: the recorder sleeps through the
    /// callee and wakes at the instruction after the call.
    Sleep,
    /// The `Return` of a function trace's activation: the trace closes.
    Return,
    /// A `Return` of the root frame in an outer side trace: the side closes
    /// with an exit before it (see [`Recorder::exit_at`]).
    Exit,
}

/// What an open-result call noted for the call that takes its results (see
/// [`Recorder::hoist_callee`]).
#[derive(Debug, Clone, Copy)]
struct Hoisted {
    /// The consumer's callee, guarded before the producer ran.
    callee: Operand,
    /// The consumer: the depth of its frame and its pc.
    consumer: (usize, usize),
    /// For open results that a native will take: the snapshot before the
    /// call that makes them, how many operations the recording held then, and
    /// the call's pc. The consumer may leave through it (see
    /// [`Recorder::open_region`]).
    start: Option<(u32, usize, usize)>,
}

/// Records one iteration of one loop.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent facts about one recording: profile flags and start-up state"
)]
pub(crate) struct Recorder {
    chunk: Arc<Chunk>,
    /// The caps of this recording.
    limits: RecordLimits,
    depth: usize,
    header: usize,
    /// The backward branch that reached the header. The loop spans
    /// `header..=latch`, and an iteration that leaves it cannot close.
    latch: usize,
    coroutine: Option<ThreadHandle>,
    /// Whether the profile has an integer subtype.
    integers: bool,
    /// The profile's float `%` and `^`.
    float_arith: FloatArith,
    /// How the profile converts a string operand of arithmetic (see
    /// `string_number`).
    string_number_mode: i64,
    /// Whether `_ENV` is a name for the environment (Lua 5.2 and later).
    profile_has_env_name: bool,
    /// Whether `pairs` calls a `__pairs` handler (every profile but the Lua
    /// 5.1 family).
    honours_pairs: bool,
    /// Whether `#t` calls a `__len` handler (every profile but the Lua 5.1
    /// family).
    honours_len: bool,
    /// Whether the profile is of the Lua 5.1 family: an ordering metamethod
    /// needs operands of one type and one handler both share.
    lua51_family: bool,
    /// Whether `a <= b` falls back on `not (b < a)` when no `__le` answers
    /// (every profile but Lua 5.5).
    le_falls_back: bool,
    /// Whether `xpcall` calls its function with no arguments (Lua 5.1).
    xpcall_drops_arguments: bool,
    /// Whether a caught `error(nil)` gives a message (Lua 5.5).
    no_error_object: bool,
    /// What a failed `assert` raises for its message.
    assert_style: inline_caught::AssertStyle,
    /// The error the recording expects the interpreter to raise and catch
    /// (see `inline_caught`).
    caught: Option<CaughtCheck>,
    /// The result window (register and count) of an `xpcall` whose handler
    /// just returned: the interpreter must show `false` there.
    handled: Option<(usize, usize)>,
    /// Guest registers of the chunk. The window the trace describes goes
    /// beyond them: `externals` follow, one window register each.
    register_count: usize,
    /// The upvalue cells and globals the iteration touched, in order of first
    /// use. Slot `i` is window register `register_count + i`.
    externals: Vec<ExternalSlot>,
    /// The chunk constant behind each root slot.
    root_constants: Vec<Arc<[u8]>>,
    /// The table storage the trace's operations need the entry to resolve.
    bindings: Vec<TableBinding>,
    /// How many of the bindings the tree gave a side recording. The side may
    /// add bindings after them but may not change one.
    seeded_bindings: usize,
    /// The tables the root frame's registers held when the recording began,
    /// by register: the entry inputs a metatable class can bind through (see
    /// `Recorder::class_window`).
    start_tables: Vec<(usize, TableHandle)>,
    /// The fields of run-time tables the trace reads and writes.
    fields: Vec<TableField>,
    /// The guest registers the loops around this one write outside it,
    /// found on first use. A table in one of them is not the same at every
    /// entry or outer iteration.
    written_around: OnceLock<Vec<bool>>,
    /// The names of the globals the trace tree stores in the root chunk
    /// (`SetGlobal`; see `stores_global`).
    stored_globals: OnceLock<Vec<Box<[u8]>>>,
    /// Whether a tree rooted at this loop could run the loops around it.
    coverable: OnceLock<bool>,
    liveness: SharedLiveness,
    /// The tag operation of each number operation whose subtype is known only
    /// at run time, by operation index: the float operation holds the
    /// number's value and the tag says whether it is an integer. A register
    /// that holds one is read only by the instructions that take a number of
    /// either subtype (see [`Self::record_number_arithmetic`]); the others
    /// first narrow it to the subtype it has now, behind a guard.
    number_tags: std::collections::HashMap<u32, u32>,
    /// Set while the destination register of a table read is recorded: the
    /// read may give a number of either subtype.
    number_wanted: bool,
    started: bool,
    /// What each register held at the header.
    entry: Vec<Kind>,
    /// What each register holds now.
    regs: Vec<Sym>,
    /// Registers the iteration has written.
    written: Vec<bool>,
    /// The control registers of the `string.gmatch` loops the trace holds
    /// (see [`Self::ignores_loop_arguments`]), each with the operation of the
    /// step result the loop-back stored there, once the recording passed it.
    gmatch_controls: Vec<(usize, Option<u32>)>,
    /// The control register of the `next` loop this trace closes at, when the
    /// recording keeps its value in a lane of the run's bank (see
    /// [`TraceIr::boxed_control`]). The register is an entry input of type
    /// string, the steps take the lane, and any write the recording makes to
    /// the register stores the value in the lane instead.
    boxed: Option<usize>,
    /// Whether the call being recorded is the step of the boxed loop.
    boxed_call: bool,
    /// The steps native code takes along the table the boxed loop traverses
    /// (see [`TraceIr::boxed_steps`]).
    boxed_steps: Option<BoxedSteps>,
    /// How many array steps of `next` loops the recording made over tables
    /// with a hash part, and over tables without one (see
    /// [`TraceIr::array_steps_plain`]).
    array_steps: (u32, u32),
    /// What the step of the boxed loop gave: an entry (`true`) or the end,
    /// for the loop's `ForInLoop`.
    boxed_found: Option<bool>,
    /// The loop variables of the boxed loop the body neither reads nor
    /// writes. The step does not produce them: an exit restores the first from
    /// the control lane (`boxed_key`) and the second from a lane of its own
    /// (`boxed_value`), and a later one is nil, as the entry left it.
    unread: Vec<usize>,
    /// Whether the first variable of the boxed loop is one of `unread`: it
    /// holds the control value.
    boxed_key: bool,
    /// Whether the second variable of the boxed loop is one of `unread`: its
    /// register is a lane the step stores the value in.
    boxed_value: bool,
    /// The pc of the instruction being recorded.
    current_pc: usize,
    /// An error a write to the boxed register met, reported when the
    /// instruction is done.
    deferred: Option<TraceAbortReason>,
    ops: Vec<RecOp>,
    snapshots: Vec<RecSnapshot>,
    /// Which snapshots a guard leaves through, and how many do: what
    /// `maxsnap` counts (see "Guards and snapshots").
    guarded: Vec<bool>,
    guarded_snapshots: usize,
    /// What the recording knows about the array slots it read or stored
    /// (see `slot_facts`).
    slot_facts: Vec<slot_facts::SlotFact>,
    /// Array reads a slot fact answered, which took no guard or snapshot.
    slot_reads_forwarded: u32,
    /// In-place array stores a slot fact proved, which took no guard or
    /// snapshot.
    slot_overwrites: u32,
    /// The callee identity guards of calls in inlined frames: the value the
    /// callee came from (past its moves), the lane it was compared with, and
    /// the guard's operation. A later call of the same value needs no guard
    /// of its own (see [`Recorder::callee_guarded`]).
    callee_guards: Vec<(Operand, u32, u32)>,
    /// Registers a branch tested without a guard. Their type decides the
    /// direction, so they belong to the entry contract.
    condition_reads: Vec<usize>,
    /// Registers of the root frame that the loop's entries find nil on some
    /// entries and a number on others (see [`Self::with_lazy`]).
    lazy: Vec<u16>,
    /// The state the recording started in, and the registers of `lazy` that
    /// held nil or a number then: each becomes an entry input of type `Any`
    /// that a type guard at the start of the trace narrows.
    lazy_start: Option<(u32, Vec<usize>)>,
    guest: u32,
    /// Guest instructions' worth of work the calls through the VM add, which
    /// count for the policy but not against the trace's size.
    host_work: u32,
    /// The native or builtin function the recording could not take, for the
    /// statistics ([`super::NativeAborts`]).
    abort_callable: RefCell<Option<Value>>,
    /// The metamethod (`__index`, `__newindex`, `__len`) the instruction
    /// being recorded runs and the recorder cannot follow, for the same
    /// statistics. Cleared at every instruction, so a recording that aborts
    /// on it names it and no other abort does.
    abort_metamethod: Option<&'static str>,
    last: Option<Last>,
    /// The rest of the arm the interpreter runs after a value select, which
    /// the recorder follows without recording (see the `select` module).
    select_skip: Option<select::SelectSkip>,
    /// The root frame's previously executed pc.
    prev: PrevPc,
    /// The calls being inlined now, outermost first. Empty while the root
    /// frame runs.
    frames: Vec<Inlined>,
    /// Every call inlined so far, in order. Entry `i` is IR frame `i + 1`.
    inlined: Vec<TraceFrameInfo>,
    /// The closures the trace pins, by identity. A call-site guard compares
    /// the live callee with one of them.
    pinned: Vec<u64>,
    /// The cell behind each upvalue slot in `externals`, to catch two window
    /// registers for one variable.
    external_cells: Vec<Option<CellHandle>>,
    /// The cell each `UpvalueField` window slot reads through (`None` when
    /// unknown). A field slot is read once at entry, so a trace that writes
    /// the cell must not use one.
    field_cells: Vec<Option<CellHandle>>,
    /// The root frame's captured locals when the recording began: guest
    /// register and cell. An inlined callee often shares one of these.
    root_captured: Vec<(u8, CellHandle)>,
    /// Whether a table set the recording saw is on a table that is no dense
    /// list now but may become one (see [`Self::bind_header`]).
    set_waiting: bool,
    /// Whether fields are read and written through the VM's lookups, never
    /// through a binding or the row of a table met by lane (see
    /// [`Site::rowless`](super::policy::Site)).
    rowless: bool,
    /// Whether this recording was refused for a store that moves hash storage
    /// beside such an access.
    refused_rows: std::cell::Cell<bool>,
    /// Whether a call reads and writes its callee's cells through the live
    /// closure ([`InlinedFrame::dynamic`]). Such a recording may not read a
    /// global through a cell (an `UpvalueField` slot would go stale) or touch
    /// a captured local of the root frame, which a write through a closure
    /// would leave stale in its register.
    dynamic_cells: bool,
    /// The frames (by `Inlined::frame`) that run whichever closure of a
    /// prototype their call had and whose live closure the recording guarded
    /// to have the pinned closure's environment
    /// ([`HelperId::CLOSURE_SAME_ENV`]): their globals are the pinned
    /// closure's ([`ExternalSlot::ClosureGlobal`]).
    env_guarded: Vec<u32>,
    /// The [`HelperId::CELL_GET_CALLABLE`] reads, by operation, of a cell
    /// that each closure of the reading prototype gets anew: one captured
    /// from a local of the enclosing function, which a new closure of the
    /// enclosing function captures again (`fn` of each xoroshiro's
    /// `next_integer`). A call of such a function varies as the closure
    /// does, and is guarded on its prototype ([`Self::callee_varies`]). A
    /// cell captured from an upvalue of the enclosing function is the
    /// enclosing closure's own (the ULL helpers `getlong` calls) and keeps
    /// the identity guard.
    instance_callees: Vec<u32>,
    /// Whether the recording writes a callee's cell through the live
    /// closure ([`HelperId::CELL_SET`]): such a write may reach a cell the
    /// window holds as a variable, or a captured local of the root frame.
    dynamic_cell_writes: bool,
    /// The call sites at which a retired root of this loop kept leaving at
    /// the callee's identity guard ([`Site::poly_calls`](super::policy::Site)):
    /// a call there is guarded on the prototype ([`Self::poly_call`]).
    poly_calls: Vec<super::policy::PolyCallPath>,
    /// Whether the trace keeps its branch guards in the loop body
    /// ([`Site::keep_guards`](super::policy::Site)).
    keep_guards: bool,
    /// For a side whose guard the policy judged to be the test of the root
    /// loop's own latch: the guard's pc in the root frame (see
    /// [`Self::close_side_at_start`]).
    latch_guard: Option<usize>,
    /// Names that share the variable of a register or window slot, and the
    /// root window register that holds it.
    aliases: Vec<(CellAlias, usize)>,
    /// How many of `aliases` a side recording took from its root.
    seeded_aliases: usize,
    /// Set for a side recording: the state it continues from. It starts at a
    /// pc inside the loop body and closes where the loop header comes round
    /// again, as a trace that links back to the root.
    side: Option<SideSeed>,
    /// A side recording takes the coroutine it finds at its first hook: the
    /// exit that started it did not say which one it ran in.
    adopt_coroutine: bool,
    /// Set for a side recording of a loop that other loops enclose: the
    /// root's exit continues into the code of the loops around it. Such a
    /// side may leave the root's span and take the backward branch of an
    /// enclosing loop, and closes where the root's header comes round again.
    outer_side: bool,
    /// Where an outer side trace that reached a `Return` of the root frame
    /// ends: it closes with an exit that resumes the interpreter there
    /// instead of a link, so the code after the nest (a short loop of its
    /// own unrolled) runs natively.
    exit_at: Option<usize>,
    /// The recording met a numeric for loop with too many trips left to
    /// unroll (see [`Recorder::met_long_inner_loop`]).
    met_long_loop: bool,
    /// Whether this outer side may go beyond the nest ([`SideSeed::beyond`]).
    beyond: bool,
    /// A recording that has not begun and watches the interpreter for the
    /// place a side should start (see [`Self::scouting`]).
    scout: Option<Scout>,
    /// Whether this is a loop side ([`SideSeed::looping`]).
    looping: bool,
    /// Whether an outer side may end in front of a loop for a loop side
    /// ([`SideSeed::loop_sides`]).
    loop_sides: bool,
    /// The header and the latch of the loop the recording ends in front of:
    /// the next hook at the header closes it with an exit.
    enter_loop: Option<(usize, usize)>,
    /// How many inlined frames deep `enter_loop` is: the loop of a call (see
    /// "Loops in calls" in the module docs), or 0 for one of the root frame.
    enter_depth: usize,
    /// The recorded snapshot of the terminal exit in front of a loop in a
    /// call, which names the call's frames.
    terminal: Option<u32>,
    /// The latch (chunk address and pc) of the loop in a call that the
    /// recording ends in front of (see [`Self::with_callee_loop`]).
    callee_loop: Option<super::policy::CalleeLoop>,
    /// For a loop side of a loop in a call: how many inlined frames deep the
    /// loop runs (the seed's frames). The loop closes in that frame. 0 for a
    /// loop of the root frame.
    loop_depth: usize,
    /// The registers of a loop side's innermost seeded frame that it reads
    /// again at its header ([`SeedRegister::Reloaded`]), with the stand-in
    /// each held until then.
    reloaded: Vec<(usize, u32)>,
    /// For a loop side of a loop in a call: the operations each register of
    /// the loop's frame held at the header, which the loop's parameters
    /// replace.
    loop_frame_start: Vec<Sym>,
    /// Whether a side may end in front of a loop inside the root's span
    /// ([`SideSeed::inner_loops`]).
    inner_loop_sides: bool,
    /// For a root recording: the headers of the loops inside its span that
    /// no tree of their own carries (see [`Self::with_free_loops`]). The
    /// recording ends in front of the first one it enters.
    free_loops: Vec<usize>,
    /// The ancestor loop sides a side may rejoin ([`SideSeed::loop_headers`]).
    loop_headers: Vec<SeedLoop>,
    /// The position in `loop_headers` of the loop side whose closing branch
    /// the recording took: the next hook at its header closes the recording
    /// with a [`TraceTerminator::LinkLoop`], or in front of the loop when
    /// the link cannot bind.
    link_loop: Option<usize>,
    /// The headers of the enclosing loops the outer side has branched back
    /// to so far.
    crossed: Vec<usize>,
    /// How many backward branches of `string.gmatch` loops inside the
    /// recorded loop the recording has followed (see [`Self::check_branch`]).
    unrolled: usize,
    /// How many backward branches of short numeric for loops inside the
    /// recorded loop the recording has followed (see
    /// [`Self::may_unroll_loop`]).
    unrolled_loops: usize,
    /// Backward branches of short loops the recording followed: the jumps of
    /// `while` loops, and the latches of the root frame's numeric for loops
    /// that `short_loops` names.
    unrolled_whiles: usize,
    /// Headers of loops in the root frame whose root was retired as too short.
    short_loops: Vec<usize>,
    /// Whether the recording may unroll short numeric for loops.
    unroll_loops: bool,
    /// For a root trace that closes at a numeric for loop's latch: the
    /// iterations left in the loop's run after the recorded one.
    remaining_trips: Option<f64>,
    /// The iterations a whole run of that loop has, when the counters bound
    /// them (see `run_trips`).
    run_trips: Option<f64>,
    /// For such a trace in the bottom frame of the main thread, inside only
    /// numeric for loops: the iterations of this loop the program has left,
    /// counting the runs the loops around it will start.
    remaining_work: Option<f64>,
    /// For such a trace, the nest a recording of an outer loop would unroll.
    nest_plan: Option<NestPlan>,
    /// While a call of a vararg function is inlined, the snapshot of the
    /// state before that call. Every guard inside it leaves through this
    /// snapshot, so the interpreter runs the whole call itself and builds
    /// the vararg frame as it always does. That is exact only while nothing
    /// observable happened in the callee, so a table store inside aborts the
    /// recording.
    reexec: Option<u32>,
    /// While a call is inlined: the snapshot before the outermost call, and
    /// how many table stores the recording held when it began. A branch
    /// guard inside the call leaves through it as long as no store followed,
    /// because the interpreter can then make the call itself (and the other
    /// arm gets a side trace that starts at the call, in the root frame).
    call_site: Option<(u32, usize)>,
    /// How many operations the recording held when [`Self::call_site`] was
    /// set: the ones after it are the call's.
    call_ops: usize,
    /// The guest instructions the recording held when [`Self::call_site`]
    /// was set, and the writes of window slots (upvalue cells) it had made
    /// (see [`Self::starts_call_over`]).
    call_guest: u32,
    call_cell_writes: usize,
    /// Writes of window slots beyond the guest registers recorded so far.
    cell_writes: usize,
    /// Table stores recorded so far.
    stores: usize,
    /// The values an open-result call left for the instruction after it, as
    /// operations. The interpreter keeps them in the frame's open-results slot
    /// where no snapshot can describe them, so the call that makes them
    /// guards its consumer's callee first (see [`Recorder::hoist_callee`])
    /// and the next instruction takes them at once.
    open: Option<Vec<Open>>,
    /// The identity guards open-result calls already made for the calls that
    /// consume their results, innermost last. The producer may be a Lua call
    /// whose inlined body holds producer and consumer pairs of its own
    /// (`tonumber (band (x, y))`), which finish before the outer consumer
    /// runs, so the notes nest. A consumer finds its note by frame depth and
    /// pc, and drops the notes above it: a consumer that took none (a
    /// symbolic native) leaves its note behind.
    hoisted: Vec<Hoisted>,
    /// Whether the call being recorded takes open results and leaves through
    /// the `start` of its [`Hoisted`] note where it needs a guard.
    open_region: bool,
    /// While an open-results region is recorded: the pc of the call that made
    /// the results, which the region's operations are sited at (their guards
    /// leave from there).
    site_override: Option<usize>,
    /// Whether a branch guard inside a call leaves through a snapshot of the
    /// callee's frames even where starting the call over would do. Only
    /// tests turn it on, to reach the exits that rebuild frames.
    rebuild_callee_frames: bool,
    /// Whether a pure two-arm branch on a boolean records as value selects
    /// (see the `select` module).
    value_selects: bool,
    /// Set when an operation that ends borrows was recorded while a register
    /// still held a borrowed string. The step that recorded it aborts.
    borrow_ended: bool,
    /// The table constructors the trace keeps virtual, in recording order.
    virtuals: Vec<VirtualState>,
    /// The `string.sub` calls recorded as [`HelperId::STRING_SLICE`] that a
    /// `string.byte` of the result may read through.
    slices: Vec<inline_slice::SliceNote>,
    /// The mask constants of the `plain_metatable` calls, with the virtual
    /// table each one checks: the recording does not know the mask until
    /// the accesses are in, so [`Self::finalize`] writes it.
    mask_ops: Vec<(u32, u32)>,
    /// The closure the root frame ran when the recording began. A tail call
    /// back into it closes a loop through tail calls.
    home_closure: Option<ClosureHandle>,
    /// While a tail call in the root frame has replaced the root closure: what
    /// runs in the root frame now.
    activation: Option<Activation>,
    /// The tail call back into the home closure that closed the loop, once
    /// recorded.
    loop_tail: Option<TailSite>,
    /// The root frame's varargs as the trace reads them, once a use named
    /// them (see `current_varargs`).
    root_varargs: Option<Vec<Sym>>,
    /// The chunk of each later activation of the root frame, by IR frame.
    activation_chunks: Vec<(u32, Arc<Chunk>)>,
    /// Whether the last recorded step pushed onto `activation_chunks`: a
    /// tail call that entered a new root activation.
    entered_activation: bool,
    /// Set for a function-entry recording or a side of a function trace.
    function: Option<FunctionSeed>,
    /// The self call whose callee the interpreter is running now.
    sleeping: Option<SelfCallWait>,
    /// The resume whose thread the interpreter is running now.
    resume_wait: Option<inline_resume::ResumeWait>,
    /// The host call whose function the interpreter is running now (see
    /// `inline_vmcall`).
    vm_wait: Option<inline_vmcall::VmCallWait>,
    /// Whether the site's earlier recording could not hold a host call
    /// itself, so this one aborts at the call.
    no_vmcall: bool,
    /// Whether this recording gave up a host call it could not hold.
    refused_vmcall: std::cell::Cell<bool>,
    /// Whether the fields of the tables the entry binds are read and written
    /// through the tables' lanes, whose rows the run makes again after a host
    /// call, instead of through slot addresses native code keeps.
    lane_fields: bool,
    /// Whether this recording held a host call beside field bindings, which
    /// the next recording avoids with [`Self::lane_fields`].
    wants_lane_fields: std::cell::Cell<bool>,
    /// The window registers self calls read their callee from.
    callee_registers: Vec<u16>,
    /// The value the closing `Return` gives.
    returned: Option<Operand>,
    /// The arguments and pc of the tail call of the function's own closure
    /// that closed the trace instead of a `Return`.
    tail: Option<(Vec<Operand>, usize)>,
    /// Whether `open` holds the result of a self call.
    open_from_self: bool,
    /// The caller registers the region in the root frame's caller has loaded
    /// from their window slots, with the operation that loaded each.
    caller_loads: Vec<(usize, u32)>,
    /// The chunk of the caller the recording returned into.
    returned_into: Option<Arc<Chunk>>,
    /// Whether the trace counts guest instructions for a count hook: every
    /// snapshot then carries the counted instructions that ran before it
    /// (see [`Recorder::may_run_lua`] for the rule operations follow).
    counting: bool,
    /// Counted instructions recorded so far in this iteration, a side's
    /// parent path included.
    counted: u32,
    /// What the instruction being recorded adds to `counted` once it has run:
    /// 1, or 0 for the register maintenance a count hook skips.
    counted_now: u32,
}

/// A variable's window register, and the name its live value is read by.
#[derive(Clone, Copy, Debug)]
struct ExternalRef {
    register: usize,
    slot: ExternalSlot,
}

impl Recorder {
    /// A recorder for the loop whose header is `header` and whose closing
    /// branch is at `latch`, started from the frame at call depth `depth`
    /// under `profile`.
    pub(crate) fn new(
        chunk: Arc<Chunk>,
        depth: usize,
        header: usize,
        latch: usize,
        coroutine: Option<ThreadHandle>,
        profile: CompatibilityProfile,
    ) -> Self {
        let register_count = usize::from(chunk.register_count);
        Self {
            chunk,
            limits: RecordLimits::default(),
            depth,
            header,
            latch,
            coroutine,
            integers: profile.is_lua53_or_later(),
            float_arith: profile.float_arith(),
            string_number_mode: match profile {
                CompatibilityProfile::Lua53 => STRING_NUMBER_LUA53_MODE,
                CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55 => {
                    STRING_NUMBER_METAMETHOD_MODE
                }
                _ => STRING_NUMBER_FLOAT_MODE,
            },
            profile_has_env_name: !matches!(
                profile,
                CompatibilityProfile::Lua51 | CompatibilityProfile::LuaJit21
            ),
            honours_pairs: !profile.is_lua51_family(),
            honours_len: !profile.is_lua51_family(),
            lua51_family: profile.is_lua51_family(),
            le_falls_back: profile != CompatibilityProfile::Lua55,
            xpcall_drops_arguments: inline_protected::xpcall_drops_arguments(profile),
            no_error_object: profile == CompatibilityProfile::Lua55,
            assert_style: inline_caught::AssertStyle::of(profile),
            caught: None,
            handled: None,
            register_count,
            externals: Vec::new(),
            root_constants: Vec::new(),
            bindings: Vec::new(),
            seeded_bindings: 0,
            start_tables: Vec::new(),
            fields: Vec::new(),
            written_around: OnceLock::new(),
            stored_globals: OnceLock::new(),
            coverable: OnceLock::new(),
            liveness: SharedLiveness::default(),
            number_tags: std::collections::HashMap::new(),
            number_wanted: false,
            started: false,
            entry: Vec::new(),
            regs: Vec::new(),
            written: Vec::new(),
            gmatch_controls: Vec::new(),
            boxed: None,
            boxed_call: false,
            boxed_steps: None,
            array_steps: (0, 0),
            boxed_found: None,
            unread: Vec::new(),
            boxed_key: false,
            boxed_value: false,
            current_pc: 0,
            deferred: None,
            ops: Vec::new(),
            snapshots: Vec::new(),
            guarded: Vec::new(),
            guarded_snapshots: 0,
            slot_facts: Vec::new(),
            slot_reads_forwarded: 0,
            slot_overwrites: 0,
            callee_guards: Vec::new(),
            condition_reads: Vec::new(),
            lazy: Vec::new(),
            lazy_start: None,
            borrow_ended: false,
            guest: 0,
            host_work: 0,
            abort_callable: RefCell::new(None),
            abort_metamethod: None,
            last: None,
            select_skip: None,
            prev: PrevPc::Latch,
            frames: Vec::new(),
            inlined: Vec::new(),
            pinned: Vec::new(),
            external_cells: Vec::new(),
            field_cells: Vec::new(),
            root_captured: Vec::new(),
            dynamic_cells: false,
            env_guarded: Vec::new(),
            instance_callees: Vec::new(),
            dynamic_cell_writes: false,
            poly_calls: Vec::new(),
            keep_guards: false,
            latch_guard: None,
            set_waiting: false,
            rowless: false,
            refused_rows: std::cell::Cell::new(false),
            aliases: Vec::new(),
            seeded_aliases: 0,
            side: None,
            adopt_coroutine: false,
            outer_side: false,
            scout: None,
            looping: false,
            loop_sides: false,
            enter_loop: None,
            enter_depth: 0,
            terminal: None,
            callee_loop: None,
            loop_depth: 0,
            reloaded: Vec::new(),
            loop_frame_start: Vec::new(),
            inner_loop_sides: false,
            free_loops: Vec::new(),
            loop_headers: Vec::new(),
            link_loop: None,
            exit_at: None,
            met_long_loop: false,
            beyond: false,
            crossed: Vec::new(),
            unrolled: 0,
            unrolled_loops: 0,
            unrolled_whiles: 0,
            short_loops: Vec::new(),
            unroll_loops: false,
            remaining_trips: None,
            run_trips: None,
            remaining_work: None,
            nest_plan: None,
            reexec: None,
            call_site: None,
            call_ops: 0,
            call_guest: 0,
            call_cell_writes: 0,
            cell_writes: 0,
            stores: 0,
            open: None,
            hoisted: Vec::new(),
            open_region: false,
            site_override: None,
            rebuild_callee_frames: false,
            value_selects: true,
            virtuals: Vec::new(),
            slices: Vec::new(),
            mask_ops: Vec::new(),
            home_closure: None,
            activation: None,
            loop_tail: None,
            root_varargs: None,
            activation_chunks: Vec::new(),
            entered_activation: false,
            function: None,
            sleeping: None,
            resume_wait: None,
            vm_wait: None,
            no_vmcall: false,
            refused_vmcall: std::cell::Cell::new(false),
            lane_fields: false,
            wants_lane_fields: std::cell::Cell::new(false),
            callee_registers: Vec::new(),
            returned: None,
            tail: None,
            open_from_self: false,
            caller_loads: Vec::new(),
            returned_into: None,
            counting: false,
            counted: 0,
            counted_now: 0,
        }
    }

    /// A recorder for a side trace of the loop at `header` (closing branch
    /// `latch`), continuing from the state a failed guard restored in the
    /// frame at call depth `depth`.
    pub(crate) fn new_side(
        chunk: Arc<Chunk>,
        depth: usize,
        header: usize,
        latch: usize,
        profile: CompatibilityProfile,
        seed: SideSeed,
    ) -> Self {
        let mut recorder = Self::new(chunk, depth, header, latch, None, profile);
        recorder.externals.clone_from(&seed.externals);
        recorder.root_constants.clone_from(&seed.root_constants);
        recorder.bindings.clone_from(&seed.bindings);
        recorder.seeded_bindings = seed.bindings.len();
        recorder.pinned.clone_from(&seed.pinned);
        recorder.fields.clone_from(&seed.fields);
        recorder.external_cells = vec![None; seed.externals.len()];
        // The root's promises, under the registers that stand for them.
        for alias in &seed.aliases {
            let register = match alias.is {
                CellSource::Captured(register) => Some(usize::from(register)),
                is => seed
                    .externals
                    .iter()
                    .position(|slot| Self::cell_source(*slot) == Some(is))
                    .map(|position| recorder.register_count + position),
            };
            if let Some(register) = register {
                recorder.aliases.push((*alias, register));
            }
        }
        recorder.seeded_aliases = recorder.aliases.len();
        recorder.field_cells = seed
            .externals
            .iter()
            .filter(|slot| {
                matches!(
                    slot,
                    ExternalSlot::UpvalueField { .. } | ExternalSlot::ClosureUpvalueField { .. }
                )
            })
            .map(|_| None)
            .collect();
        recorder.prev = seed.last.map_or(PrevPc::Unset, PrevPc::At);
        // A side that starts inside inlined calls: the root frame stands at
        // the outermost call, and the first hook comes in the innermost
        // frame, that many calls deeper.
        if let (Some(outermost), Some(innermost)) = (seed.frames.first(), seed.frames.last()) {
            recorder.prev = PrevPc::At(outermost.info.call_pc);
            let first = Place::of(depth + seed.frames.len(), &innermost.chunk, seed.start_pc);
            recorder.last = Some(Last {
                at: first,
                next: first,
            });
        }
        recorder.home_closure = seed.caller.as_ref().map(|caller| caller.root);
        recorder.adopt_coroutine = true;
        recorder.outer_side = seed.outer;
        recorder.beyond = seed.outer && seed.beyond;
        recorder.loop_sides = recorder.beyond && seed.loop_sides;
        recorder.looping = seed.looping;
        if seed.looping {
            recorder.loop_depth = seed.frames.len();
        }
        recorder.inner_loop_sides = seed.inner_loops && !seed.looping;
        recorder.loop_headers.clone_from(&seed.loop_headers);
        recorder.crossed.extend(seed.entered.filter(|_| seed.outer));
        recorder.function.clone_from(&seed.function);
        let lanes = nest::passes_loaded_tables(&recorder.chunk);
        if let Some(function) = &mut recorder.function {
            function.lanes = lanes;
        }
        if let Some(counted) = seed.counted {
            recorder.counting = true;
            recorder.counted = counted;
        }
        recorder.side = Some(seed);
        recorder
    }

    /// Makes this a function-entry recording of the frame running `closure`
    /// (see "Function traces" in the module docs).
    pub(crate) fn for_function(mut self, closure: ClosureHandle) -> Self {
        let lanes = nest::passes_loaded_tables(&self.chunk);
        self.function = Some(FunctionSeed {
            closure,
            parameters: Vec::new(),
            result: None,
            lanes,
        });
        // A fresh frame has run nothing yet.
        self.prev = PrevPc::Unset;
        self
    }

    /// The pc of the site the recording belongs to: the loop header, or
    /// [`super::policy::FUNCTION_ENTRY_PC`] for a function-entry recording.
    /// Whether the recording met the backward branch of a numeric for loop
    /// with more trips ahead than unrolling takes ([`MAX_UNROLLED_LOOP_TRIPS`]):
    /// a recording that unrolls loops ends at it the same way.
    pub(crate) const fn met_long_inner_loop(&self) -> bool {
        self.met_long_loop
    }

    pub(crate) fn site_pc(&self) -> usize {
        if self.function.is_some() {
            super::policy::FUNCTION_ENTRY_PC
        } else {
            self.header
        }
    }

    /// Guest instructions recorded so far.
    pub(crate) const fn guest_instructions(&self) -> u32 {
        self.guest
    }

    /// Whether the recording followed the backward branch of a loop it
    /// unrolled.
    pub(crate) const fn has_unrolled_loops(&self) -> bool {
        self.unrolled_loops > 0
    }

    /// Lets the recording unroll short numeric for loops (see
    /// [`Self::may_unroll_loop`]) within [`MAX_UNROLLED_GUEST_INSTRUCTIONS`].
    /// The policy allows it once a recording of the same header has met an
    /// inner loop.
    pub(crate) const fn with_loop_unrolling(mut self, unroll: bool) -> Self {
        self.unroll_loops = unroll;
        self
    }

    /// Names the loops of the root frame whose own root was retired for
    /// running too few trips per entry. A recording that unrolls `while`
    /// loops follows the jump of those and ends at any other as an inner
    /// loop.
    pub(crate) fn with_short_loops(mut self, short: Vec<usize>) -> Self {
        self.short_loops = short;
        self
    }

    /// Names the registers of the root frame that hold nil on some entries
    /// of the loop and a number on others (`lx_dom` and `rx_dom` of
    /// Mineclonia's `region_evaluate`). The entry check of a trace that
    /// typed one would turn every entry of the other type away. A register
    /// of these that the trace reads and never writes, and that holds nil or
    /// a number when the recording starts, becomes an entry input of type
    /// `Any` instead: a type guard at the start of the trace checks the type
    /// this recording saw, and a side of that guard takes the other.
    pub(crate) fn with_lazy(mut self, lazy: Vec<u16>) -> Self {
        self.lazy = lazy;
        self
    }

    /// Names the loops inside the root's span that no tree of their own
    /// carries (`TracePolicy::free_loops_of`): their root was rejected, or
    /// it is installed but no outer side of it can run the loops around it.
    /// A root recording that takes the backward branch of one of them ends
    /// in front of the loop's next trip with a terminal exit, and the loop
    /// becomes a loop side of this root's tree (see "Outer roots" in the
    /// module docs of `trace`).
    pub(crate) fn with_free_loops(mut self, free: Vec<usize>) -> Self {
        self.free_loops = free;
        self
    }

    /// Reads and writes fields through the VM's lookups, never through a
    /// binding or the row of a table met by lane: the recording a store that
    /// moves hash storage allows.
    pub(crate) const fn with_rowless(mut self, rowless: bool) -> Self {
        self.rowless = rowless;
        self
    }

    /// Makes the recording read and write the fields of bound tables through
    /// their lanes (see `inline_vmcall`).
    pub(crate) const fn with_lane_fields(mut self, lane_fields: bool) -> Self {
        self.lane_fields = lane_fields;
        self
    }

    /// Whether the recording held a host call beside field bindings, which a
    /// recording with lane fields avoids.
    pub(crate) const fn wants_lane_fields(&self) -> bool {
        self.wants_lane_fields.get()
    }

    /// Whether this recording reads fields through lanes.
    pub(crate) const fn lane_fields(&self) -> bool {
        self.lane_fields
    }

    /// Makes the recording abort at host calls instead of making them from
    /// the trace (see `inline_vmcall`).
    pub(crate) const fn with_no_vmcall(mut self, no_vmcall: bool) -> Self {
        self.no_vmcall = no_vmcall;
        self
    }

    /// The call sites of the loop that take turns between closures of one
    /// prototype (see [`Self::poly_call`]).
    pub(crate) fn with_poly_calls(mut self, calls: Vec<super::policy::PolyCallPath>) -> Self {
        self.poly_calls = calls;
        self
    }

    /// Keeps the branch guards of the loop body in it (see
    /// [`Site::keep_guards`](super::policy::Site)).
    pub(crate) const fn with_kept_guards(mut self, keep: bool) -> Self {
        self.keep_guards = keep;
        self
    }

    /// Says that the side continues the guard of the root loop's latch, at
    /// `site` in the root frame (see [`Self::close_side_at_start`]).
    pub(crate) const fn with_latch_guard(mut self, latch_guard: Option<usize>) -> Self {
        self.latch_guard = latch_guard;
        self
    }

    /// Whether the call at `pc` of the running frame is a site at which a
    /// retired root of this loop kept leaving at the callee's identity guard:
    /// the callee took turns between closures, one per structure set in
    /// Mineclonia's `get_structure_starts` (`placement.chunk_test`,
    /// structures.lua:299) or one per fork in its xoroshiro self-test
    /// (`fork:next_integer`, random.lua:1159). The path is the call pcs of
    /// the inlined frames, outermost first, then `pc`.
    fn poly_call(&self, pc: usize) -> bool {
        if self.poly_calls.is_empty() || self.function.is_some() || self.activation.is_some() {
            return false;
        }
        self.poly_calls.iter().any(|path| {
            path.len() == self.frames.len() + 1
                && path.last() == Some(&pc)
                && path
                    .iter()
                    .zip(&self.frames)
                    .all(|(call_pc, frame)| *call_pc == frame.call_pc)
        })
    }

    /// Whether the function that the instruction at `pc` reads into register
    /// `a` is the callee of a call at a [`Self::poly_call`] site a few
    /// instructions on, with nothing between that writes `a` (`fork:next_integer
    /// (3000)`: the `SelfField`, a constant, the call).
    fn reads_poly_callee(&self, at: &HookPoint<'_>, a: usize) -> bool {
        if self.poly_calls.is_empty() {
            return false;
        }
        let code = &at.chunk.code;
        let end = code.len().min(at.pc + 1 + MAX_POLY_CALLEE_LEAD);
        for (call_pc, &instruction) in code.iter().enumerate().take(end).skip(at.pc + 1) {
            let Ok(opcode) = instruction.opcode() else {
                return false;
            };
            let callee = callee_register(instruction);
            let call = callee.is_some();
            if callee.is_some_and(|callee| usize::from(callee) == a) {
                return self.poly_call(call_pc);
            }
            if call
                || opcode.is_test()
                || matches!(opcode, OpCode::Jump | OpCode::JumpIfFalse)
                || u8::try_from(a).is_ok_and(|register| {
                    at.chunk
                        .writes_register_between(call_pc, call_pc + 1, register)
                })
            {
                return false;
            }
        }
        false
    }

    /// Whether the recording was refused for a store that moves hash storage
    /// beside an access that addresses it, which a rowless recording avoids.
    pub(crate) const fn refused_rows(&self) -> bool {
        self.refused_rows.get()
    }

    /// Whether the recording gave up a host call it could not hold in the
    /// trace, at which the next recording of the site aborts.
    pub(crate) const fn refused_vmcall(&self) -> bool {
        self.refused_vmcall.get()
    }

    /// Makes the trace count guest instructions for a count hook: it records
    /// the counted instructions each snapshot has run, and the backend
    /// carries the hook's budget through the loop. Only a loop trace can
    /// count, so a recording that turns out to be a function trace or a
    /// loop through tail calls aborts.
    pub(crate) const fn with_counting(mut self, counting: bool) -> Self {
        self.counting = counting;
        self
    }

    /// The rule for an operation that can run Lua code, a hook, a metamethod
    /// or any callout into the interpreter while the trace runs: under a
    /// count hook such an operation would count (or fire) instructions
    /// inside native code, where the trace has not yet paid for the ones it
    /// ran. The recorder calls this when it meets one, and a counting
    /// recording stops there with the error. An operation that cannot do
    /// that (pure arithmetic, a length, a byte read, a table store that
    /// runs no `__newindex`) never calls it.
    ///
    /// A future operation that must run inside counted traces commits the
    /// counted instructions up to itself in the control block before the
    /// callout and guards the budget again after it, instead of failing here.
    #[allow(
        dead_code,
        reason = "no recorded operation runs Lua yet; the rule is the contract for those that will"
    )]
    pub(crate) const fn may_run_lua(&self) -> Result<(), TraceAbortReason> {
        if self.counting {
            Err(TraceAbortReason::UnsupportedOpcode)
        } else {
            Ok(())
        }
    }

    /// Tells the recorder which of the root frame's registers are captured
    /// locals, with their cells.
    pub(crate) fn with_captured(mut self, captured: Vec<(u8, CellHandle)>) -> Self {
        self.root_captured = captured;
        self
    }

    /// Sets the caps of this recording (see [`RecordLimits`]).
    pub(crate) const fn with_limits(mut self, limits: RecordLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Makes branch guards inside calls leave through snapshots of the
    /// callee's frames (see [`Recorder::rebuild_callee_frames`]).
    pub(crate) const fn with_rebuilt_callee_frames(mut self, rebuild: bool) -> Self {
        self.rebuild_callee_frames = rebuild;
        self
    }

    /// Turns value selects on or off.
    pub(crate) const fn with_value_selects(mut self, select: bool) -> Self {
        self.value_selects = select;
        self
    }

    /// Tells the recorder which closure the root frame runs.
    pub(crate) const fn with_home_closure(mut self, closure: Option<ClosureHandle>) -> Self {
        self.home_closure = closure;
        self
    }

    pub(crate) const fn home_closure(&self) -> Option<ClosureHandle> {
        self.home_closure
    }

    /// Shares the chunk's liveness cache with this recorder.
    pub(crate) fn with_liveness(mut self, liveness: SharedLiveness) -> Self {
        self.liveness = liveness;
        self
    }

    pub(crate) fn chunk(&self) -> &Arc<Chunk> {
        &self.chunk
    }

    /// Whether the recording is inside a frame that runs whichever closure
    /// of a prototype its call had (a call guarded on the prototype, see
    /// [`Self::poly_call`]), or a call such a frame makes.
    pub(crate) fn in_dynamic_frame(&self) -> bool {
        self.frames.iter().any(|frame| frame.dynamic.is_some())
    }

    /// Whether the recording is inside an inlined call.
    pub(crate) const fn in_inlined_call(&self) -> bool {
        !self.frames.is_empty()
    }

    /// Whether this is a function-entry recording.
    pub(crate) const fn is_function(&self) -> bool {
        self.function.is_some()
    }

    /// Ends the recording in front of the loop in a call whose latch is
    /// `latch` (chunk address and pc), when it meets that loop's backward
    /// branch: the loop runs as a loop side of the tree, in the call's
    /// frames (see "Loops in calls" in the module docs).
    pub(crate) fn with_callee_loop(mut self, latch: Option<super::policy::CalleeLoop>) -> Self {
        self.callee_loop = latch;
        self
    }

    /// Whether the loop closed by the backward branch at `latch` to `header`
    /// contains another loop's backward branch. The recorder aborts on those
    /// with `InnerLoop`, except for short numeric for loops it unrolls (see
    /// [`Self::may_unroll_loop`]), which only the run time tells apart.
    /// Mirrors the recorder's own branch check.
    pub(crate) fn has_inner_loop(chunk: &Chunk, header: usize, latch: usize) -> bool {
        Self::inner_loop_headers(chunk, header, latch)
            .next()
            .is_some()
    }

    /// The header of the numeric for loop whose latch is the instruction at
    /// `pc`, when `pc` is one that branches back.
    pub(crate) fn numeric_latch_target(chunk: &Chunk, pc: usize) -> Option<usize> {
        let instruction = *chunk.code.get(pc)?;
        if instruction.opcode() != Ok(OpCode::NumericForLoop) {
            return None;
        }
        let target = i64::try_from(pc).ok()? + 1 + i64::from(short_offset(instruction));
        usize::try_from(target).ok().filter(|target| *target <= pc)
    }

    /// The header of the loop whose closing branch is the instruction at
    /// `pc`, when it branches back: a numeric for latch, a generic for latch
    /// or the backward jump of a `while` loop. These are the loops a
    /// recording may end in front of for a loop side.
    pub(crate) fn loop_latch_target(chunk: &Chunk, pc: usize) -> Option<usize> {
        let instruction = *chunk.code.get(pc)?;
        let offset = match instruction.opcode() {
            Ok(OpCode::NumericForLoop | OpCode::ForInLoop) => short_offset(instruction),
            Ok(OpCode::Jump) => instruction.jump_offset(),
            _ => return None,
        };
        let target = i64::try_from(pc).ok()? + 1 + i64::from(offset);
        usize::try_from(target).ok().filter(|target| *target <= pc)
    }

    /// The call depth the recording runs at.
    pub(crate) const fn depth(&self) -> usize {
        self.depth
    }

    /// The coroutine the recording runs in, `None` for the main thread.
    pub(crate) const fn coroutine(&self) -> Option<ThreadHandle> {
        self.coroutine
    }

    /// A recorder that records nothing yet. It watches the interpreter run
    /// the frame's code, along the path of a queued side of a root whose loop
    /// closes at `latch` and begins at `header`, and wakes (see
    /// [`Step::Wake`]) when the interpreter fails a guard of that side that
    /// leaves the nest: the compiled code would have taken the side's own
    /// side there. `candidates` are the side's guards. It gives up after
    /// `budget` instructions, and when the frame is left.
    pub(crate) fn scouting(
        mut self,
        (latch, header): (usize, usize),
        candidates: Vec<WakeCandidate>,
        budget: u32,
    ) -> Self {
        self.scout = Some(Scout {
            latch,
            header,
            candidates,
            // A scout for the root's own exit follows from the start.
            following: header == usize::MAX,
            previous: None,
            budget,
        });
        self
    }

    /// The captured locals of the root frame the recording knows.
    pub(crate) fn captured(&self) -> Vec<(u8, CellHandle)> {
        self.root_captured.clone()
    }

    /// The headers of the loops inside the loop closed by the backward branch
    /// at `latch` to `header`, one for each of their backward branches.
    pub(crate) fn inner_loop_headers(
        chunk: &Chunk,
        header: usize,
        latch: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        let end = latch.min(chunk.code.len().saturating_sub(1));
        (header..=end).filter_map(move |pc| {
            let instruction = chunk.code[pc];
            let offset = match instruction.opcode() {
                Ok(OpCode::Jump) => instruction.jump_offset(),
                Ok(OpCode::JumpIfFalse | OpCode::NumericForLoop | OpCode::ForInLoop) => {
                    short_offset(instruction)
                }
                _ => return None,
            };
            let target = i64::try_from(pc).unwrap_or(i64::MAX) + 1 + i64::from(offset);
            let backward = target <= i64::try_from(pc).unwrap_or(i64::MAX)
                && target != i64::try_from(header).unwrap_or(i64::MAX)
                && pc != latch;
            backward.then(|| usize::try_from(target).ok()).flatten()
        })
    }

    /// The deepest call depth whose instructions the recorder still needs to
    /// see. While it sleeps through a self call, only the callee's own frame
    /// matters (its return gives an open result's type): the frames the
    /// callee calls in turn are skipped without a hook.
    pub(crate) fn hook_depth_limit(&self) -> usize {
        // A scout watches the frame only.
        if self.scout.is_some() {
            return self.depth;
        }
        self.sleeping.map_or(usize::MAX, |wait| wait.depth + 1)
    }

    /// Handles the hook before the instruction at `at.pc` runs.
    #[allow(
        clippy::too_many_lines,
        reason = "one linear check of the hook against the recorder's model"
    )]
    pub(crate) fn step(&mut self, at: &HookPoint<'_>, id: TraceId) -> Step {
        if at.hooks_active {
            return Step::Abort(TraceAbortReason::Hook);
        }
        if self.adopt_coroutine {
            self.adopt_coroutine = false;
            self.coroutine = at.coroutine;
        }
        if let Some(wait) = self.vm_wait.take() {
            // The host function runs in the interpreter. The general loop
            // hooks the call again after the fast loop declined it.
            if at.coroutine != self.coroutine || (at.depth == wait.depth && at.pc == wait.pc) {
                self.vm_wait = Some(wait);
                return Step::Skipped;
            }
            if let Err(reason) = self.finish_vm_call(at, wait) {
                return Step::Abort(reason);
            }
        }
        if let Some(wait) = self.resume_wait.take() {
            // The thread the resume runs executes its own instructions, which
            // are no part of this recording. The general loop hooks the call
            // again after the fast loop declined it (a resume runs there
            // only while no recording does).
            if at.coroutine != self.coroutine || (at.depth == wait.depth && at.pc == wait.pc) {
                self.resume_wait = Some(wait);
                return Step::Skipped;
            }
            if let Err(reason) = self.finish_resume(at, wait) {
                return Step::Abort(reason);
            }
        }
        if at.coroutine != self.coroutine {
            return Step::Abort(TraceAbortReason::Divergence);
        }
        if let Some(wait) = self.sleeping.as_mut() {
            // The callee of a self call runs in the interpreter. Its frame's
            // return is the value an open result count takes.
            if at.depth == wait.depth + 1
                && let Some(instruction) = at.chunk.code.get(at.pc)
                && matches!(
                    instruction.opcode(),
                    Ok(OpCode::Return | OpCode::ReturnOpen | OpCode::ReturnVararg | OpCode::Halt)
                )
            {
                wait.returned = (instruction.opcode() == Ok(OpCode::Return)
                    && instruction.b() == 1)
                    .then(|| {
                        at.registers
                            .get(usize::from(instruction.a()))
                            .map(|value| Kind::of(value, self.integers))
                    })
                    .flatten();
            }
            if at.depth > wait.depth {
                return Step::Skipped;
            }
            let wait = *wait;
            if let Err(reason) = self.wake(at, wait) {
                return Step::Abort(reason);
            }
        }
        if let Some(scout) = self.scout.as_mut() {
            // The frame runs too long before it leaves the loop: a nest with
            // much work in each iteration of the loop around the root, in
            // the frame or in the calls it makes.
            if scout.budget == 0 {
                return Step::Abort(limit("scout budget"));
            }
            scout.budget -= 1;
            // Calls the frame makes run unobserved.
            if at.depth > self.depth {
                return Step::Skipped;
            }
            if at.depth < self.depth || !Arc::ptr_eq(at.chunk, &self.chunk) {
                return Step::Abort(TraceAbortReason::Divergence);
            }
            return match scout.observe(at.pc) {
                Some(candidate) => Step::Wake(candidate),
                None => Step::Skipped,
            };
        }
        let here = Place::of(at.depth, at.chunk, at.pc);
        if let Some(last) = self.last {
            // The general loop reaches an instruction the fast loop hooked
            // and then declined to run. It has not executed, so the only
            // way to see its place again is a hook for the same instruction.
            if here == last.at && here != last.next {
                return Step::Skipped;
            }
            if here != last.next {
                return Step::Abort(if at.depth == last.next.depth {
                    TraceAbortReason::Divergence
                } else {
                    TraceAbortReason::DepthChange
                });
            }
        } else if at.depth != self.depth {
            return Step::Abort(TraceAbortReason::DepthChange);
        } else if !Arc::ptr_eq(at.chunk, &self.chunk) {
            return Step::Abort(TraceAbortReason::Divergence);
        }
        if at.registers.len() < usize::from(at.chunk.register_count) {
            return Step::Abort(TraceAbortReason::Divergence);
        }
        if let Err(reason) = self.check_caught(at) {
            return Step::Abort(reason);
        }
        // The interpreter runs the arm of a value select it takes; the trace
        // has its values already.
        if let Some(next) = self.skip_selected(at.pc) {
            self.last = Some(Last {
                at: here,
                next: Place::of(at.depth, at.chunk, next),
            });
            self.guest += 1;
            return Step::Recorded;
        }
        if self.started
            && (self.in_home() || self.in_loop_frame())
            && at.pc == self.header
            && self.function.is_none()
        {
            // A loop side of a loop in a call publishes its state in the
            // call's frames.
            if self.in_loop_frame() {
                self.terminal = Some(self.snapshot(
                    SnapshotPhase::After {
                        pc: self.latch,
                        successor: self.header,
                    },
                    &[],
                ));
            }
            // A pinned callee collected during the iteration would make
            // every entry mismatch.
            if !self.pinned.is_empty() {
                let alive = at.surroundings.is_some_and(|around| {
                    self.pinned.iter().all(|identity| {
                        pinned_native_index(*identity).is_some()
                            || pinned_builtin_kind(*identity).is_some()
                            || around
                                .runtime
                                .closure_call_parts(ClosureHandle::from_native_identity(*identity))
                                .is_some()
                    })
                });
                if !alive {
                    return Step::Abort(TraceAbortReason::Divergence);
                }
            }
            // The loop-back and the exits that follow carry typed values.
            if let Err(reason) = self.narrow_live_numbers(at) {
                return Step::Abort(reason);
            }
            return match self.finalize(id) {
                Ok(ir) => Step::Closed(Box::new(ClosedTrace {
                    guest_instructions: self.guest.saturating_add(self.host_work),
                    slot_reads_forwarded: self.slot_reads_forwarded,
                    slot_overwrites: self.slot_overwrites,
                    header_pc: self.header,
                    crossed: std::mem::take(&mut self.crossed),
                    remaining_trips: self.remaining_trips,
                    run_trips: self.run_trips,
                    remaining_work: self.remaining_work,
                    nest_plan: self.nest_plan,
                    entered_loop: None,
                    ir,
                })),
                Err(reason) => Step::Abort(reason),
            };
        }
        let started_now = !self.started;
        if !self.started {
            let start = self.side.as_ref().map_or(self.header, |seed| seed.start_pc);
            if at.pc != start {
                return Step::Abort(TraceAbortReason::Divergence);
            }
            self.started = true;
            // A side that starts inside inlined calls sees the root frame's
            // registers through its seed.
            let seeded = self
                .side
                .as_ref()
                .filter(|seed| !seed.frames.is_empty())
                .map(|seed| seed.root_registers.clone());
            if seeded
                .as_ref()
                .is_some_and(|registers| registers.len() < self.register_count)
            {
                return Step::Abort(TraceAbortReason::Divergence);
            }
            let root_registers: &[Value] = seeded.as_deref().unwrap_or(at.registers);
            self.entry = root_registers[..self.register_count]
                .iter()
                .map(|value| Kind::of(value, self.integers))
                .collect();
            self.start_tables = root_registers[..self.register_count]
                .iter()
                .enumerate()
                .filter_map(|(register, value)| match value {
                    Value::Table(table) => Some((register, *table)),
                    _ => None,
                })
                .collect();
            self.regs = vec![Sym::Carried; self.register_count];
            self.written = vec![false; self.register_count];
            if let Err(reason) = self.start_side(at) {
                return Step::Abort(reason);
            }
            if let Err(reason) = self.start_function() {
                return Step::Abort(reason);
            }
            self.start_lazy(at.pc);
            if self.side.is_none()
                && self.function.is_none()
                && self.in_home()
                && let Err(reason) = self.reload_ipairs_element(at)
            {
                return Step::Abort(reason);
            }
            if self.in_loop_frame() {
                self.loop_frame_start = self.frames.last().expect("a loop frame").regs.clone();
                if let Err(reason) = self.reload_ipairs_element(at) {
                    return Step::Abort(reason);
                }
                let frame = self.frames.last().expect("a loop frame");
                if self
                    .reloaded
                    .iter()
                    .any(|(register, nil)| frame.regs.get(*register) == Some(&Sym::Op(*nil)))
                {
                    return Step::Abort(TraceAbortReason::UnsupportedOperand);
                }
            }
        }
        if started_now && let Some(closed) = self.close_side_at_start(at, id) {
            return closed;
        }
        // A recording that unrolled loops stops early: a big nest compiles
        // too slowly to repay a short run.
        let guest_limit = if self.unrolled_loops > 0 {
            if self.side.is_some() {
                MAX_UNROLLED_SIDE_GUEST_INSTRUCTIONS
            } else {
                MAX_UNROLLED_GUEST_INSTRUCTIONS
            }
        } else {
            self.limits.guest_ops
        };
        if self.guest as usize >= guest_limit {
            return Step::Abort(limit("guest instructions"));
        }
        if self.ops.len() >= self.limits.ir_ops {
            return Step::Abort(limit("IR operations"));
        }
        // One instruction (the for-loop latch) can add two snapshots. A
        // snapshot no guard leaves through is never taken in LuaJIT and does
        // not count.
        if self.guarded_snapshots + 2 > self.limits.snapshots {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: {} guest instructions at the snapshot cap ({} array reads forwarded, {} stores proven)",
                    self.guest, self.slot_reads_forwarded, self.slot_overwrites
                );
            }
            return Step::Abort(limit("snapshots"));
        }
        let depth = at.depth;
        self.counted_now = u32::from(
            self.counting
                && at
                    .chunk
                    .code
                    .get(at.pc)
                    .and_then(|instruction| instruction.opcode().ok())
                    .is_none_or(OpCode::counts_for_hooks),
        );
        let next = match self.record(at) {
            Ok(Flow::Next(next)) if self.in_home() => {
                if !self.outer_side && (next < self.header || next > self.latch) {
                    // The iteration left the loop (a break, a loop test, a
                    // fall-through at the latch). Whether it does depends on
                    // the iteration recording started on, so a later one may
                    // close.
                    return Step::Abort(TraceAbortReason::LoopNotClosed);
                }
                self.set_prev(PrevPc::At(at.pc));
                Place::of(depth, at.chunk, next)
            }
            Ok(Flow::Next(next)) => {
                // Inside an inlined call, a pc outside the callee's code is a
                // model failure.
                if next >= at.chunk.code.len() {
                    return Step::Abort(TraceAbortReason::Divergence);
                }
                self.set_prev(PrevPc::At(at.pc));
                Place::of(depth, at.chunk, next)
            }
            Ok(Flow::Enter) => {
                let callee = self.frames.last().expect("a call was just inlined");
                Place::of(depth + 1, &callee.chunk, 0)
            }
            Ok(Flow::Replace) => {
                let callee = self.frames.last().expect("a tail call was just inlined");
                Place::of(depth, &callee.chunk, 0)
            }
            Ok(Flow::ReplaceRoot) => Place::of(depth, self.root_chunk(), 0),
            Ok(Flow::Leave { resume }) => {
                // The frame returned into is the innermost one left.
                let caller = self
                    .frames
                    .last()
                    .map_or_else(|| self.root_chunk(), |frame| &frame.chunk);
                Place::of(depth - 1, caller, resume)
            }
            Ok(Flow::LeaveRoot { resume }) => {
                let caller = self
                    .frames
                    .last()
                    .map_or_else(|| self.root_chunk(), |frame| &frame.chunk);
                Place::of(depth - 1, caller, resume)
            }
            Ok(Flow::EnterRoot) => Place::of(depth + 1, self.root_chunk(), 0),
            Ok(Flow::Caught { resume, levels }) => {
                let caller = self
                    .frames
                    .last()
                    .map_or_else(|| self.root_chunk(), |frame| &frame.chunk);
                Place::of(depth - levels, caller, resume)
            }
            Ok(Flow::CaughtHandler { levels }) => {
                let handler = self.frames.last().expect("a handler frame was pushed");
                Place::of(depth - levels + 1, &handler.chunk, 0)
            }
            Ok(Flow::Sleep) => Place::of(depth, at.chunk, at.pc + 1),
            Ok(Flow::Exit) => {
                if let Err(reason) = self.narrow_live_numbers(at) {
                    return Step::Abort(reason);
                }
                return match self.finalize(id) {
                    Ok(ir) => Step::Closed(Box::new(ClosedTrace {
                        guest_instructions: self.guest.saturating_add(self.host_work),
                        slot_reads_forwarded: self.slot_reads_forwarded,
                        slot_overwrites: self.slot_overwrites,
                        header_pc: self.header,
                        crossed: std::mem::take(&mut self.crossed),
                        remaining_trips: self.remaining_trips,
                        run_trips: self.run_trips,
                        remaining_work: self.remaining_work,
                        nest_plan: self.nest_plan,
                        // A link to an ancestor loop side that could not bind
                        // ends in front of a copy of that loop.
                        entered_loop: self.enter_loop.or_else(|| {
                            self.link_loop
                                .filter(|_| matches!(ir.terminator, TraceTerminator::Exit { .. }))
                                .map(|position| {
                                    let held = &self.loop_headers[position];
                                    (held.header, held.latch)
                                })
                        }),
                        ir,
                    })),
                    Err(reason) => Step::Abort(reason),
                };
            }
            Ok(Flow::Return) => {
                self.guest += 1;
                return match self.finalize(id) {
                    Ok(ir) => Step::Closed(Box::new(ClosedTrace {
                        guest_instructions: self.guest.saturating_add(self.host_work),
                        slot_reads_forwarded: self.slot_reads_forwarded,
                        slot_overwrites: self.slot_overwrites,
                        header_pc: self.site_pc(),
                        crossed: Vec::new(),
                        remaining_trips: None,
                        run_trips: None,
                        remaining_work: None,
                        nest_plan: None,
                        entered_loop: None,
                        ir,
                    })),
                    Err(reason) => Step::Abort(reason),
                };
            }
            Err(reason) => return Step::Abort(reason),
        };
        self.last = Some(Last { at: here, next });
        self.counted += self.counted_now;
        self.guest += 1;
        // A constant-operand opcode stands for a constant load and the
        // register operation, and a test for a comparison and its branch.
        // The trace policy weighs work in those instructions, so each counts
        // as all of them.
        if let Some(opcode) = at
            .chunk
            .code
            .get(at.pc)
            .and_then(|instruction| instruction.opcode().ok())
        {
            if opcode.register_form().is_some() || opcode.is_test_constant() {
                self.guest += 1;
            }
            if opcode.is_test() {
                self.guest += 1;
            }
            // `ForInCall` stands for three copies and a call, `ForInLoop`
            // for a copy, a test and the jump back.
            if opcode == OpCode::ForInCall {
                self.guest += 3;
            }
            if opcode == OpCode::ForInLoop {
                self.guest += 2;
            }
            // `GetTabUp` and `SetTabUp` stand for the upvalue read and the
            // field access, `SelfField` for a copy and a field read.
            if matches!(
                opcode,
                OpCode::GetTabUp | OpCode::SetTabUp | OpCode::SelfField
            ) {
                self.guest += 1;
            }
        }
        Step::Recorded
    }

    /// Sets up the window slots and the registers the parent state holds as
    /// string constants, for a side recording. A root recording adds its
    /// slots as it meets them, so this does nothing there.
    fn start_side(&mut self, at: &HookPoint<'_>) -> Result<(), TraceAbortReason> {
        let Some(seed) = self.side.clone() else {
            return Ok(());
        };
        for &(register, slot) in &seed.roots {
            if register < self.register_count {
                self.regs[register] = Sym::Root(slot);
            }
        }
        // The tree's control lane: the register holds a key of any type in
        // the interpreter, and the lane stands for it.
        if let Some(register) = seed
            .boxed
            .filter(|register| *register < self.register_count)
        {
            self.box_loop(register);
        }
        for &register in &seed.borrowed {
            if register < self.register_count && self.entry[register] == Kind::String {
                self.entry[register] = Kind::Borrowed;
            }
        }
        let root_view = if seed.frames.is_empty() {
            None
        } else {
            Some(RootView::of(at, &seed, &self.chunk)?)
        };
        for slot in &seed.externals {
            // A function trace's tail call count is no Lua value: an integer.
            if *slot == ExternalSlot::TailCalls && self.function.is_some() {
                self.entry.push(Kind::Integer);
                self.regs.push(Sym::Carried);
                self.written.push(false);
                continue;
            }
            // A register of the root frame's caller is not in the frame the
            // hook shows: the tree's entry says what it holds.
            if matches!(
                slot,
                ExternalSlot::CallerRegister(_) | ExternalSlot::CallerGlobal(_)
            ) && self.caller_seed().is_none()
            {
                let position = self.entry.len() - self.register_count;
                let ty = seed
                    .external_types
                    .get(position)
                    .copied()
                    .flatten()
                    .ok_or(TraceAbortReason::Divergence)?;
                self.entry.push(Kind::from_type(ty));
                self.regs.push(Sym::Carried);
                self.written.push(false);
                continue;
            }
            // The hook of a side that starts inside inlined calls runs in
            // the innermost callee: the root frame's view reads the slot.
            let live = match &root_view {
                Some(view) => self.external_value(&view.at(at), *slot)?,
                None => self.external_value(at, *slot)?,
            };
            let kind = Kind::of(&live, self.integers);
            if kind == Kind::Other {
                self.note_callable(&live);
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            // The cell behind the slot, so that a cell the side names later
            // under another slot (a callee's, say) is known to be this one.
            let position = self.entry.len() - self.register_count;
            self.external_cells[position] = match slot {
                ExternalSlot::Upvalue(index) => match &root_view {
                    Some(view) => Self::upvalue_cell(&view.at(at), *index).ok().map(|c| c.0),
                    None => Self::upvalue_cell(at, *index).ok().map(|c| c.0),
                },
                ExternalSlot::ClosureUpvalue { closure, index } => self
                    .pinned_closure_parts(at, *closure)
                    .ok()
                    .and_then(|(cells, _)| cells.get(usize::from(*index)).copied()),
                _ => None,
            };
            self.entry.push(kind);
            self.regs.push(Sym::Carried);
            self.written.push(false);
        }
        self.start_seed_frames(at, &seed)
    }

    /// Rebuilds the inlined calls a side starts inside (see [`SeedFrame`]):
    /// one frame per call, outermost first, whose registers are frame inputs
    /// typed by what the exit restored, or the root slots of string
    /// constants. The hook runs in the innermost of them, at the seed's
    /// start.
    fn start_seed_frames(
        &mut self,
        at: &HookPoint<'_>,
        seed: &SideSeed,
    ) -> Result<(), TraceAbortReason> {
        let Some(innermost) = seed.frames.last() else {
            return Ok(());
        };
        if !Arc::ptr_eq(at.chunk, &innermost.chunk) || seed.frames.len() + 1 > MAX_TRACE_FRAME_DEPTH
        {
            return Err(TraceAbortReason::Divergence);
        }
        let site = seed.frames[0].info.call_pc;
        for (position, frame) in seed.frames.iter().enumerate() {
            let id = u32::try_from(position + 1).expect("seed frames fit u32");
            let innermost = position + 1 == seed.frames.len();
            let register_count = usize::from(frame.chunk.register_count);
            // A frame that runs whichever closure of the prototype its call
            // had holds that closure in the register after the chunk's.
            let dynamic = frame.info.dynamic;
            if frame.registers.len() != register_count + usize::from(dynamic)
                || (innermost && at.registers.len() < register_count)
            {
                return Err(TraceAbortReason::Divergence);
            }
            if dynamic && !self.may_use_dynamic_cells() {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            let mut regs = Vec::with_capacity(register_count);
            let mut closure = None;
            for (register, held) in frame.registers.iter().enumerate() {
                if dynamic && register == register_count {
                    let SeedRegister::Live(Value::Closure(_)) = held else {
                        return Err(TraceAbortReason::Divergence);
                    };
                    let register = u16::try_from(register).expect("register fits u16");
                    closure = Some(self.emit(
                        site,
                        RecKind::FrameInput {
                            frame: id,
                            register,
                        },
                        Some(TraceType::Closure),
                    ));
                    continue;
                }
                let sym = match held {
                    SeedRegister::Root(slot) => Sym::Root(*slot),
                    // A stand-in that the reload at the header replaces.
                    SeedRegister::Reloaded => {
                        // Recorded in the frame that made this one, at its
                        // call.
                        let nil = self.emit(
                            frame.info.call_pc,
                            RecKind::Constant(TraceScalar::Nil),
                            Some(TraceType::Nil),
                        );
                        self.reloaded.push((register, nil));
                        Sym::Op(nil)
                    }
                    SeedRegister::Dead => Sym::Op(self.emit(
                        frame.info.call_pc,
                        RecKind::Constant(TraceScalar::Nil),
                        Some(TraceType::Nil),
                    )),
                    SeedRegister::Live(value) => {
                        let kind = Kind::of(value, self.integers);
                        // The innermost frame is the one the hook shows.
                        if innermost && Kind::of(&at.registers[register], self.integers) != kind {
                            return Err(TraceAbortReason::Divergence);
                        }
                        let ty = kind
                            .trace_type()
                            .filter(|ty| !ty.is_borrowed())
                            .ok_or(TraceAbortReason::UnsupportedOperand)?;
                        let register = u16::try_from(register).expect("register fits u16");
                        Sym::Op(self.emit(
                            site,
                            RecKind::FrameInput {
                                frame: id,
                                register,
                            },
                            Some(ty),
                        ))
                    }
                };
                regs.push(sym);
            }
            let mut info = frame.info;
            info.parent = FrameId(id - 1);
            self.inlined.push(TraceFrameInfo {
                register_count: u16::try_from(register_count + usize::from(dynamic))
                    .expect("register count fits u16"),
                code_len: frame.chunk.code.len(),
                inlined: Some(info),
                activation: None,
            });
            if dynamic {
                self.dynamic_cells = true;
            }
            // Each frame stands at the call that made the next one, and the
            // innermost at what the exit restored.
            let prev = match seed.frames.get(position + 1) {
                Some(next) => PrevPc::At(next.info.call_pc),
                None => frame.last.map_or(PrevPc::Unset, PrevPc::At),
            };
            self.frames.push(Inlined {
                chunk: Arc::clone(&frame.chunk),
                closure: info.closure,
                frame: id,
                regs,
                prev,
                call_pc: info.call_pc,
                result_base: usize::from(info.result_base),
                results: usize::from(info.results),
                varargs: Vec::new(),
                owns_reexec: false,
                tail_calls: 0,
                protected: Protected::No,
                handler: false,
                entry: Entry::Call,
                after: Pending::Nothing,
                dynamic: closure,
            });
        }
        Ok(())
    }

    /// Records the instruction at `at.pc` and says where execution goes.
    fn record(&mut self, at: &HookPoint<'_>) -> Result<Flow, TraceAbortReason> {
        self.current_pc = at.pc;
        self.abort_metamethod = None;
        let flow = match self.record_instruction(at) {
            Ok(flow) => flow,
            Err(reason) => {
                if reason == TraceAbortReason::UnsupportedOperand
                    && let Some(instruction) = at.chunk.code.get(at.pc)
                {
                    self.note_failed_callable(at, *instruction);
                }
                return Err(reason);
            }
        };
        if let Some(reason) = self.deferred.take() {
            return Err(reason);
        }
        if self.field_cell_written() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if self.borrow_ended {
            return Err(TraceAbortReason::BorrowOutlivesEffect);
        }
        Ok(flow)
    }

    fn record_instruction(&mut self, at: &HookPoint<'_>) -> Result<Flow, TraceAbortReason> {
        // The side ends in front of the loop it left to a loop side, with
        // the closing branch committed and the interpreter at the loop's
        // header.
        if let Some((header, latch)) = self.enter_loop
            && at.pc == header
            && self.frames.len() == self.enter_depth
        {
            self.exit_at = Some(at.pc);
            // In front of a loop in a call: the exit rebuilds the call's
            // frames, from a snapshot of them all.
            if self.enter_depth > 0 {
                self.own_frame_borrows(at, latch);
                self.terminal = Some(self.snapshot(
                    SnapshotPhase::After {
                        pc: latch,
                        successor: header,
                    },
                    &[],
                ));
            }
            return Ok(Flow::Exit);
        }
        // A side that took the closing branch of an ancestor loop side ends
        // at its header, where the loop goes on.
        if let Some(position) = self.link_loop
            && at.pc == self.loop_headers[position].header
            && self.frames.is_empty()
        {
            self.exit_at = Some(at.pc);
            return Ok(Flow::Exit);
        }
        let instruction = *at
            .chunk
            .code
            .get(at.pc)
            .ok_or(TraceAbortReason::Divergence)?;
        // The values of an open-result call belong to the call or return
        // that takes them, which follows it after at most some moves (of the
        // callee into place, say). Nothing else may run between.
        let takes_open = matches!(
            instruction.opcode(),
            Ok(OpCode::CallOpen | OpCode::TailCallOpen | OpCode::ReturnOpen)
        );
        if takes_open != self.open.is_some()
            && !(self.open.is_some() && instruction.opcode() == Ok(OpCode::Move))
        {
            return Err(TraceAbortReason::Divergence);
        }
        // The region in the root frame's caller runs a few kinds of
        // instructions, and reads its registers from their window slots.
        if self.running_caller() {
            let closure = self
                .frames
                .first()
                .is_some_and(|frame| frame.closure != linked::NO_CLOSURE);
            if !instruction
                .opcode()
                .is_ok_and(|opcode| Self::caller_may_run(opcode, closure))
            {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            self.load_caller_reads(at)?;
        }
        // A number of run-time subtype reaches only the instructions that
        // take one; the others read it as the subtype it has now.
        self.narrow_numbers_read(at, instruction)?;
        // A call or return takes its registers as values: a virtual table
        // among them is made real first.
        if let Ok(
            opcode @ (OpCode::Call
            | OpCode::CallOne
            | OpCode::CallVararg
            | OpCode::CallOpen
            | OpCode::TailCall
            | OpCode::TailCallVararg
            | OpCode::TailCallOpen
            | OpCode::Return
            | OpCode::ReturnOpen
            | OpCode::ReturnVararg),
        ) = instruction.opcode()
        {
            self.escape_virtual_reads(at, instruction, opcode)?;
        }
        match instruction.opcode() {
            Ok(
                OpCode::Call
                | OpCode::CallOne
                | OpCode::CallVararg
                | OpCode::CallOpen
                | OpCode::TailCall
                | OpCode::TailCallVararg
                | OpCode::TailCallOpen,
            ) => {
                if self.running_caller() && self.is_root_call(at.pc) {
                    return self.call_root_again(at, instruction);
                }
                return self.record_call(at, instruction);
            }
            Ok(OpCode::ForInCall) => {
                return self.record_for_in_call(at, instruction);
            }
            Ok(OpCode::Return | OpCode::ReturnOpen | OpCode::Halt) if !self.frames.is_empty() => {
                return self.record_return(at, instruction);
            }
            Ok(OpCode::Return | OpCode::ReturnOpen | OpCode::ReturnVararg | OpCode::Halt)
                if self.function.is_some() =>
            {
                return self.record_function_return(at, instruction);
            }
            // The root frame's return runs on in its caller.
            Ok(OpCode::Return | OpCode::ReturnOpen | OpCode::ReturnVararg | OpCode::Halt)
                if self.may_return_through() =>
            {
                return self.return_through(at, instruction);
            }
            // A side with little work before the return would only move the
            // exit there, at the price of compiling the tree again.
            Ok(OpCode::Return | OpCode::ReturnOpen | OpCode::ReturnVararg | OpCode::Halt)
                if self.frames.is_empty()
                    && ((self.beyond && self.guest >= MIN_BEYOND_WORK)
                        || self.returns_to_interpreter()) =>
            {
                self.exit_at = Some(at.pc);
                return Ok(Flow::Exit);
            }
            _ => {}
        }
        if let Some(flow) = self.record_metamethod(at, instruction)? {
            return Ok(flow);
        }
        self.record_plain(at).map(Flow::Next)
    }

    /// Whether a plain side trace that reaches a `Return` of the root frame
    /// ends there: with an exit that resumes the interpreter in front of the
    /// instruction. The return has no loop to link to (no other loop of the
    /// function encloses the root's, and no caller was seeded), so the side
    /// runs natively up to it, as `LuaJIT`'s return link does.
    fn returns_to_interpreter(&self) -> bool {
        self.side.as_ref().is_some_and(|seed| seed.return_exit)
            && !self.counting
            && !self.looping
            && self.function.is_none()
            && self.guest >= 1
    }

    /// Records an instruction that stays in its frame and returns the pc that
    /// follows it.
    #[allow(clippy::too_many_lines)]
    fn record_plain(&mut self, at: &HookPoint<'_>) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let registers = at.registers;
        let instruction = *at.chunk.code.get(pc).ok_or(TraceAbortReason::Divergence)?;
        let opcode = instruction
            .opcode()
            .map_err(|_| TraceAbortReason::UnsupportedOpcode)?;
        let a = usize::from(instruction.a());
        let b = usize::from(instruction.b());
        let c = usize::from(instruction.c());
        // `SelfField` is a copy of the receiver and the `GetField` of the
        // method from it, in that order, so it records as the two.
        let (instruction, opcode) = if opcode == OpCode::SelfField {
            if matches!(registers.get(b), Some(Value::String(_))) {
                return self.record_string_method(at, instruction, true);
            }
            self.record_move(registers, pc, a + 1, b)?;
            (
                Instruction::get_field(instruction.a(), instruction.b(), instruction.c()),
                OpCode::GetField,
            )
        } else {
            (instruction, opcode)
        };
        self.escape_virtual_reads(at, instruction, opcode)?;
        match opcode {
            OpCode::Move => {
                self.record_move(registers, pc, a, b)?;
                Ok(pc + 1)
            }
            OpCode::LoadConstant => {
                self.check_register(a)?;
                if let Some(Constant::String(text)) =
                    at.chunk.constants.get(usize::from(instruction.d()))
                {
                    // A string only rides to a snapshot, as a root string.
                    let slot = self.root_slot(text.as_bytes())?;
                    self.write(a, Sym::Root(slot));
                    return Ok(pc + 1);
                }
                let scalar = self.constant(at.chunk, instruction)?;
                let value = self.emit(pc, RecKind::Constant(scalar), Some(scalar.ty()));
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::LoadNil => {
                self.check_register(a)?;
                let value = self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Nil),
                    Some(TraceType::Nil),
                );
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::ClearRange => {
                if b > 0 {
                    self.check_register(a + b - 1)?;
                    let value = self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Nil),
                        Some(TraceType::Nil),
                    );
                    for register in a..a + b {
                        self.write(register, value);
                    }
                }
                Ok(pc + 1)
            }
            OpCode::Not => {
                self.check_register(a)?;
                let (source, _) = self.read(registers, b)?;
                let value = self.emit(pc, RecKind::Not(source), Some(TraceType::Boolean));
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::Negate => {
                self.check_register(a)?;
                if matches!(registers[b], Value::Int64(_) | Value::UInt64(_)) {
                    let (source, ty) = self.read(registers, b)?;
                    let value = self.emit(pc, RecKind::Negate(source), Some(ty));
                    self.write(a, value);
                    return Ok(pc + 1);
                }
                let source = self.read_arithmetic(at, b, Some(7))?;
                let value = self.emit(pc, RecKind::Negate(source.operand), Some(source.ty()));
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::BitNot => {
                self.check_register(a)?;
                let source = self.read_number(registers, b)?;
                if source.ty() != TraceType::Integer {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // `~x` is `x ~ -1`.
                let ones = self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Integer(-1)),
                    Some(TraceType::Integer),
                );
                let value = self.emit(
                    pc,
                    RecKind::Arithmetic(ArithmeticOp::BitXor, source.operand, Operand::Op(ones)),
                    Some(TraceType::Integer),
                );
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::Add
            | OpCode::Subtract
            | OpCode::Multiply
            | OpCode::Divide
            | OpCode::Modulo
            | OpCode::AddK
            | OpCode::SubtractK
            | OpCode::MultiplyK
            | OpCode::DivideK
            | OpCode::ModuloK
            | OpCode::Power
            | OpCode::FloorDivide
            | OpCode::BitAnd
            | OpCode::BitOr
            | OpCode::BitXor
            | OpCode::ShiftLeft
            | OpCode::ShiftRight => {
                self.check_register(a)?;
                // A constant operand records as the constant loaded into a
                // register, then the register form.
                let (opcode, constant) = match register_form(opcode) {
                    Some(register_opcode) => {
                        let scalar = self.constant_at(at.chunk, c)?;
                        let operand = Operand::Op(self.emit(
                            pc,
                            RecKind::Constant(scalar),
                            Some(scalar.ty()),
                        ));
                        (register_opcode, Some((operand, scalar)))
                    }
                    None => (opcode, None),
                };
                if let Some(value) = self.record_number_arithmetic(at, opcode, (b, c), constant)? {
                    self.write(a, Sym::Op(value));
                    return Ok(pc + 1);
                }
                let wide_right = constant.is_none()
                    && matches!(registers[c], Value::Int64(_) | Value::UInt64(_));
                if wide_right || matches!(registers[b], Value::Int64(_) | Value::UInt64(_)) {
                    let value = self.record_wide_arithmetic(
                        registers,
                        opcode,
                        (b, c),
                        constant.map(|(operand, scalar)| (operand, scalar.ty())),
                        pc,
                    )?;
                    self.write(a, value);
                    return Ok(pc + 1);
                }
                // The index of the operator among the string metatable's
                // arithmetic metamethods; the bitwise ones have none, and a
                // string there is the interpreter's.
                let operator = match opcode {
                    OpCode::Add => Some(0),
                    OpCode::Subtract => Some(1),
                    OpCode::Multiply => Some(2),
                    OpCode::Divide => Some(3),
                    OpCode::Modulo => Some(4),
                    OpCode::Power => Some(5),
                    OpCode::FloorDivide => Some(6),
                    _ => None,
                };
                let left = self.read_arithmetic(at, b, operator)?;
                let right = match constant {
                    Some((operand, TraceScalar::Float(value))) => Number {
                        operand,
                        live: Live::Float(value),
                    },
                    Some((operand, TraceScalar::Integer(value))) => Number {
                        operand,
                        live: Live::Integer(value),
                    },
                    Some(_) => return Err(TraceAbortReason::UnsupportedOperand),
                    None => self.read_arithmetic(at, c, operator)?,
                };
                let op = match opcode {
                    OpCode::Add => ArithmeticOp::Add,
                    OpCode::Subtract => ArithmeticOp::Subtract,
                    OpCode::Multiply => ArithmeticOp::Multiply,
                    OpCode::Divide => ArithmeticOp::Divide,
                    OpCode::Modulo => ArithmeticOp::Modulo,
                    OpCode::Power => ArithmeticOp::Power,
                    OpCode::FloorDivide => ArithmeticOp::FloorDivide,
                    OpCode::BitAnd => ArithmeticOp::BitAnd,
                    OpCode::BitOr => ArithmeticOp::BitOr,
                    OpCode::BitXor => ArithmeticOp::BitXor,
                    OpCode::ShiftLeft => ArithmeticOp::ShiftLeft,
                    _ => ArithmeticOp::ShiftRight,
                };
                let integers = left.ty() == TraceType::Integer && right.ty() == TraceType::Integer;
                if !op.allows_floats() && !integers {
                    // The interpreter converts an integral float or a string
                    // here. Leave that to it.
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let value = if integers && op.allows_integers() {
                    if matches!(op, ArithmeticOp::Modulo | ArithmeticOp::FloorDivide) {
                        let Live::Integer(divisor) = right.live else {
                            return Err(TraceAbortReason::Divergence);
                        };
                        // A zero divisor raises an error, which the
                        // interpreter does by running the instruction.
                        if divisor == 0 {
                            return Err(TraceAbortReason::UnsupportedOperand);
                        }
                        self.guard_nonzero(pc, right.operand);
                    }
                    self.emit(
                        pc,
                        RecKind::Arithmetic(op, left.operand, right.operand),
                        Some(TraceType::Integer),
                    )
                } else {
                    // A float operand, or a division: both sides as floats.
                    let left = self.float_operand(pc, left);
                    let right = self.float_operand(pc, right);
                    let op = match op {
                        ArithmeticOp::Modulo if self.float_arith.modulo != FloatModulo::Floor => {
                            ArithmeticOp::ModuloWith(self.float_arith.modulo)
                        }
                        ArithmeticOp::Power if self.float_arith.square_shortcut => {
                            ArithmeticOp::PowerSquare
                        }
                        op => op,
                    };
                    self.emit(
                        pc,
                        RecKind::Arithmetic(op, left, right),
                        Some(TraceType::Float),
                    )
                };
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::GetUpvalue if self.dynamic_closure().is_some() => {
                self.check_register(a)?;
                self.record_dynamic_get_upvalue(at, a, instruction.d())
            }
            OpCode::SetUpvalue if self.dynamic_closure().is_some() => {
                self.record_dynamic_set_upvalue(at, a, instruction.d())
            }
            OpCode::GetUpvalue => {
                self.check_register(a)?;
                // A callee's cell is a window slot like the root's. Whether it
                // is one the tree names under another slot is a promise the
                // entry checks (see `upvalues_alias`), side or root.
                let slot = self.upvalue_slot(instruction.d());
                let external = self.external(at, slot)?;
                let (source, ty) = self.read_external(at, external)?;
                let value = self.emit(pc, RecKind::Move(source), Some(ty));
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::SetUpvalue => {
                // Every activation of a function trace shares the cells.
                if self.function.is_some() {
                    return Err(TraceAbortReason::SelfCall);
                }
                // The cell takes the register's value. The trace keeps it in
                // the window register of the upvalue and writes the cell back
                // at every exit.
                let (source, ty) = self.read(registers, a)?;
                let slot = self.upvalue_slot(instruction.d());
                let external = self.external(at, slot)?;
                self.read_external(at, external)?;
                let value = self.emit(pc, RecKind::Move(source), Some(ty));
                self.write_external(external.register, value);
                Ok(pc + 1)
            }
            OpCode::GetGlobal if self.stores_global(at, u32::from(instruction.d())) => {
                // The trace stores this global, so it reads it from the
                // environment table like the store does.
                let name = self.global_name(at, u32::from(instruction.d()))?;
                let receiver = self.environment_receiver(at)?;
                self.record_get_keyed(at, a, TableKey::Field(name), receiver)
            }
            OpCode::SetGlobal => {
                if self.function.is_some() {
                    return Err(TraceAbortReason::SelfCall);
                }
                if self.reexec.is_some() {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let name = self.global_name(at, u32::from(instruction.d()))?;
                // A slot holds the value the tree read at entry, which this
                // store would leave stale (see `stores_global`).
                if self.holds_global_slot(at, &name) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let receiver = self.environment_receiver(at)?;
                self.record_set_keyed(at, a, TableKey::Field(name), receiver)
            }
            OpCode::GetGlobal => {
                self.check_register(a)?;
                self.guard_dynamic_environment(pc)?;
                let slot = self.global_slot(u32::from(instruction.d()));
                let external = self.external(at, slot)?;
                let (source, ty) = self.read_external(at, external)?;
                let value = self.emit(pc, RecKind::Move(source), Some(ty));
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::Vararg => {
                // Exactly `b` of the call's extra values, nil past the end.
                let count = b;
                if count > 0 {
                    self.check_register(a + count - 1)?;
                }
                let mut values = self.current_varargs(at)?;
                values.truncate(count);
                if values.len() < count {
                    let nil = self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Nil),
                        Some(TraceType::Nil),
                    );
                    values.resize(count, Sym::Op(nil));
                }
                for (index, value) in values.into_iter().enumerate() {
                    self.write(a + index, value);
                }
                Ok(pc + 1)
            }
            OpCode::NewTable => self.record_new_table(at, instruction),
            OpCode::GetTable | OpCode::GetField
                if self.virtual_in(usize::from(instruction.b())).is_some() =>
            {
                self.record_virtual_get(at, instruction, opcode == OpCode::GetField)
            }
            OpCode::SetTable | OpCode::SetField
                if self.virtual_in(usize::from(instruction.a())).is_some() =>
            {
                self.record_virtual_set(at, instruction, opcode == OpCode::SetField)
            }
            OpCode::Length if self.virtual_in(usize::from(instruction.b())).is_some() => {
                self.record_virtual_length(at, instruction)
            }
            OpCode::GetField if matches!(registers.get(b), Some(Value::String(_))) => {
                self.record_string_method(at, instruction, false)
            }
            OpCode::GetTable => self.record_get_table(at, instruction, false),
            OpCode::SetTable => self.record_set_table(at, instruction, false),
            OpCode::GetField => self.record_get_table(at, instruction, true),
            OpCode::SetField => self.record_set_table(at, instruction, true),
            // The table is the upvalue's value, found where `GetUpvalue`
            // would read it, and never held in a register.
            OpCode::GetTabUp if self.side.is_some() && !self.frames.is_empty() => {
                self.record_side_global(at, instruction)
            }
            OpCode::GetTabUp => {
                let receiver = self.tab_up_receiver(at, u16::from(instruction.b()))?;
                if let Some(next) =
                    self.record_tab_up_table(at, instruction, (receiver.1, receiver.2))?
                {
                    return Ok(next);
                }
                self.record_get_table_of(at, instruction, true, receiver)
            }
            OpCode::SetTabUp => {
                // A function trace and a re-execution region refuse before
                // the upvalue is looked at, as for `SetField`.
                if self.function.is_some() {
                    return Err(TraceAbortReason::SelfCall);
                }
                if self.reexec.is_some() {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let receiver = self.tab_up_receiver(at, u16::from(instruction.a()))?;
                self.record_set_table_of(at, instruction, true, receiver)
            }
            OpCode::Length => self.record_length(at, instruction),
            OpCode::Concat => self.record_concat(at, instruction),
            OpCode::Equal | OpCode::LessThan | OpCode::LessEqual => {
                self.check_register(a)?;
                if opcode == OpCode::Equal
                    && let Some(outcome) = self.root_strings_equal(registers, b, c)
                {
                    let value = self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Boolean(outcome)),
                        Some(TraceType::Boolean),
                    );
                    self.write(a, value);
                    return Ok(pc + 1);
                }
                if opcode == OpCode::Equal
                    && let Some(outcome) = self.slice_equals_register(at, b, c)?
                {
                    let value = self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Boolean(outcome)),
                        Some(TraceType::Boolean),
                    );
                    self.write(a, value);
                    return Ok(pc + 1);
                }
                // A loaded string constant takes part in an equality as its
                // root string.
                let (left, right) = if opcode == OpCode::Equal {
                    (
                        self.host_operand(pc, registers, b)?,
                        self.host_operand(pc, registers, c)?,
                    )
                } else {
                    (self.read(registers, b)?, self.read(registers, c)?)
                };
                let comparison = match opcode {
                    OpCode::Equal => Comparison::Equal,
                    OpCode::LessThan => Comparison::LessThan,
                    _ => Comparison::LessEqual,
                };
                let value = self.record_comparison(
                    pc,
                    comparison,
                    left,
                    right,
                    Some((&registers[b], &registers[c])),
                )?;
                self.write(a, value);
                Ok(pc + 1)
            }
            OpCode::TestEqual
            | OpCode::TestLessThan
            | OpCode::TestLessEqual
            | OpCode::TestEqualK
            | OpCode::TestLessThanK
            | OpCode::TestLessEqualK => self.record_test(at, instruction),
            OpCode::Jump => {
                let target = self.target(pc, instruction.jump_offset())?;
                if target <= pc && self.links_loop(pc, target) {
                    return Ok(target);
                }
                if target <= pc && self.wants_loop_side_at(pc, target, LoopKind::While) {
                    // The recording ends in front of the loop's next trip.
                    self.enter_loop = Some((target, pc));
                    self.enter_depth = self.frames.len();
                    return Ok(target);
                }
                match self.check_branch(pc, target) {
                    // The backward jump of a `while` loop. A recording
                    // that unrolls short loops follows it like the latch of
                    // a numeric loop: the loop's test is a guard on every
                    // trip, so a run of another length leaves there.
                    Err(TraceAbortReason::InnerLoop) if self.may_unroll_while(pc, target) => {
                        self.unrolled_loops += 1;
                        self.unrolled_whiles += 1;
                    }
                    Err(TraceAbortReason::InnerLoop) => {
                        // Tried unrolled already (or cannot be): the next
                        // recording meets the same loop.
                        self.met_long_loop = self.unroll_loops;
                        return Err(TraceAbortReason::InnerLoop);
                    }
                    other => other?,
                }
                Ok(target)
            }
            OpCode::JumpIfFalse => self.jump_if_false(at, instruction),
            OpCode::NumericForPrep => self.numeric_for_prep(at, instruction),
            OpCode::NumericForLoop => self.numeric_for_loop(at, instruction),
            OpCode::ForInLoop => self.for_in_loop(at, instruction),
            // The iteration leaves the function. A later iteration may not.
            // Inside an inlined call only a fixed `Return` is followed, and
            // that is handled before this point.
            OpCode::Return | OpCode::ReturnVararg | OpCode::ReturnOpen | OpCode::Halt => {
                Err(if self.frames.is_empty() {
                    TraceAbortReason::LoopNotClosed
                } else {
                    TraceAbortReason::UnsupportedOpcode
                })
            }
            _ => Err(TraceAbortReason::UnsupportedOpcode),
        }
    }

    /// Records `Move a, b` at `pc`.
    fn record_move(
        &mut self,
        registers: &[Value],
        pc: usize,
        a: usize,
        b: usize,
    ) -> Result<(), TraceAbortReason> {
        self.check_register(a)?;
        // A number of run-time subtype moves as the same number.
        if let Some((value, _)) = self.number_pair(b) {
            self.write(a, Sym::Op(value));
            return Ok(());
        }
        // A string constant moves as the constant it is.
        if let Sym::Root(slot) = self.sym(b) {
            self.write(a, Sym::Root(slot));
            return Ok(());
        }
        // A side trace that runs through an outer loop copies a table into a
        // register that holds it already (the state of a `pairs` loop that
        // starts again). The copy changes nothing, and a write to the
        // register would keep the side out of a tree that reads the table's
        // array part through it, so the trace checks that the two are one
        // table and leaves the register as it is.
        if self.side.is_some()
            && self.frames.is_empty()
            && self.sym(a) == Sym::Carried
            && !self.written[a]
            && matches!((&registers[a], &registers[b]),
                (Value::Table(held), Value::Table(source)) if held == source)
        {
            let (held, held_ty) = self.read(registers, a)?;
            let (source, source_ty) = self.read(registers, b)?;
            if held_ty == TraceType::Table && source_ty == TraceType::Table {
                self.guard_same(pc, held, source);
                return Ok(());
            }
        }
        // A virtual table moves as the same table: two registers
        // name it.
        if let Some(id) = self.virtual_in(b) {
            if !self.may_hold_virtual(a) {
                return Err(TraceAbortReason::VirtualEscape);
            }
            self.write(a, Sym::Virtual(id));
            return Ok(());
        }
        let (source, ty) = self.read(registers, b)?;
        let value = self.emit(pc, RecKind::Move(source), Some(ty));
        self.write(a, value);
        Ok(())
    }

    /// `ForInCall`: the three copies into the call window at `a + 3`, then
    /// the call of the iterator there with two arguments and `b` results,
    /// recorded as the `Call` it stands for. The call sees the registers as
    /// the copies leave them.
    fn record_for_in_call(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        let base = usize::from(instruction.a());
        let window = base + 3;
        self.check_register(window + 2)?;
        // A `string.gmatch` step ignores both arguments, so the copies of the
        // state and the control value are not made: the control value would
        // otherwise be an entry input typed by the shape of the last match
        // (a string, a position, and nil on the loop's first pass).
        let ignored = self.ignores_loop_arguments(at, base, at.pc + 1);
        if ignored && !self.gmatch_controls.iter().any(|(own, _)| *own == base + 2) {
            self.gmatch_controls.push((base + 2, None));
        }
        // A `next` loop over a table with a hash part takes its table from
        // the state register and its key from the control lane, whatever the
        // types of the keys: only the iterator is copied.
        let boxed = self.boxes_loop(at, base);
        self.boxed_call = boxed;
        let ignored = ignored || boxed;
        // The copy of the iterator into the call window is the callee of the
        // step. A variable the body never reads gets no value in the end.
        let window_before = boxed.then(|| (self.regs[window], self.written[window]));
        for offset in 0..if ignored { 1 } else { 3 } {
            self.record_move(at.registers, at.pc, window + offset, base + offset)?;
        }
        let mut copied = at.registers.to_vec();
        for offset in 0..3 {
            copied[window + offset] = copied[base + offset].clone();
        }
        let call = Instruction::call(
            instruction.a() + 3,
            if ignored { 0 } else { 2 },
            instruction.b(),
        );
        let at = HookPoint {
            chunk: at.chunk,
            depth: at.depth,
            top_level: at.top_level,
            pc: at.pc,
            registers: &copied,
            open: at.open,
            coroutine: at.coroutine,
            hooks_active: at.hooks_active,
            source: at.source,
            surroundings: at.surroundings,
        };
        let flow = self.record_call(&at, call)?;
        if let Some(before) = window_before
            && self.unread.contains(&window)
        {
            (self.regs[window], self.written[window]) = before;
        }
        Ok(flow)
    }

    /// Whether the `ForInLoop` at `loop_pc` closes the loop this trace
    /// belongs to, in the root frame.
    fn closes_at_header(&self, at: &HookPoint<'_>, loop_pc: usize) -> bool {
        if !self.in_home() {
            return false;
        }
        let Some(instruction) = at.chunk.code.get(loop_pc) else {
            return false;
        };
        instruction.opcode() == Ok(OpCode::ForInLoop)
            && self.target(loop_pc, short_offset(*instruction)).ok() == Some(self.header)
    }

    /// Whether the generic `for` at `base`, whose `ForInCall` is being
    /// recorded, is a `next` loop whose control value the trace keeps in a
    /// lane of the run's bank (see [`TraceIr::boxed_control`]). A root
    /// recording decides when it meets the loop's call: a table with a hash
    /// part has keys of several types, and a typed control register would
    /// pin the type of the first. A table with an array part alone is
    /// traversed by the array operations of [`Self::record_array_next`]
    /// instead.
    fn boxes_loop(&mut self, at: &HookPoint<'_>, base: usize) -> bool {
        let control = base + 2;
        if !self.closes_at_header(at, at.pc + 1) {
            return false;
        }
        if self.boxed.is_some() {
            return self.boxed == Some(control);
        }
        let (Some(Value::BuiltinFunction(BuiltinFunction::Next)), Some(Value::Table(_))) =
            (at.registers.get(base), at.registers.get(base + 1))
        else {
            return false;
        };
        let Some(surroundings) = at.surroundings else {
            return false;
        };
        if self.side.is_some()
            || self.function.is_some()
            || self.regs[control] != Sym::Carried
            || self.written[control]
        {
            return false;
        }
        // A table with no hash part is read in place, whether or not other
        // loops enclose this one: every key is an integer of the array part
        // (see [`Self::record_array_next`]). A table with a hash part has
        // keys of several types, and the lane holds them, array part
        // included: one version of the loop traverses the whole table, where
        // a version for each part would leave the trace and enter the other
        // at every traversal.
        let steps_along_array = crate::vm::jit::trace::helpers::steps_along_array(
            surroundings.runtime,
            &at.registers[control - 1],
            &at.registers[control],
        );
        let Value::Table(handle) = at.registers[control - 1] else {
            return false;
        };
        if steps_along_array
            && !surroundings
                .runtime
                .table_has_hash_part(handle)
                .unwrap_or(true)
        {
            return false;
        }
        self.box_loop(control);
        true
    }

    /// Makes the `next` loop with control register `control` a boxed one:
    /// the register stands for the lane, and the loop variables the body
    /// never reads are not produced. An exit still gives them the values the
    /// interpreter holds (`debug.getlocal` shows them): the first is the key,
    /// which the control lane holds, and the second has a lane of its own.
    fn box_loop(&mut self, control: usize) {
        self.boxed = Some(control);
        self.entry[control] = Kind::String;
        let variables = self
            .latch
            .checked_sub(1)
            .and_then(|pc| self.chunk.code.get(pc))
            .map_or(0, |call| usize::from(call.b()).max(1));
        for register in control + 1..control + 1 + variables {
            if !self.body_reads(register)
                && !self.body_writes(register)
                && !self.unread.contains(&register)
            {
                self.unread.push(register);
            }
        }
        self.boxed_key = self.unread.contains(&(control + 1));
        self.boxed_value = variables > 1
            && self.unread.contains(&(control + 2))
            && control + 2 < self.register_count;
        if self.boxed_value {
            self.entry[control + 2] = Kind::String;
        }
    }

    /// Whether the body of the loop may write `register`, other than the
    /// step does.
    fn body_writes(&self, register: usize) -> bool {
        u8::try_from(register).map_or(true, |register| {
            self.chunk
                .writes_register_between(self.header, self.latch.saturating_sub(1), register)
        })
    }

    /// Whether the body of the loop reads `register` before it writes it:
    /// the value a step wrote to it in the last iteration is used. A loop
    /// variable the body never mentions is no such register, whatever its
    /// scope says.
    fn body_reads(&self, register: usize) -> bool {
        u8::try_from(register).map_or(true, |register| {
            self.chunk.reads_before_write(self.header, register)
        })
    }

    /// Whether the iterator of the generic `for` at `base` is a
    /// `string.gmatch` one, which takes no notice of its arguments. The loop
    /// does not track the control value then: it is no entry input typed by
    /// the shape of the last match, and an exit restores it from the step
    /// result the loop-back stored (see `finalize`).
    ///
    /// Only the loop the trace closes at does this: in an inlined callee or
    /// an inner loop the control slot is an ordinary register, so those keep
    /// the copies and the control write. `loop_pc` is the pc of the loop's
    /// `ForInLoop`.
    fn ignores_loop_arguments(&self, at: &HookPoint<'_>, base: usize, loop_pc: usize) -> bool {
        if !matches!(at.registers.get(base), Some(Value::GmatchIterator(_)))
            || !self.in_home()
            || self.side.is_some()
        {
            return false;
        }
        let Some(instruction) = at.chunk.code.get(loop_pc) else {
            return false;
        };
        instruction.opcode() == Ok(OpCode::ForInLoop)
            && self.target(loop_pc, short_offset(*instruction)).ok() == Some(self.header)
    }

    /// `ForInLoop`: when the first loop variable is not nil it becomes the
    /// control value and the loop goes back to its body. Like a branch on a
    /// non-boolean, the value's type decides and the entry contract holds
    /// it, so no guard is needed.
    fn for_in_loop(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let base = usize::from(instruction.a());
        let target = self.target(pc, short_offset(instruction))?;
        if self.boxed == Some(base + 2) && self.closes_at_header(at, pc) {
            // The step knows whether it gave an entry: the first variable
            // may be one the body never reads.
            return match self.boxed_found.take() {
                Some(true) => {
                    self.check_branch(pc, target)?;
                    Ok(target)
                }
                Some(false) => Ok(pc + 1),
                None => Err(TraceAbortReason::Divergence),
            };
        }
        self.check_register(base + 3)?;
        let (variable, ty) = self.read(at.registers, base + 3)?;
        if let Operand::Carried(register) = variable {
            self.condition_reads.push(register);
        }
        if ty == TraceType::Nil {
            return Ok(pc + 1);
        }
        if self.links_loop(pc, target) {
            // The side rejoins the loop side it descends from.
        } else if self.wants_loop_side_at(pc, target, LoopKind::Generic)
            && !matches!(at.registers.get(base), Some(Value::GmatchIterator(_)))
        {
            // The recording ends in front of the loop's next trip.
            self.enter_loop = Some((target, pc));
            self.enter_depth = self.frames.len();
        } else {
            let gmatch = matches!(at.registers.get(base), Some(Value::GmatchIterator(_)));
            match self.check_branch_of(pc, target, gmatch) {
                // The latch of a generic for loop in an inlined call (or a
                // root-frame loop found short) is followed like the jump of
                // a `while` loop: the step's result type is a guard on every
                // trip, so a run of another length leaves there.
                Err(TraceAbortReason::InnerLoop)
                    if !gmatch && self.may_unroll_while(pc, target) =>
                {
                    self.unrolled_loops += 1;
                    self.unrolled_whiles += 1;
                }
                Err(TraceAbortReason::InnerLoop) => {
                    self.met_long_loop = self.unroll_loops;
                    return Err(TraceAbortReason::InnerLoop);
                }
                other => other?,
            }
        }
        if target == self.header {
            self.note_gmatch_trips(at, base);
        }
        if self.ignores_loop_arguments(at, base, pc) {
            // The body may overwrite the variable, and the interpreter's
            // control slot keeps the step result: remember the operation.
            if let Operand::Op(index) = variable
                && let Some(control) = self
                    .gmatch_controls
                    .iter_mut()
                    .find(|(own, _)| *own == base + 2)
            {
                control.1 = Some(index);
            }
        } else {
            let value = self.emit(pc, RecKind::Move(variable), Some(ty));
            self.write(base + 2, value);
        }
        Ok(target)
    }

    /// Records the comparison of two values and returns the operation that
    /// holds its boolean result. `tables` are the live operands, which two
    /// tables need: only identical ones are followed.
    #[allow(clippy::too_many_lines)]
    fn record_comparison(
        &mut self,
        pc: usize,
        comparison: Comparison,
        (left, left_ty): (Operand, TraceType),
        (right, right_ty): (Operand, TraceType),
        tables: Option<(&Value, &Value)>,
    ) -> Result<u32, TraceAbortReason> {
        // An integer and a float are both numbers and compare exactly.
        let mut numbers = is_number(left_ty) && is_number(right_ty);
        let (mut left, mut right) = (left, right);
        let (mut left_ty, mut right_ty) = (left_ty, right_ty);
        if left_ty.is_wide() || right_ty.is_wide() {
            let convertible = |ty: TraceType| ty.is_wide() || ty == TraceType::Float;
            if convertible(left_ty) && convertible(right_ty) {
                // Both sides become one 64-bit type, as the
                // interpreter's `common_operands` does.
                let target = Self::wide_target(&[left_ty, right_ty]);
                left = self.to_wide(pc, left, left_ty, target);
                right = self.to_wide(pc, right, right_ty, target);
                (left_ty, right_ty) = (target, target);
                numbers = false;
            } else if comparison != Comparison::Equal {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        }
        // Two strings are equal when their bytes are, whatever kind
        // of string value holds them here.
        let strings = matches!(left_ty, TraceType::String | TraceType::StringRef)
            && matches!(right_ty, TraceType::String | TraceType::StringRef);
        if comparison == Comparison::Equal && left_ty != right_ty && !numbers && !strings {
            // Values of different types are never equal, and `__eq`
            // only runs for two tables or two userdata, so no
            // metamethod can intervene: the entry types decide the
            // answer and the comparison is a constant. The entry
            // contract must still hold those types, or a frame of
            // other types would run a trace that never looked.
            for operand in [left, right] {
                if let Operand::Carried(register) = operand {
                    self.condition_reads.push(register);
                }
            }
            let value = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Boolean(false)),
                Some(TraceType::Boolean),
            );
            return Ok(value);
        }
        if strings && comparison == Comparison::Equal {
            // Two strings are equal when their bytes are, which no lane or
            // address comparison tells: equal bytes may sit in two
            // allocations. One value is equal to itself; two root strings
            // fold by their bytes; any other pair is a helper call that
            // compares the bytes. `__eq` never runs for strings.
            if left == right {
                return Ok(self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Boolean(true)),
                    Some(TraceType::Boolean),
                ));
            }
            if let (Some(left_text), Some(right_text)) = (
                self.root_constant_bytes(left),
                self.root_constant_bytes(right),
            ) {
                let equal = left_text == right_text;
                return Ok(self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Boolean(equal)),
                    Some(TraceType::Boolean),
                ));
            }
            for operand in [left, right] {
                self.note_type_read(operand);
            }
            return Ok(self.emit_helper_call(
                pc,
                HelperId::STRING_EQUAL,
                &[(left, left_ty), (right, right_ty)],
            ));
        }
        if strings {
            // Strings order by their bytes, which native code does not
            // compare.
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if (left_ty != right_ty && !numbers)
            || (comparison != Comparison::Equal && !is_number(left_ty) && !left_ty.is_wide())
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let guarded = matches!(left_ty, TraceType::Table | TraceType::Userdata);
        if guarded {
            // Two different tables or userdata may still be equal through
            // `__eq`, which only the interpreter can run. Record the
            // identical case and guard it, so any other pair leaves
            // before the comparison. Closures have no `__eq`.
            let identical = matches!(
                tables,
                Some((Value::Table(x), Value::Table(y))) if x == y
            ) || matches!(
                tables,
                Some((Value::Userdata(x), Value::Userdata(y))) if x == y
            );
            if !identical {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        }
        let value = if left_ty == TraceType::Table {
            self.compare_tables(pc, left, right)
        } else {
            self.emit(
                pc,
                RecKind::Compare(comparison, left, right),
                Some(TraceType::Boolean),
            )
        };
        if guarded {
            let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            self.emit(
                pc,
                RecKind::GuardBranch {
                    condition: Operand::Op(value),
                    expected: true,
                    snapshot: before,
                },
                None,
            );
        }
        Ok(value)
    }

    /// Whether the tables `left` and `right` name are one table. A function
    /// trace's run gives a table a new lane at every read, so two lanes of
    /// one table differ and the answer comes from the handles the lanes
    /// hold. Every other trace gives a table one lane, which a compare tells.
    fn compare_tables(&mut self, pc: usize, left: Operand, right: Operand) -> u32 {
        if self.function.is_some() {
            return self.emit_helper_call(
                pc,
                HelperId::TABLE_SAME,
                &[(left, TraceType::Table), (right, TraceType::Table)],
            );
        }
        self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, left, right),
            Some(TraceType::Boolean),
        )
    }

    /// The virtual table the root frame's `register` names, if it names one.
    fn virtual_in(&self, register: usize) -> Option<u32> {
        match self.sym(register) {
            Sym::Virtual(id) if self.activation.is_none() => Some(id),
            _ => None,
        }
    }

    /// Whether the trace has written a cell that an `UpvalueField` slot reads
    /// through (`_ENV.math` after `_ENV = other`). The slot holds the value
    /// from entry, so it would go stale.
    fn field_cell_written(&self) -> bool {
        if self.field_cells.is_empty() {
            return false;
        }
        let cell_written = |cell: Option<CellHandle>| {
            let matches = |held: CellHandle| cell.is_none_or(|cell| cell == held);
            self.external_cells
                .iter()
                .enumerate()
                .any(|(position, held)| {
                    held.is_some_and(matches) && self.written[self.register_count + position]
                })
                || self
                    .root_captured
                    .iter()
                    .any(|(register, held)| matches(*held) && self.written[usize::from(*register)])
        };
        self.field_cells.iter().any(|cell| cell_written(*cell))
    }

    /// Whether the root frame's `register` may hold a virtual table: a
    /// captured local is a cell other closures read, so a table there is
    /// reachable from outside the trace.
    fn may_hold_virtual(&self, register: usize) -> bool {
        // An inlined callee's registers are no cell's: nothing records a
        // closure there.
        if !self.frames.is_empty() {
            return self.activation.is_none()
                && register < usize::from(self.running_chunk().register_count)
                && !(self.running_caller()
                    && self.caller_seed().is_some_and(|seed| {
                        u8::try_from(register).is_ok_and(|narrow| seed.captured.contains(&narrow))
                    }));
        }
        self.in_home()
            && register < self.register_count
            && !self
                .root_captured
                .iter()
                .any(|(captured, _)| usize::from(*captured) == register)
    }

    /// Records a `NewTable` as a virtual table: no table is made, a phantom
    /// charge stands for it, and the register names its contents.
    fn record_new_table(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let a = usize::from(instruction.a());
        self.check_register(a)?;
        // A function trace makes its tables for real, where the interpreter's
        // constructor would: an exit deep in a recursion has no virtual table
        // to build for each activation it unwinds.
        if self.function.is_some() {
            return self.record_made_table(at, instruction);
        }
        // A side recording does not know the root's captured locals. A later
        // activation of the root frame keeps no virtual table either. An
        // inlined callee's table may only leave as a caught error value.
        // The frame that called the root frame is the exception: the
        // recording knows which of its registers a closure captured.
        if self.activation.is_some() || (self.side.is_some() && !self.in_caller()) {
            return Err(TraceAbortReason::UnsupportedOpcode);
        }
        if !self.may_hold_virtual(a) {
            return Err(TraceAbortReason::VirtualEscape);
        }
        if self.virtuals.len() >= MAX_TRACE_VIRTUALS {
            return Err(limit("virtuals"));
        }
        // The storage the interpreter's constructor would give: a table with
        // the site's shared key list when it has one.
        let alloc = at
            .chunk
            .constructor_shape(pc)
            .map_or(VirtualAlloc::Plain, |shape| VirtualAlloc::Shaped {
                keys: shape
                    .keys()
                    .iter()
                    .map(|key| Arc::from(key.as_bytes()))
                    .collect(),
            });
        let id = u32::try_from(self.virtuals.len()).expect("virtual count fits u32");
        self.virtuals.push(VirtualState {
            alloc,
            array: Vec::new(),
            fields: Vec::new(),
            peak: 0,
            metatable: None,
            real: None,
        });
        self.emit(pc, RecKind::PhantomTable { virt: id }, None);
        self.write(a, Sym::Virtual(id));
        Ok(pc + 1)
    }

    /// Records a `NewTable` of a function trace as a call of
    /// [`HelperId::TABLE_NEW`]: the table is made where the constructor runs,
    /// with the storage the interpreter's constructor would give it, and the
    /// register holds its lane. The fields the constructor stores are calls
    /// of [`HelperId::TABLE_INIT`] ([`Self::record_made_store`]). The call
    /// leaves through the instruction's `Before` snapshot when it cannot
    /// make the table (the run has no lane or the collector is owed a step),
    /// where the interpreter runs the constructor. A table in a register the
    /// code captures in a closure is no lane's to hold, so a function that
    /// makes closures records none.
    fn record_made_table(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let a = usize::from(instruction.a());
        if !self.frames.is_empty()
            || self.activation.is_some()
            || !self.root_captured.is_empty()
            || self
                .chunk
                .code
                .iter()
                .any(|code| code.opcode() == Ok(OpCode::Closure))
        {
            return Err(TraceAbortReason::UnsupportedOpcode);
        }
        let keys: Vec<Arc<[u8]>> = match at.chunk.constructor_shape(pc) {
            Some(shape) if shape.keys().len() <= MAX_NEW_TABLE_KEYS => shape
                .keys()
                .iter()
                .map(|key| Arc::from(key.as_bytes()))
                .collect(),
            Some(_) => return Err(limit("new table keys")),
            None => Vec::new(),
        };
        let length = self.scalar(pc, TraceScalar::Integer(i64::from(instruction.b())));
        let mut args = vec![(Operand::Op(length), TraceType::Integer)];
        for key in &keys {
            let slot = self.root_slot(key)?;
            args.push(self.root_operand(pc, slot));
        }
        let table = self.emit_helper_call(pc, HelperId::TABLE_NEW, &args);
        self.write(a, Sym::Op(table));
        Ok(pc + 1)
    }

    /// Records a `SetField` into a table a function trace made
    /// ([`Self::record_made_table`]) as a call of [`HelperId::TABLE_INIT`].
    /// Nothing but the trace's lane names the table, so the store is seen by
    /// no row or binding. Any other store aborts the recording: a function
    /// trace stores into no table it did not make.
    fn record_made_store(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (table, ty) = self.read(at.registers, usize::from(instruction.a()))?;
        let made = ty == TraceType::Table
            && matches!(
                self.origin(table),
                Operand::Op(index) if matches!(
                    self.ops[index as usize].kind,
                    RecKind::CallHelper { helper: HelperId::TABLE_NEW, .. }
                )
            );
        if !made || instruction.opcode() != Ok(OpCode::SetField) {
            return Err(TraceAbortReason::SelfCall);
        }
        // The call names the table by the operation that made it.
        let table = self.origin(table);
        let TableKey::Field(text) = Self::field_key(at, instruction.b())? else {
            return Err(TraceAbortReason::Divergence);
        };
        let slot = self.root_slot(text.as_bytes())?;
        let key = self.root_operand(pc, slot);
        let (value, value_ty) = self.read(at.registers, usize::from(instruction.c()))?;
        if !HelperType::Value.accepts(value_ty) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        self.emit_helper_call(
            pc,
            HelperId::TABLE_INIT,
            &[(table, TraceType::Table), key, (value, value_ty)],
        );
        Ok(pc + 1)
    }

    /// The key a virtual table access names, which must be a constant of the
    /// trace: a string constant or a number the recorded operations load as a
    /// constant. A key the run computes could name another slot next time.
    /// Numbers are normalized as the interpreter's table does: a float with an
    /// integral value is that integer.
    fn virtual_key(
        &self,
        at: &HookPoint<'_>,
        register: usize,
    ) -> Result<VirtualKey, TraceAbortReason> {
        match self.table_key(at, register)? {
            TableKey::Field(text) => Ok(VirtualKey::String(Arc::from(text.as_bytes()))),
            TableKey::Hash(..) | TableKey::Object(..) => Err(TraceAbortReason::VirtualEscape),
            TableKey::Index(number) => {
                let origin = self.origin(number.operand);
                let Operand::Op(index) = origin else {
                    return Err(TraceAbortReason::VirtualEscape);
                };
                match self.ops[index as usize].kind {
                    RecKind::Constant(TraceScalar::Integer(key)) => Ok(VirtualKey::Integer(key)),
                    RecKind::Constant(TraceScalar::Float(key)) => Self::float_key(key),
                    _ => Err(TraceAbortReason::VirtualEscape),
                }
            }
        }
    }

    /// The key of a float. A NaN key makes the store an error and an infinite
    /// one is left to the interpreter.
    fn float_key(key: f64) -> Result<VirtualKey, TraceAbortReason> {
        if !key.is_finite() {
            return Err(TraceAbortReason::VirtualEscape);
        }
        // Integral and inside the 64-bit range: the same key as the integer.
        if key.fract() == 0.0
            && (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&key)
        {
            #[allow(clippy::cast_possible_truncation)]
            return Ok(VirtualKey::Integer(key as i64));
        }
        Ok(VirtualKey::Float(key))
    }

    /// The value register `register` stores into a virtual table: `None` for
    /// nil, which stores nothing. Only scalars and string constants go in.
    fn virtual_value(
        &mut self,
        registers: &[Value],
        register: usize,
    ) -> Result<Option<VirtualItem>, TraceAbortReason> {
        if let Sym::Root(slot) = self.sym(register) {
            return Ok(Some(VirtualItem::Root(slot)));
        }
        // A table built inside the table. It stays a recipe of its own, and
        // a table that holds the one it is stored in would be a cycle.
        if let Sym::Virtual(inner) = self.sym(register) {
            return Ok(Some(VirtualItem::Virtual(inner)));
        }
        let (operand, ty) = self.read(registers, register)?;
        match ty {
            TraceType::Nil => {
                // Dropping a nil still depends on the register being nil.
                if let Operand::Carried(carried) = operand {
                    self.condition_reads.push(carried);
                }
                Ok(None)
            }
            TraceType::Boolean
            | TraceType::Integer
            | TraceType::Float
            | TraceType::Int64
            | TraceType::UInt64 => Ok(Some(VirtualItem::Value(operand, ty))),
            _ => Err(TraceAbortReason::VirtualEscape),
        }
    }

    /// What virtual table `id` holds at `key`: `None` for nil.
    fn virtual_load(&self, id: u32, key: &VirtualKey) -> Option<VirtualItem> {
        let state = &self.virtuals[id as usize];
        if let VirtualKey::Integer(index) = key
            && let Ok(index) = usize::try_from(*index)
            && (1..=state.array.len()).contains(&index)
        {
            return Some(state.array[index - 1]);
        }
        state
            .fields
            .iter()
            .find(|(known, _)| known == key)
            .map(|(_, item)| *item)
    }

    /// Stores `item` (`None` for nil, which deletes) at `key` of virtual table
    /// `id`, as the interpreter's raw set would leave the table. A key right
    /// after the array extends it and pulls the fields that follow into it.
    /// Deleting inside the array would leave a hole, and where the interpreter
    /// keeps such a table's border is not something the trace can promise, so
    /// that rejects.
    fn virtual_store(
        &mut self,
        id: u32,
        key: VirtualKey,
        item: Option<VirtualItem>,
    ) -> Result<(), TraceAbortReason> {
        let state = &mut self.virtuals[id as usize];
        let length = state.array.len();
        let position = match &key {
            VirtualKey::Integer(index) => usize::try_from(*index).ok().filter(|index| *index >= 1),
            _ => None,
        };
        match (position, item) {
            (Some(index), Some(item)) if index <= length => state.array[index - 1] = item,
            (Some(index), None) if index == length => {
                state.array.pop();
            }
            (Some(index), None) if index < length => {
                return Err(TraceAbortReason::VirtualEscape);
            }
            (Some(index), Some(item)) if index == length + 1 => {
                state.array.push(item);
                // Fields that now follow the array belong to it.
                loop {
                    let next = i64::try_from(state.array.len() + 1).expect("length fits i64");
                    let Some(found) = state
                        .fields
                        .iter()
                        .position(|(key, _)| *key == VirtualKey::Integer(next))
                    else {
                        break;
                    };
                    let (_, item) = state.fields.remove(found);
                    state.array.push(item);
                }
            }
            (_, item) => match state.fields.iter().position(|(known, _)| *known == key) {
                Some(found) => match item {
                    Some(item) => state.fields[found].1 = item,
                    None => {
                        state.fields.remove(found);
                    }
                },
                None => {
                    if let Some(item) = item {
                        state.fields.push((key, item));
                    }
                }
            },
        }
        state.peak = state.peak.max(state.entries());
        if state.entries() > MAX_VIRTUAL_ENTRIES {
            return Err(limit("virtual entries"));
        }
        Ok(())
    }

    /// A `GetTable` or `GetField` of a virtual table.
    fn record_virtual_get(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        field: bool,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, b, c) = (
            usize::from(instruction.a()),
            usize::from(instruction.b()),
            usize::from(instruction.c()),
        );
        self.check_register(a)?;
        let id = self.virtual_in(b).ok_or(TraceAbortReason::Divergence)?;
        let key = if field {
            match Self::field_key(at, instruction.c())? {
                TableKey::Field(text) => VirtualKey::String(Arc::from(text.as_bytes())),
                _ => return Err(TraceAbortReason::Divergence),
            }
        } else {
            self.virtual_key(at, c)?
        };
        self.require_current_shadow(id)?;
        // A missing key reads as nil unless the metatable has an `__index`.
        let loaded = self.virtual_load(id, &key);
        if loaded.is_none() {
            self.rely_on_absence(id, 1)?;
        }
        match loaded {
            // A table inside the table: the register names it too.
            Some(VirtualItem::Virtual(inner)) => {
                if !self.may_hold_virtual(a) {
                    return Err(TraceAbortReason::VirtualEscape);
                }
                self.write(a, Sym::Virtual(inner));
            }
            None => {
                let value = self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Nil),
                    Some(TraceType::Nil),
                );
                self.write(a, value);
            }
            Some(VirtualItem::Root(slot)) => self.write(a, Sym::Root(slot)),
            Some(VirtualItem::Value(source, ty)) => {
                let value = self.emit(pc, RecKind::Move(source), Some(ty));
                self.write(a, value);
            }
        }
        Ok(pc + 1)
    }

    /// Notes that the recording leans on the absence of the metamethod `bit`
    /// (see [`VirtualMeta::relied`]) from the metatable of virtual table
    /// `id`, if it has one. A metatable that has the metamethod sends the
    /// table to the interpreter.
    fn rely_on_absence(&mut self, id: u32, bit: u8) -> Result<(), TraceAbortReason> {
        let Some(meta) = self.virtuals[id as usize].metatable.as_mut() else {
            return Ok(());
        };
        let absent = match bit {
            1 => meta.no_index,
            2 => meta.no_newindex,
            _ => meta.no_len,
        };
        if !absent {
            return Err(TraceAbortReason::VirtualEscape);
        }
        meta.relied |= bit;
        Ok(())
    }

    /// A `SetTable` or `SetField` into a virtual table.
    fn record_virtual_set(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        field: bool,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, b, c) = (
            usize::from(instruction.a()),
            usize::from(instruction.b()),
            usize::from(instruction.c()),
        );
        let id = self.virtual_in(a).ok_or(TraceAbortReason::Divergence)?;
        let key = if field {
            match Self::field_key(at, instruction.b())? {
                TableKey::Field(text) => VirtualKey::String(Arc::from(text.as_bytes())),
                _ => return Err(TraceAbortReason::Divergence),
            }
        } else {
            self.virtual_key(at, b)?
        };
        // The table already escaped (a store closed a cycle, say), so the
        // store goes to the real table like any other. The model follows it
        // while it still describes the table, so reads go on resolving.
        if self.virtuals[id as usize].real.is_some() {
            return self.record_escaped_set(at, instruction, id, &key);
        }
        let item = self.virtual_value(at.registers, c)?;
        // A key the table lacks runs `__newindex` if the metatable has one.
        if self.virtual_load(id, &key).is_none() {
            self.rely_on_absence(id, 2)?;
        }
        // A recipe cannot hold itself, so a store that closes a cycle makes
        // the tables real and fills the field of the real table.
        if let Some(VirtualItem::Virtual(inner)) = item
            && self.virtual_holds(inner, id)
        {
            self.materialize(pc, inner, Some((id, &key)))?;
            let (Some(holder), Some(held)) = (
                self.virtuals[id as usize].real,
                self.virtuals[inner as usize].real,
            ) else {
                return Err(TraceAbortReason::VirtualEscape);
            };
            let key_operand = self.virtual_key_operand(pc, &key)?;
            self.emit_helper_call(
                pc,
                HelperId::TABLE_INIT,
                &[
                    (Operand::Op(holder.table), TraceType::Table),
                    key_operand,
                    (Operand::Op(held.table), TraceType::Table),
                ],
            );
            // The model follows the real table, so reads of the field go on
            // resolving, and the cycle is no recipe: both tables are real.
            self.virtual_store(id, key, item)?;
            return Ok(pc + 1);
        }
        self.virtual_store(id, key, item)?;
        Ok(pc + 1)
    }

    /// A store into virtual table `id` after it became a real table, as a
    /// call of [`HelperId::TABLE_INIT`] like the stores that filled it. That
    /// call has no effect on the heap the optimizer and the shadow check can
    /// see, so it is only sound while nothing but such calls names the table
    /// or a table holding it ([`Self::is_private`]): no load can reach the
    /// table, and the model is the only reader. A table stored anywhere else
    /// (an array, a global, a cell) rejects the recording. A virtual value
    /// stored in it is made real first. The model takes the store, and has to,
    /// since reads of the table go through it.
    fn record_escaped_set(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        id: u32,
        key: &VirtualKey,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let current = self.shadow_is_current(id);
        // A key the table lacks runs `__newindex` if the metatable has one,
        // which only a current model can say.
        if self.virtuals[id as usize].metatable.is_some() {
            if !current {
                return Err(TraceAbortReason::VirtualEscape);
            }
            if self.virtual_load(id, key).is_none() {
                self.rely_on_absence(id, 2)?;
            }
        }
        let register = usize::from(instruction.c());
        if let Sym::Virtual(inner) = self.sym(register) {
            self.materialize(pc, inner, None)?;
        }
        let item = self.virtual_value(at.registers, register);
        let Some(escaped) = self.virtuals[id as usize].real else {
            return Err(TraceAbortReason::VirtualEscape);
        };
        if !current || item.is_err() || !self.is_private(escaped.table) {
            return Err(TraceAbortReason::VirtualEscape);
        }
        let key_operand = self.virtual_key_operand(pc, key)?;
        let (value, ty) = self.read(at.registers, register)?;
        if !HelperType::Value.accepts(ty) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let item = item?;
        self.virtual_store(id, key.clone(), item)?;
        self.emit_helper_call(
            pc,
            HelperId::TABLE_INIT,
            &[
                (Operand::Op(escaped.table), TraceType::Table),
                key_operand,
                (value, ty),
            ],
        );
        self.virtuals[id as usize].real = Some(Escaped {
            after: self.ops.len(),
            ..escaped
        });
        Ok(pc + 1)
    }

    /// Whether nothing but calls of [`HelperId::TABLE_INIT`] names the table
    /// made by operation `made`, or a table that holds it at any depth.
    ///
    /// One walk over the operations collects the fills (which table each
    /// value went into) and every origin another operation names; the tables
    /// holding `made` are then the closure of the fills from it.
    fn is_private(&self, made: u32) -> bool {
        let mut holders: HashMap<Operand, Vec<Operand>> = HashMap::new();
        let mut mentioned = HashSet::new();
        for op in &self.ops {
            match &op.kind {
                RecKind::CallHelper {
                    helper: HelperId::TABLE_INIT,
                    args,
                    ..
                } => holders
                    .entry(self.origin(args[2]))
                    .or_default()
                    .push(self.origin(args[0])),
                RecKind::Move(_) => {}
                kind => kind.operands(|operand| {
                    mentioned.insert(self.origin(operand));
                }),
            }
        }
        let start = self.origin(Operand::Op(made));
        let mut named = HashSet::from([start]);
        let mut pending = vec![start];
        while let Some(table) = pending.pop() {
            if mentioned.contains(&table) {
                return false;
            }
            for &holder in holders.get(&table).into_iter().flatten() {
                if named.insert(holder) {
                    pending.push(holder);
                }
            }
        }
        true
    }

    /// Whether virtual table `outer` is `inner` or holds it, at any depth:
    /// storing `outer` in `inner` would make a cycle.
    fn virtual_holds(&self, inner: u32, outer: u32) -> bool {
        let mut pending = vec![inner];
        let mut seen = Vec::new();
        while let Some(id) = pending.pop() {
            if id == outer {
                return true;
            }
            if seen.contains(&id) {
                continue;
            }
            seen.push(id);
            pending.extend(
                self.virtuals[id as usize]
                    .items()
                    .filter_map(|item| match item {
                        VirtualItem::Virtual(held) => Some(*held),
                        _ => None,
                    }),
            );
        }
        false
    }

    /// Whether the contents recorded for virtual table `id` still describe
    /// it. They do while it has no real table. Once it has (it escaped), a
    /// write to a table the trace did not know when it made this one cannot
    /// have changed it, and any write to a table that could be this one can.
    /// The recorder tells the first kind from the second by where the table
    /// came from: an entry input or an operation before this table's own
    /// existed (it is another table), or one after (it might be this one,
    /// read back out of what it was stored in).
    fn shadow_is_current(&self, id: u32) -> bool {
        let Some(escaped) = self.virtuals[id as usize].real else {
            return true;
        };
        let born = escaped.table;
        // A cell holds the escaped table only if the trace stored it there
        // after the table was made: nothing else writes a cell while it runs.
        let stored_in_cell = self.ops[born as usize..].iter().any(|op| {
            matches!(
                &op.kind,
                RecKind::CallHelper { helper, args, .. }
                    if *helper == HelperId::CELL_SET
                        && args.get(2).is_some_and(|value| self.operand_type(*value) == TraceType::Table)
            )
        });
        let could_be_it = |table: Operand| match self.origin(table) {
            Operand::Carried(_) => false,
            Operand::Op(index) => {
                index >= born
                    && (stored_in_cell
                        || !matches!(
                            self.ops[index as usize].kind,
                            RecKind::CallHelper {
                                helper: HelperId::CELL_GET_TABLE,
                                ..
                            }
                        ))
            }
        };
        self.ops[escaped.after..].iter().all(|op| match &op.kind {
            RecKind::ArrayStore { table, .. }
            | RecKind::ArrayOverwrite { table, .. }
            | RecKind::FieldStore { table, .. }
            | RecKind::TableFieldStore { table, .. }
            | RecKind::LaneArrayStore { table, .. } => !could_be_it(*table),
            RecKind::SelfCall { .. } => false,
            // The table a helper writes is its first argument. A table among
            // the others is a value it stores or a metatable it sets.
            RecKind::CallHelper { helper, args, .. } => {
                !helper.effects().contains(HelperEffects::WRITES_HEAP)
                    || !args.first().is_some_and(|table| {
                        self.operand_type(*table) == TraceType::Table && could_be_it(*table)
                    })
            }
            RecKind::Constant(_)
            | RecKind::FrameInput { .. }
            | RecKind::Move(_)
            | RecKind::Negate(_)
            | RecKind::IntToFloat(_)
            | RecKind::Numeric(..)
            | RecKind::ToWide(..)
            | RecKind::NativeUnary(..)
            | RecKind::NativeBinary(..)
            | RecKind::Arithmetic(..)
            | RecKind::Compare(..)
            | RecKind::Not(_)
            | RecKind::NumberTag(_)
            | RecKind::BoolAnd(..)
            | RecKind::Select(..)
            | RecKind::NumberModulo(..)
            | RecKind::GuardBranch { .. }
            | RecKind::ArrayLoad { .. }
            | RecKind::LaneArrayLoad { .. }
            | RecKind::NumberLoad { .. }
            | RecKind::ArrayNext { .. }
            | RecKind::HashLoad { .. }
            | RecKind::FieldLoad { .. }
            | RecKind::TableLength { .. }
            | RecKind::StringLength { .. }
            | RecKind::StringByte { .. }
            | RecKind::TableFieldLoad { .. }
            | RecKind::MetaGuard { .. }
            | RecKind::PhantomTable { .. }
            | RecKind::HostCall { .. }
            | RecKind::HelperResult { .. } => true,
        })
    }

    /// Rejects the recording when the contents of escaped virtual table `id`
    /// may be stale ([`Self::shadow_is_current`]).
    fn require_current_shadow(&self, id: u32) -> Result<(), TraceAbortReason> {
        if self.shadow_is_current(id) {
            Ok(())
        } else {
            Err(TraceAbortReason::VirtualEscape)
        }
    }

    /// The operand a helper call names the virtual key `key` by.
    fn virtual_key_operand(
        &mut self,
        pc: usize,
        key: &VirtualKey,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        Ok(match key {
            VirtualKey::Integer(index) => (
                Operand::Op(self.scalar(pc, TraceScalar::Integer(*index))),
                TraceType::Integer,
            ),
            VirtualKey::Float(number) => (
                Operand::Op(self.scalar(pc, TraceScalar::Float(*number))),
                TraceType::Float,
            ),
            VirtualKey::String(text) => {
                let slot = self.root_slot(text)?;
                self.root_operand(pc, slot)
            }
        })
    }

    /// Makes real the virtual table `id` and every table it holds that has
    /// none yet, at the instruction at `pc`. Each table is made by a call of
    /// [`HelperId::TABLE_NEW`], all of them before any is filled, and then
    /// filled in the order an exit would store the recipe, by calls of
    /// [`HelperId::TABLE_INIT`]. The calls leave through the instruction's
    /// `Before` snapshot, where each table is still a recipe, so a miss
    /// abandons what was made and the interpreter builds the tables itself.
    /// The recorded contents stay as a model of the real tables.
    ///
    /// A table of a plain constructor (`{}`) that holds only string fields
    /// is made with the shape of those fields, and of the field `closing`
    /// names when the store that closes a cycle is about to fill it: a
    /// shape's keys sort as the keys of an ordered table do, so the table
    /// reads and iterates as the one the interpreter fills key by key, and
    /// its fills are stores into the slots the shape names
    /// ([`Self::plain_fill_keys`]).
    fn materialize(
        &mut self,
        pc: usize,
        id: u32,
        closing: Option<(u32, &VirtualKey)>,
    ) -> Result<(), TraceAbortReason> {
        if self.virtuals[id as usize].real.is_some() {
            return Ok(());
        }
        let mut nest = vec![id];
        let mut position = 0;
        while position < nest.len() {
            for item in self.virtuals[nest[position] as usize].items() {
                if let VirtualItem::Virtual(inner) = item
                    && self.virtuals[*inner as usize].real.is_none()
                    && !nest.contains(inner)
                {
                    nest.push(*inner);
                }
            }
            position += 1;
        }
        let mut calls = nest.len();
        for &table in &nest {
            let state = &self.virtuals[table as usize];
            // The exit sets a metatable after the contents; the helpers do
            // not.
            if state.metatable.is_some() {
                return Err(TraceAbortReason::VirtualEscape);
            }
            calls += state.entries();
            for item in state.items() {
                if let VirtualItem::Value(_, ty) = item
                    && !HelperType::Value.accepts(*ty)
                {
                    return Err(TraceAbortReason::VirtualEscape);
                }
            }
        }
        if calls > MAX_MATERIALIZE_CALLS {
            return Err(limit("materialize calls"));
        }
        let mut made: Vec<(u32, u32)> = Vec::with_capacity(nest.len());
        for &table in &nest {
            let state = &self.virtuals[table as usize];
            let array_len = state.array.len();
            let keys: Vec<Arc<[u8]>> = match &state.alloc {
                VirtualAlloc::Shaped { keys } if keys.len() <= MAX_NEW_TABLE_KEYS => keys.clone(),
                VirtualAlloc::Shaped { .. } => Vec::new(),
                VirtualAlloc::Plain => Self::plain_fill_keys(
                    state,
                    closing
                        .filter(|(holder, _)| *holder == table)
                        .map(|(_, key)| key),
                ),
            };
            let length = self.scalar(
                pc,
                TraceScalar::Integer(i64::try_from(array_len).expect("array length fits i64")),
            );
            let mut args = vec![(Operand::Op(length), TraceType::Integer)];
            for key in &keys {
                let slot = self.root_slot(key)?;
                args.push(self.root_operand(pc, slot));
            }
            let table_op = self.emit_helper_call(pc, HelperId::TABLE_NEW, &args);
            made.push((table, table_op));
        }
        let lane_of = |this: &Self, table: u32| {
            made.iter()
                .find(|(known, _)| *known == table)
                .map(|(_, op)| *op)
                .or_else(|| this.virtuals[table as usize].real.map(|real| real.table))
                .expect("a table inside a table has a table")
        };
        for &table in &nest {
            let holder = Operand::Op(lane_of(self, table));
            let state = self.virtuals[table as usize].clone();
            let mut stores: Vec<(VirtualKey, VirtualItem)> = Vec::new();
            for (index, item) in state.array.iter().enumerate() {
                let key = i64::try_from(index + 1).expect("array length fits i64");
                stores.push((VirtualKey::Integer(key), *item));
            }
            stores.extend(state.fields.iter().cloned());
            for (key, item) in stores {
                let key = self.virtual_key_operand(pc, &key)?;
                let value = match item {
                    VirtualItem::Value(operand, ty) => (operand, ty),
                    VirtualItem::Root(slot) => self.root_operand(pc, slot),
                    VirtualItem::Virtual(inner) => {
                        (Operand::Op(lane_of(self, inner)), TraceType::Table)
                    }
                };
                self.emit_helper_call(
                    pc,
                    HelperId::TABLE_INIT,
                    &[(holder, TraceType::Table), key, value],
                );
            }
        }
        let after = self.ops.len();
        for (table, op) in made {
            self.virtuals[table as usize].real = Some(Escaped { table: op, after });
        }
        Ok(())
    }

    /// The keys of the shape a plain virtual table is made with: the string
    /// keys it holds, and `closing` when it is a string. None for a table
    /// with an array part (made with its slots instead), with a key that is
    /// no string (which a shape cannot hold, so the first such store would
    /// turn the table into an ordered one), or with more keys than
    /// [`HelperId::TABLE_NEW`] names. Every key named is filled right after
    /// the call, so no slot of the shape stays empty unless a fill misses,
    /// and a miss abandons the table.
    fn plain_fill_keys(state: &VirtualState, closing: Option<&VirtualKey>) -> Vec<Arc<[u8]>> {
        if !state.array.is_empty() {
            return Vec::new();
        }
        let mut keys: Vec<Arc<[u8]>> = Vec::new();
        for key in state.fields.iter().map(|(key, _)| key).chain(closing) {
            let VirtualKey::String(name) = key else {
                return Vec::new();
            };
            if !keys.contains(name) {
                keys.push(name.clone());
            }
        }
        if keys.len() > MAX_NEW_TABLE_KEYS {
            return Vec::new();
        }
        keys
    }

    /// Makes real every virtual table that a register the instruction reads
    /// names, unless the instruction is one the recorder resolves on a
    /// virtual table (a move, a read or write of a constant key, `#`). The
    /// table is about to be used as a value: stored, compared, tested.
    fn escape_virtual_reads(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        opcode: OpCode,
    ) -> Result<(), TraceAbortReason> {
        if self.virtuals.is_empty() || self.activation.is_some() {
            return Ok(());
        }
        // A closure captures its registers as cells, which no recording
        // models for a table made here, and a halt ends the run.
        if matches!(opcode, OpCode::Closure | OpCode::Halt) {
            return Ok(());
        }
        let is_call = matches!(
            opcode,
            OpCode::Call
                | OpCode::CallOne
                | OpCode::CallVararg
                | OpCode::CallOpen
                | OpCode::TailCall
                | OpCode::TailCallVararg
                | OpCode::TailCallOpen
        );
        // A call of a native function that takes a virtual table its own way
        // (`setmetatable`, an error value) keeps it virtual. Any other callee
        // gets the real table: an inlined Lua function reads it by lane.
        if is_call
            && let Value::NativeFunction(function) =
                at.registers[usize::from(if opcode == OpCode::CallOne {
                    instruction.b()
                } else {
                    instruction.a()
                })]
            && matches!(
                Self::native_kind(at, function),
                Some(
                    DirectNativeKind::SetMetatable
                        | DirectNativeKind::Error
                        | DirectNativeKind::Assert
                )
            )
        {
            return Ok(());
        }
        let reads = at.chunk.instruction_reads(at.pc);
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        let count = usize::from(self.running_chunk().register_count);
        for register in 0..count {
            let Sym::Virtual(id) = self.sym(register) else {
                continue;
            };
            let Ok(narrow) = u8::try_from(register) else {
                continue;
            };
            if !reads.contains(narrow) {
                continue;
            }
            // What the recorder resolves with the table still virtual.
            let resolved = match opcode {
                OpCode::Move | OpCode::GetTable | OpCode::GetField | OpCode::Length => {
                    register == b
                }
                OpCode::SetTable | OpCode::SetField => {
                    register == a
                        || (self.virtual_in(a).is_some()
                            && register == usize::from(instruction.c()))
                }
                _ => false,
            };
            if !resolved {
                self.materialize(at.pc, id, None)?;
            }
        }
        Ok(())
    }

    /// A `#` of a virtual table: the length of its array, when that is the
    /// only border it can have.
    fn record_virtual_length(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        self.check_register(a)?;
        let id = self.virtual_in(b).ok_or(TraceAbortReason::Divergence)?;
        // Lua 5.1 and `LuaJIT` never ask `__len` for `#`.
        if self.profile_has_env_name {
            self.rely_on_absence(id, 4)?;
        }
        self.require_current_shadow(id)?;
        let state = &self.virtuals[id as usize];
        // Another integer key may start a border of its own, and then the
        // interpreter picks one the trace cannot name.
        if state
            .fields
            .iter()
            .any(|(key, _)| matches!(key, VirtualKey::Integer(_)))
        {
            return Err(TraceAbortReason::VirtualEscape);
        }
        let length = state.array.len();
        let (scalar, ty) = if self.integers {
            (
                TraceScalar::Integer(i64::try_from(length).expect("length fits i64")),
                TraceType::Integer,
            )
        } else {
            #[allow(clippy::cast_precision_loss)]
            (TraceScalar::Float(length as f64), TraceType::Float)
        };
        let value = self.emit(pc, RecKind::Constant(scalar), Some(ty));
        self.write(a, value);
        Ok(pc + 1)
    }

    /// The table in `register` and the window register that holds it at
    /// entry. A table is only known by identity here, so a table access needs
    /// the table to be the same one in every iteration: it must come, through
    /// moves, from a register the iteration never writes. [`Self::finalize`]
    /// checks the "never writes" part once the iteration is complete.
    fn table_receiver<'a>(
        &self,
        at: &HookPoint<'a>,
        register: usize,
    ) -> Result<(Receiver, TableHandle, &'a Runtime), TraceAbortReason> {
        let (operand, ty) = self.read(at.registers, register)?;
        if ty != TraceType::Table {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let Value::Table(handle) = at.registers[register] else {
            return Err(TraceAbortReason::Divergence);
        };
        self.receiver_of(at, operand, handle)
    }

    /// The table in upvalue `upvalue` of the running frame, as
    /// [`Self::table_receiver`] finds the table of a register that a
    /// `GetUpvalue` filled.
    fn tab_up_receiver<'a>(
        &mut self,
        at: &HookPoint<'a>,
        upvalue: u16,
    ) -> Result<(Receiver, TableHandle, &'a Runtime), TraceAbortReason> {
        let slot = self.upvalue_slot(upvalue);
        let external = self.external(at, slot)?;
        let (operand, ty) = self.read_external(at, external)?;
        if ty != TraceType::Table {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let Value::Table(handle) = self.external_value(at, slot)? else {
            return Err(TraceAbortReason::Divergence);
        };
        self.receiver_of(at, operand, handle)
    }

    /// The receiver of the table `handle`, which `operand` holds.
    fn receiver_of<'a>(
        &self,
        at: &HookPoint<'a>,
        mut operand: Operand,
        handle: TableHandle,
    ) -> Result<(Receiver, TableHandle, &'a Runtime), TraceAbortReason> {
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        // Follow moves (an argument, an upvalue or a global read) down to the
        // register the iteration found the table in, or to the array read
        // that produced it.
        let receiver = loop {
            match operand {
                // A table the loop itself replaces (`if t == a then t = b
                // else t = a end`) is another table at each iteration, so no
                // storage can be bound for it at entry: its accesses look the
                // table up through its lane.
                Operand::Carried(window) if self.replaced_in_loop(window) => {
                    break Receiver::Loaded(operand);
                }
                Operand::Carried(window) if self.changes_around(window) => {
                    break Receiver::Around(window);
                }
                Operand::Carried(window)
                    if self.lane_parameter(window) || self.seeded_lane(window) =>
                {
                    break Receiver::Loaded(operand);
                }
                Operand::Carried(window) => break Receiver::Bound(window),
                Operand::Op(index) => match self.ops[index as usize].kind {
                    RecKind::Move(source) => operand = source,
                    RecKind::ArrayLoad {
                        expected: TraceType::Table,
                        ..
                    }
                    | RecKind::LaneArrayLoad {
                        expected: TraceType::Table,
                        ..
                    }
                    | RecKind::CallHelper {
                        helper:
                            HelperId::TABLE_REMOVE_TABLE
                            | HelperId::INDEX_HANDLER
                            | HelperId::NEWINDEX_HANDLER
                            | HelperId::CELL_GET_TABLE
                            | HelperId::TABLE_GET_TABLE,
                        ..
                    } => break Receiver::Loaded(operand),
                    // A table a host function call gave: the helper put it in
                    // a lane of the bank and filled its row.
                    RecKind::CallHelper { helper, .. } if helper.is_vm_call() => {
                        break Receiver::Loaded(operand);
                    }
                    RecKind::HelperResult { call, .. }
                        if matches!(
                            self.ops[call as usize].kind,
                            RecKind::CallHelper { helper, .. } if helper.is_vm_call()
                        ) =>
                    {
                        break Receiver::Loaded(operand);
                    }
                    // A table read out of a field of another table, or one a
                    // register of a seeded frame holds (its parent's lane):
                    // no entry binding can stand for either.
                    RecKind::TableFieldLoad { .. } | RecKind::FrameInput { .. }
                        if self.ops[index as usize].ty == Some(TraceType::Table) =>
                    {
                        break Receiver::Loaded(operand);
                    }
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                },
            }
        };
        Ok((receiver, handle, runtime))
    }

    /// Whether `window` is a register a side starts with that holds a table
    /// its parent met at run time (see [`SideSeed::lanes`]).
    fn seeded_lane(&self, window: usize) -> bool {
        self.function.is_none()
            && self
                .side
                .as_ref()
                .is_some_and(|seed| seed.lanes.contains(&window))
    }

    /// Whether `window` is a table parameter of a function trace that reads
    /// its fields through the table's lane (see [`FunctionSeed::lanes`]).
    fn lane_parameter(&self, window: usize) -> bool {
        self.function
            .as_ref()
            .is_some_and(|function| function.lanes && window < function.parameters.len())
    }

    /// Whether the code of this loop may write the guest register `window`,
    /// which a table held there then differs between iterations. Only a
    /// rowless recording (see [`Site::rowless`](super::policy::Site)) takes
    /// that to be a table met by lane: the first recording binds the table at
    /// entry, which holds as long as the iteration it records does not
    /// replace it, and a recording that closes with the table replaced is
    /// refused to try again.
    fn replaced_in_loop(&self, window: usize) -> bool {
        self.rowless
            && self.function.is_none()
            && window < self.register_count
            && nest::written_within(
                &self.chunk,
                LoopSpan {
                    header: self.header,
                    latch: self.latch,
                },
                window,
            )
    }

    /// Whether the loops around this one write the guest register `window`
    /// outside it (see [`nest::written_around`]).
    fn changes_around(&self, window: usize) -> bool {
        // A cell or global is no guest register. A side that writes the cell
        // of a bound table is refused by `plan_tree`.
        window < self.register_count
            && self
                .written_around
                .get_or_init(|| {
                    nest::written_around(
                        &self.chunk,
                        LoopSpan {
                            header: self.header,
                            latch: self.latch,
                        },
                    )
                })
                .get(window)
                .copied()
                .unwrap_or(false)
    }

    /// How a field operation reaches the table of `receiver`.
    ///
    /// A table the loops around this one rewrite is another table at every
    /// outer iteration, so a tree that runs through those loops cannot bind its
    /// fields at entry: they are found through the table's lane. That only
    /// pays when such a tree can exist. Elsewhere the entry binds them, which
    /// is cheaper per access.
    // `Bound` and `Around` are distinct receiver cases that happen to share
    // the bound answer outside a coverable tree.
    #[allow(clippy::match_same_arms)]
    fn field_access(&self, receiver: Receiver) -> FieldAccess {
        match receiver {
            Receiver::Bound(window) | Receiver::Around(window) if self.lane_fields => {
                FieldAccess::Lane(Operand::Carried(window))
            }
            Receiver::Bound(window) => FieldAccess::Bound(window),
            Receiver::Around(window) if self.coverable() => {
                FieldAccess::Lane(Operand::Carried(window))
            }
            Receiver::Around(window) => FieldAccess::Bound(window),
            Receiver::Loaded(table) => FieldAccess::Lane(table),
        }
    }

    /// [`Self::field_access`] for a field the entry checks and the trace reads
    /// as a constant (a function or a string), which lanes do not hold.
    // The arms keep the order of `field_access`, whose bound cases differ.
    #[allow(clippy::match_same_arms)]
    fn field_access_in_place(&self, receiver: Receiver) -> FieldAccess {
        match receiver {
            Receiver::Bound(window) => FieldAccess::Bound(window),
            Receiver::Around(window) if self.coverable() => {
                FieldAccess::Lane(Operand::Carried(window))
            }
            Receiver::Around(window) => FieldAccess::Bound(window),
            Receiver::Loaded(table) => FieldAccess::Lane(table),
        }
    }

    /// The window register of a table whose storage the entry binds for an
    /// array access or a length: tables read out of an array have none.
    fn bound_window(receiver: Receiver) -> Result<usize, TraceAbortReason> {
        match receiver {
            Receiver::Bound(window) | Receiver::Around(window) => Ok(window),
            Receiver::Loaded(_) => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// Whether the loops around this one could run in a trace tree rooted at
    /// it (see [`nest::can_cover`]).
    fn coverable(&self) -> bool {
        *self.coverable.get_or_init(|| {
            nest::can_cover(
                &self.chunk,
                LoopSpan {
                    header: self.header,
                    latch: self.latch,
                },
            )
        })
    }

    /// The index of the field `text` of type `ty` in the trace's field list,
    /// adding it on first use.
    fn field(
        &mut self,
        text: &crate::string::LuaString,
        ty: TraceType,
    ) -> Result<u32, TraceAbortReason> {
        let position = match self
            .fields
            .iter()
            .position(|known| known.ty == ty && *known.key == *text.as_bytes())
        {
            Some(position) => position,
            None if self.fields.len() >= MAX_TRACE_FIELDS => {
                return Err(limit("fields"));
            }
            None => {
                self.fields.push(TableField {
                    key: text.as_bytes().into(),
                    ty,
                });
                self.fields.len() - 1
            }
        };
        Ok(u32::try_from(position).expect("fields fit u32"))
    }

    /// What a table access names: a constant string field, or a number.
    fn table_key(&self, at: &HookPoint<'_>, register: usize) -> Result<TableKey, TraceAbortReason> {
        if let Sym::Root(_) = self.sym(register) {
            // A string constant the interpreter loaded: its bytes are the
            // key, whichever chunk's constant it was.
            let Value::String(text) = &at.registers[register] else {
                return Err(TraceAbortReason::Divergence);
            };
            return Ok(TableKey::Field(text.clone()));
        }
        if let Value::String(text) = &at.registers[register] {
            let (operand, _) = self.read(at.registers, register)?;
            // A string constant that moved through a call is a field name.
            if self.root_constant_bytes(operand).is_some() {
                return Ok(TableKey::Field(text.clone()));
            }
            return Ok(TableKey::Hash(operand, text.clone()));
        }
        if let live @ (Value::Table(_) | Value::Closure(_)) = &at.registers[register] {
            let (operand, _) = self.read(at.registers, register)?;
            return Ok(TableKey::Object(operand, live.clone()));
        }
        let number = self.read_number(at.registers, register)?;
        Ok(TableKey::Index(number))
    }

    /// The index of the binding that resolves `kind` for the table in window
    /// register `register`, adding it on first use.
    fn binding(&mut self, register: usize, kind: BindingKind) -> u32 {
        let wanted = TableBinding {
            register: register_u16(register),
            kind,
        };
        // One array binding per table register: the first access names the
        // hint.
        let same = |known: &TableBinding| {
            known.register == wanted.register
                && match (&known.kind, &wanted.kind) {
                    (BindingKind::Array { .. }, BindingKind::Array { .. })
                    | (BindingKind::Header { .. }, BindingKind::Header { .. }) => true,
                    (known, wanted) => known == wanted,
                }
        };
        let lenient = matches!(wanted.kind, BindingKind::Header { strict: false });
        let position = self.bindings.iter().position(same).unwrap_or_else(|| {
            self.bindings.push(wanted);
            self.bindings.len() - 1
        });
        // A table is strict only if every set of it is: one set that needs
        // the VM makes the header lenient (a binding of the tree stays as
        // the tree has it).
        if lenient && position >= self.seeded_bindings {
            self.bindings[position].kind = BindingKind::Header { strict: false };
        }
        u32::try_from(position).expect("bindings fit u32")
    }

    /// The type slot 1 of the table holds, as the hint of an array binding.
    fn first_slot_type(&self, runtime: &Runtime, handle: TableHandle) -> Option<TraceType> {
        runtime
            .table_raw_array_range(handle, 1, 1)
            .ok()
            .flatten()
            .and_then(<[Value]>::first)
            .and_then(|value| self.slot_type(value))
    }

    /// The array slot the number `key` names in a table whose array part has
    /// `len` slots, as a zero-based index. `None` for a key outside the
    /// array: a fractional, huge, non-positive or hash-part key.
    fn array_slot(key: Live, len: usize) -> Option<usize> {
        let index = match key {
            Live::Integer(index) => usize::try_from(index).ok()?,
            Live::Float(value) => {
                // Integral and inside the array: a whole number from 1 to
                // `len`. NaN fails the range test.
                #[allow(clippy::cast_precision_loss)]
                let last = len as f64;
                if !(1.0..=last).contains(&value) || value.fract() != 0.0 {
                    return None;
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    value as usize
                }
            }
        };
        (1..=len).contains(&index).then(|| index - 1)
    }

    /// The type a table slot holding `value` has for the trace, if it is one
    /// native code can read and overwrite in place.
    fn slot_type(&self, value: &Value) -> Option<TraceType> {
        match Kind::of(value, self.integers) {
            Kind::Boolean => Some(TraceType::Boolean),
            Kind::Integer => Some(TraceType::Integer),
            Kind::Float => Some(TraceType::Float),
            _ => None,
        }
    }

    /// The type an array slot holding `value` has for a read: a native type,
    /// or a table, a Lua closure or a coroutine, which a read hands native
    /// code as a lane.
    fn array_slot_type(&self, value: &Value) -> Option<TraceType> {
        match value {
            Value::Table(_) => Some(TraceType::Table),
            Value::Closure(_) => Some(TraceType::Closure),
            Value::Thread(_) => Some(TraceType::Thread),
            // A string is borrowed from the slot, not given a lane.
            Value::String(_) => Some(TraceType::StringRef),
            other => self.slot_type(other),
        }
    }

    /// The key a field instruction names in constant `index`.
    fn field_key(at: &HookPoint<'_>, index: u8) -> Result<TableKey, TraceAbortReason> {
        match at.chunk.constants.get(usize::from(index)) {
            Some(Constant::String(text)) => Ok(TableKey::Field(text.clone())),
            _ => Err(TraceAbortReason::Divergence),
        }
    }

    /// Records a `SelfField` (`copies_receiver`) or a `GetField` on a string:
    /// `s:upper ()` and the like. The method is looked up through the string
    /// metatable's `__index` table, which the trace does not read. A helper
    /// checks on every run that the metatable, its `__index` and the field
    /// are still the ones the recording saw (the field holds the pinned
    /// native), and leaves through the `Before` snapshot of the instruction
    /// otherwise. The method is then that native as a constant, and the call
    /// after it records as the `string.f (s, ...)` form does.
    fn record_string_method(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        copies_receiver: bool,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        self.check_register(a)?;
        if copies_receiver {
            self.check_register(a + 1)?;
        }
        let TableKey::Field(name) = Self::field_key(at, instruction.c())? else {
            return Err(TraceAbortReason::Divergence);
        };
        // The receiver is an argument of the check, so it joins the entry
        // contract as a string: a later entry with another type there must
        // not run this trace.
        let receiver = if let Sym::Root(slot) = self.sym(b) {
            self.root_operand(pc, slot)
        } else {
            let (operand, ty) = self.read(at.registers, b)?;
            if ty != TraceType::String && !ty.is_borrowed() {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            (operand, ty)
        };
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let Ok(Some(metatable)) = runtime.raw_metatable(&at.registers[b]) else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let Some(Value::Table(index)) =
            runtime.table_get_string_ref(metatable, &crate::string::LuaString::from("__index"))
        else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let method = runtime
            .table_get_string_ref(*index, &name)
            .ok_or(TraceAbortReason::TableMiss)?;
        // A method with no identity to pin is the builtin that aborts.
        let Some(identity) = method_identity(method) else {
            self.note_callable(method);
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let lane = self.pin(identity)?;
        let slot = self.root_slot(name.as_bytes())?;
        let (key, _) = self.root_operand(pc, slot);
        let identity = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
        self.emit_helper_call(
            pc,
            HelperId::STRING_METHOD,
            &[
                (key, TraceType::String),
                (Operand::Op(identity), TraceType::Integer),
                receiver,
            ],
        );
        if copies_receiver {
            self.record_move(at.registers, pc, a + 1, b)?;
        }
        let callee = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        self.write(a, callee);
        Ok(pc + 1)
    }

    /// Records a `GetTable`, or with `field` a `GetField`.
    fn record_get_table(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        field: bool,
    ) -> Result<usize, TraceAbortReason> {
        self.check_register(usize::from(instruction.a()))?;
        let receiver = self.table_receiver(at, usize::from(instruction.b()))?;
        self.record_get_table_of(at, instruction, field, receiver)
    }

    /// [`Self::record_get_table`] on a table already found.
    fn record_get_table_of(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        field: bool,
        receiver: (Receiver, TableHandle, &Runtime),
    ) -> Result<usize, TraceAbortReason> {
        if !field
            && matches!(
                at.registers.get(usize::from(instruction.c())),
                Some(Value::Nil)
            )
        {
            let a = usize::from(instruction.a());
            self.check_register(a)?;
            let value = self.read_nil_key(at, usize::from(instruction.c()), receiver)?;
            self.write(a, value);
            return Ok(at.pc + 1);
        }
        let key = if field {
            Self::field_key(at, instruction.c())?
        } else {
            self.table_key(at, usize::from(instruction.c()))?
        };
        self.record_get_keyed(at, usize::from(instruction.a()), key, receiver)
    }

    /// `t[nil]` with the key in `register`: nil, the value of the read of
    /// any absent key, when the table has no metatable (Mineclonia's
    /// `static_node_shapes[cid]` at nodeprops.lua:410 and `is_cid_solid[cid]`
    /// at :1111, where `get_block` gives no content id outside the chunk).
    /// The key's type is the trace's, so only the table needs a check: a
    /// [`HelperId::TABLE_ABSENT_INDEX`] call with the nil key, which leaves
    /// if the table has gained a metatable. Only a root trace makes the
    /// call, as for the number key a table lacks.
    fn read_nil_key(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
        (receiver, handle, runtime): (Receiver, TableHandle, &Runtime),
    ) -> Result<u32, TraceAbortReason> {
        let (key, ty) = self.read(at.registers, register)?;
        if ty != TraceType::Nil || !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if self.function.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let table = match receiver {
            Receiver::Bound(window) | Receiver::Around(window) => Operand::Carried(window),
            Receiver::Loaded(table) => table,
        };
        self.emit_helper_call(
            at.pc,
            HelperId::TABLE_ABSENT_INDEX,
            &[(table, TraceType::Table), (key, TraceType::Nil)],
        );
        Ok(self.emit(
            at.pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        ))
    }

    /// Records the read of `key` from the table of `receiver` into register
    /// `a`.
    fn record_get_keyed(
        &mut self,
        at: &HookPoint<'_>,
        a: usize,
        key: TableKey,
        receiver: (Receiver, TableHandle, &Runtime),
    ) -> Result<usize, TraceAbortReason> {
        self.check_register(a)?;
        self.number_wanted = Self::first_reader_takes_numbers(at, a);
        let value = self.read_keyed(at, key, receiver, false);
        self.number_wanted = false;
        self.write(a, value?);
        Ok(at.pc + 1)
    }

    /// The operation that reads `key` from the table of `receiver`. With
    /// `raw` the read is `rawget`'s: a key the table does not hold itself is
    /// nil whatever `__index` says, so only a key the table holds can be read,
    /// and a function stored in a field is no constant.
    fn read_keyed(
        &mut self,
        at: &HookPoint<'_>,
        key: TableKey,
        (receiver, handle, runtime): (Receiver, TableHandle, &Runtime),
        raw: bool,
    ) -> Result<u32, TraceAbortReason> {
        let pc = at.pc;
        let value = match key {
            TableKey::Field(text) => {
                // Only a present value of a native type, or a function: a
                // missing key runs `__index`, which a chain of tables
                // answers from the table that holds the field, and
                // anything else has no payload to read.
                let Some((holder, depth)) =
                    runtime.table_string_holder(handle, &text, MAX_INDEX_HOPS)
                else {
                    return self.read_absent_field(pc, (receiver, handle, runtime), &text);
                };
                if raw && depth != 0 {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let live = runtime
                    .table_get_string_ref(holder, &text)
                    .ok_or(TraceAbortReason::Divergence)?;
                // A table known only by lane has no entry binding to check a
                // function or a string against, and its row holds its own
                // fields of a number, a boolean or a table: the rest is one
                // call that looks the field up as the interpreter does.
                //
                // A rowless recording (see [`Site::rowless`]) reads every
                // field that way: a store that moves hash storage sits in
                // the loop, and the address of a field, bound at entry or in
                // a lane's row, would not survive it.
                let callable_or_text = matches!(
                    live,
                    Value::Closure(_)
                        | Value::NativeFunction(_)
                        | Value::BuiltinFunction(_)
                        | Value::String(_)
                );
                // A userdata has no binding either: it is a lane the lookup
                // call gives.
                let userdata = matches!(live, Value::Userdata(_));
                // A closure that a call at a site taking turns between
                // closures takes (`fork:next_integer`, a new closure each
                // fork) is no constant of the run: the call guards it on
                // its prototype, and the trace reads it at each iteration.
                let poly_callee = !raw
                    && matches!(live, Value::Closure(_))
                    && at.chunk.code.get(pc).is_some_and(|instruction| {
                        self.reads_poly_callee(at, usize::from(instruction.a()))
                    });
                let looked_up = match self.field_access_in_place(receiver) {
                    FieldAccess::Lane(table) => {
                        (depth != 0 || callable_or_text || userdata || self.rowless)
                            .then_some(table)
                    }
                    FieldAccess::Bound(window)
                        if self.function.is_none()
                            && (userdata || poly_callee || (self.rowless && !callable_or_text)) =>
                    {
                        Some(Operand::Carried(window))
                    }
                    FieldAccess::Bound(_) => None,
                };
                if let Some(table) = looked_up {
                    let key = self.field_key_operand(pc, &text)?;
                    return self.lane_read(
                        pc,
                        (table, handle, runtime),
                        (key, TraceType::String, Value::String(text)),
                        depth != 0,
                    );
                }
                let identity = match live {
                    Value::Closure(handle) => Some(handle.native_identity()),
                    Value::NativeFunction(function) => {
                        Some(pinned_native_identity(function.index()))
                    }
                    Value::BuiltinFunction(BuiltinFunction::ToString) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_TOSTRING))
                    }
                    Value::BuiltinFunction(BuiltinFunction::StringFormat) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_STRING_FORMAT))
                    }
                    Value::BuiltinFunction(BuiltinFunction::StringGsub) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_STRING_GSUB))
                    }
                    Value::BuiltinFunction(BuiltinFunction::Next) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_NEXT))
                    }
                    Value::BuiltinFunction(BuiltinFunction::Pairs) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_PAIRS))
                    }
                    Value::BuiltinFunction(BuiltinFunction::IPairs) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_IPAIRS))
                    }
                    Value::BuiltinFunction(BuiltinFunction::ProtectedCall) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_PCALL))
                    }
                    Value::BuiltinFunction(BuiltinFunction::ExtendedProtectedCall) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_XPCALL))
                    }
                    Value::BuiltinFunction(BuiltinFunction::DebugTraceback) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_DEBUG_TRACEBACK))
                    }
                    Value::BuiltinFunction(BuiltinFunction::CoroutineResume) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_COROUTINE_RESUME))
                    }
                    Value::BuiltinFunction(BuiltinFunction::CoroutineYield) => {
                        Some(pinned_builtin_identity(PINNED_BUILTIN_COROUTINE_YIELD))
                    }
                    _ => None,
                };
                if let Some(identity) = identity.filter(|_| !raw) {
                    // A function in a field (a method, `math.sin`). A store
                    // the trace makes under this key misses (see
                    // `RunTables::pinned_keys`), so nothing in a run replaces
                    // it: the entry checks that the field holds this very
                    // function and the trace reads it as that constant.
                    // A table met at run time (a lane receiver) has no entry
                    // binding to check the function against, and its fields
                    // are found without `__index`: refuse.
                    let FieldAccess::Bound(window) = self.field_access_in_place(receiver) else {
                        return Err(TraceAbortReason::UnsupportedOperand);
                    };
                    let lane = self.pin(identity)?;
                    // No operation reads the table, but the binding needs it
                    // as an entry input.
                    self.condition_reads.push(window);
                    self.binding(
                        window,
                        BindingKind::Callee {
                            key: text.as_bytes().into(),
                            lane,
                            depth,
                        },
                    );
                    return Ok(self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
                        Some(TraceType::Closure),
                    ));
                }
                if let (Value::String(text_value), false) = (live, raw) {
                    // A string in a field (`utf8.charpattern`). A store the
                    // trace makes under this key either keeps a number or
                    // boolean or misses (see `RunTables::pinned_keys`), so
                    // nothing in a run replaces it: the entry checks that the
                    // field holds these bytes and the trace reads it as a
                    // constant.
                    let FieldAccess::Bound(window) = self.field_access_in_place(receiver) else {
                        return Err(TraceAbortReason::UnsupportedOperand);
                    };
                    let slot = self.root_slot(text_value.as_bytes())?;
                    self.condition_reads.push(window);
                    self.binding(
                        window,
                        BindingKind::StringField {
                            key: text.as_bytes().into(),
                            text: text_value.as_bytes().into(),
                            depth,
                        },
                    );
                    return Ok(self.root_constant(pc, slot));
                }
                self.note_callable(live);
                // A table in a field is read as the lane the run gives it
                // (see
                // [`TableFieldLoad`](crate::jit_ir::trace::TraceOp::TableFieldLoad)),
                // from the table's row: its own field, found without
                // `__index`.
                let lane_table = matches!(live, Value::Table(_)) && depth == 0;
                let access = if lane_table {
                    match self.field_access(receiver) {
                        FieldAccess::Bound(window) => FieldAccess::Lane(Operand::Carried(window)),
                        lane @ FieldAccess::Lane(_) => lane,
                    }
                } else {
                    self.field_access(receiver)
                };
                let ty = if lane_table {
                    TraceType::Table
                } else {
                    self.slot_type(live)
                        .ok_or(TraceAbortReason::UnsupportedOperand)?
                };
                match access {
                    FieldAccess::Bound(window) => {
                        let binding = self.binding(
                            window,
                            BindingKind::Field {
                                key: text.as_bytes().into(),
                                ty,
                                depth,
                            },
                        );
                        self.emit(
                            pc,
                            RecKind::FieldLoad {
                                binding,
                                table: Operand::Carried(window),
                            },
                            Some(ty),
                        )
                    }
                    FieldAccess::Lane(table) => {
                        // The lane's row holds the table's own fields only: a
                        // field found through `__index` has no slot there.
                        if depth != 0 {
                            return Err(TraceAbortReason::UnsupportedOperand);
                        }
                        let field = self.field(&text, ty)?;
                        let snapshot = self.read_snapshot(pc);
                        self.emit(
                            pc,
                            RecKind::TableFieldLoad {
                                field,
                                table,
                                snapshot,
                            },
                            Some(ty),
                        )
                    }
                }
            }
            TableKey::Hash(key, text) => {
                // Only a table the entry binds. A key the recording found
                // absent needs a table without a metatable: a missing key is
                // then nil whatever the trace does, since nothing it does
                // can give a table a metatable, and the lookup never runs
                // `__index`. A key it found present may sit in a table with
                // a metatable: the lookup leaves unless the value has the
                // expected type, so a key that went missing never skips
                // `__index`. The result is a number, a boolean or nil.
                if let Receiver::Loaded(table) = receiver {
                    let operand_type = self.operand_type(key);
                    return self.lane_read(
                        pc,
                        (table, handle, runtime),
                        (key, operand_type, Value::String(text)),
                        false,
                    );
                }
                let window = Self::bound_window(receiver)?;
                let plain = matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None));
                let expected = match runtime.table_get_string_ref(handle, &text) {
                    None => TraceType::Nil,
                    Some(live) => {
                        self.note_callable(live);
                        self.slot_type(live)
                            .ok_or(TraceAbortReason::UnsupportedOperand)?
                    }
                };
                let binding = if expected == TraceType::Nil {
                    if !plain {
                        return Err(TraceAbortReason::UnsupportedOperand);
                    }
                    self.binding(window, BindingKind::NoMetatable)
                } else {
                    self.binding(window, BindingKind::Hash)
                };
                let snapshot = self.read_snapshot(pc);
                self.emit(
                    pc,
                    RecKind::HashLoad {
                        binding,
                        table: Operand::Carried(window),
                        key,
                        expected,
                        snapshot,
                    },
                    Some(expected),
                )
            }
            TableKey::Object(key, live_key) => {
                // One lookup call through the table's lane: a table the
                // entry binds is its own lane. The call leaves unless the
                // table holds the key with a value of the type it held now,
                // so a key gone missing never skips `__index`.
                let table = match receiver {
                    Receiver::Loaded(table) => table,
                    Receiver::Bound(window) | Receiver::Around(window) => Operand::Carried(window),
                };
                let operand_type = self.operand_type(key);
                return self.lane_read(
                    pc,
                    (table, handle, runtime),
                    (key, operand_type, live_key),
                    false,
                );
            }
            TableKey::Index(key) => {
                // A table known only by lane has no array to bind: its
                // array words come through its row, or one call reads the
                // slot.
                if let Receiver::Loaded(table) = receiver {
                    if let Some(read) = self.lane_array_read(pc, table, (handle, runtime), key)? {
                        return Ok(read);
                    }
                    let (operand_type, live) = match key.live {
                        Live::Integer(index) => (TraceType::Integer, Value::Integer(index)),
                        Live::Float(number) => (TraceType::Float, Value::Number(number)),
                    };
                    // A key the table lacks, in a table without a metatable:
                    // nil, behind a call that leaves unless the key is still
                    // absent and the table still has no metatable (the
                    // stronghold starts of Mineclonia's concentric ring
                    // `chunk_test`, `is_start[hash]` at structures.lua:406,
                    // hold a few chunks of the many it asks about).
                    if !raw
                        && self.function.is_none()
                        && matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None))
                        && matches!(
                            runtime.table_get_raw_value(handle, &live),
                            Ok(None | Some(Value::Nil))
                        )
                        && HelperType::Value.accepts(operand_type)
                    {
                        self.emit_helper_call(
                            pc,
                            HelperId::TABLE_ABSENT_INDEX,
                            &[(table, TraceType::Table), (key.operand, operand_type)],
                        );
                        return Ok(self.emit(
                            pc,
                            RecKind::Constant(TraceScalar::Nil),
                            Some(TraceType::Nil),
                        ));
                    }
                    return self.lane_read(
                        pc,
                        (table, handle, runtime),
                        (key.operand, operand_type, live),
                        false,
                    );
                }
                // Only a table the entry binds has an array to read.
                let window = Self::bound_window(receiver)?;
                // A userdata in the slot has no array word to read: one call
                // looks the slot up and gives its lane.
                let live_key = match key.live {
                    Live::Integer(index) => Value::Integer(index),
                    Live::Float(number) => Value::Number(number),
                };
                if self.function.is_none()
                    && matches!(
                        runtime.table_get_raw_value(handle, &live_key),
                        Ok(Some(Value::Userdata(_)))
                    )
                {
                    let key_type = match key.live {
                        Live::Integer(_) => TraceType::Integer,
                        Live::Float(_) => TraceType::Float,
                    };
                    return self.lane_read(
                        pc,
                        (Operand::Carried(window), handle, runtime),
                        (key.operand, key_type, live_key),
                        false,
                    );
                }
                // A table held in two registers cannot bind its header.
                let past_end = !self.table_is_aliased(at.registers, window, handle);
                self.array_read_or_miss(
                    pc,
                    (Operand::Carried(window), window),
                    (handle, runtime),
                    key,
                    (true, past_end),
                )?
                .0
            }
        };
        Ok(value)
    }

    /// The operand of the string constant `text`, a field name, for a call.
    fn field_key_operand(
        &mut self,
        pc: usize,
        text: &crate::string::LuaString,
    ) -> Result<Operand, TraceAbortReason> {
        let slot = self.root_slot(text.as_bytes())?;
        Ok(self.root_operand(pc, slot).0)
    }

    /// The read of `key` (an operand of type `key_type` that holds `live`) from
    /// the table in lane `table`, for a table the entry binds nothing for: one
    /// call into the VM that reads as `rawget` does, or, with `walk`, follows
    /// the tables `__index` names from a key the table lacks. The call leaves
    /// before the instruction unless the key is present with a value of the
    /// type recorded, so the interpreter runs a nil, a function `__index`, a
    /// type that changed or an error. See [`HelperId::TABLE_GET_INTEGER`].
    fn lane_read(
        &mut self,
        pc: usize,
        (table, handle, runtime): (Operand, TableHandle, &Runtime),
        (key, key_type, live_key): (Operand, TraceType, Value),
        walk: bool,
    ) -> Result<u32, TraceAbortReason> {
        let found = super::helpers::table_read_value(runtime, handle, &live_key, walk)
            .ok_or(TraceAbortReason::TableMiss)?;
        let helper = match &found {
            Value::Integer(_) => HelperId::TABLE_GET_INTEGER,
            Value::Number(_) => HelperId::TABLE_GET_FLOAT,
            Value::Boolean(_) => HelperId::TABLE_GET_BOOLEAN,
            Value::Table(_) => HelperId::TABLE_GET_TABLE,
            Value::Closure(_) => HelperId::TABLE_GET_CLOSURE,
            Value::Userdata(_) => HelperId::TABLE_GET_USERDATA,
            Value::String(_) => HelperId::TABLE_GET_STRING,
            other => {
                self.note_callable(other);
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        };
        if !HelperType::Value.accepts(key_type) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let walk = self.scalar(pc, TraceScalar::Integer(i64::from(walk)));
        Ok(self.emit_helper_call(
            pc,
            helper,
            &[
                (table, TraceType::Table),
                (key, key_type),
                (Operand::Op(walk), TraceType::Integer),
            ],
        ))
    }

    /// The read of the array slot `key` names of the table in lane `table`,
    /// a table met at run time, inline ([`TraceOp::LaneArrayLoad`]): the
    /// array words come from the table, whose address the lane's row holds.
    /// Only for a key that names a slot inside the array part now, holding a
    /// number, a boolean or a table; the read then leaves before the
    /// instruction unless the slot holds a value of that type when the trace
    /// runs, so no `__index` can run. `None` leaves the read to the lookup
    /// call ([`Self::lane_read`]): a string, a closure, a nil or a key of
    /// the hash part, a function trace (whose runs may give one table two
    /// lanes), a number of either subtype the read must keep apart, or a
    /// build whose table layout native code cannot read.
    fn lane_array_read(
        &mut self,
        pc: usize,
        table: Operand,
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
    ) -> Result<Option<u32>, TraceAbortReason> {
        if self.function.is_some()
            || crate::runtime::NativeTableLayout::get().is_none()
            || !HelperType::Value.accepts(self.operand_type(key.operand))
        {
            return Ok(None);
        }
        let len = runtime
            .table_array_len(handle)
            .map_err(|_| TraceAbortReason::Divergence)?;
        let Some(slot) = Self::array_slot(key.live, len) else {
            return Ok(None);
        };
        let live = runtime
            .table_raw_array_range(handle, slot + 1, slot + 1)
            .ok()
            .flatten()
            .and_then(<[Value]>::first)
            .ok_or(TraceAbortReason::Divergence)?;
        let expected = match live {
            Value::Table(_) => TraceType::Table,
            // A loop side borrows a string from the slot: it may not make
            // lanes its iterations would leave to the root's loop to release
            // (see `finalize`).
            Value::String(_) if self.looping => TraceType::StringRef,
            other => match self.slot_type(other) {
                Some(ty) => ty,
                None => return Ok(None),
            },
        };
        if matches!(expected, TraceType::Integer | TraceType::Float)
            && self.wants_number_load((handle, runtime), live)
        {
            return Ok(None);
        }
        if let Some(value) = self.forward_slot_read(pc, table, key.operand, expected) {
            return Ok(Some(value));
        }
        let field = self.field(
            &crate::string::LuaString::from(TABLE_ADDRESS_KEY),
            TraceType::Integer,
        )?;
        let snapshot = self.read_snapshot(pc);
        let value = self.emit(
            pc,
            RecKind::LaneArrayLoad {
                table,
                field,
                key: key.operand,
                expected,
                snapshot,
            },
            Some(expected),
        );
        self.note_slot_read(table, key.operand, value, expected);
        Ok(Some(value))
    }

    /// The store of `value`, of type `ty`, into the array slot `key` names
    /// of the table in lane `table`, inline ([`TraceOp::LaneArrayStore`]):
    /// the store of [`Self::lane_array_read`]. Only for a key that names a
    /// slot inside the array part now, holding a non-nil number or boolean
    /// of the stored value's type, in a writable table; the store then
    /// leaves before the instruction unless that still holds when the trace
    /// runs, so the payload overwrite moves nothing and no `__newindex` can
    /// run. `false` leaves the store to the raw-set call
    /// ([`HelperId::TABLE_SET_INDEX`]), under the same conditions as
    /// [`Self::lane_array_read`] leaves a read to its call.
    ///
    /// The call's set may move hash storage, so a tree that holds one
    /// refuses lanes that address storage
    /// ([`TraceIr::storage_a_set_would_move`]). Mineclonia's
    /// `mark_occupied` (`region.solids[idx] = bor (region.solids[idx],
    /// mask)`, `mcl_util/shape.lua`) stored through that call beside the
    /// field read of `region.solids`, and every recording of the
    /// `region_op` loop that met it was refused.
    fn lane_array_store(
        &mut self,
        pc: usize,
        table: Operand,
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
        (value, ty): (Operand, TraceType),
    ) -> Result<bool, TraceAbortReason> {
        if self.function.is_some()
            || crate::runtime::NativeTableLayout::get().is_none()
            || !matches!(
                ty,
                TraceType::Integer | TraceType::Float | TraceType::Boolean
            )
            || !HelperType::Value.accepts(self.operand_type(key.operand))
            || runtime.table_is_readonly(handle).unwrap_or(true)
        {
            return Ok(false);
        }
        let len = runtime
            .table_array_len(handle)
            .map_err(|_| TraceAbortReason::Divergence)?;
        let Some(slot) = Self::array_slot(key.live, len) else {
            return Ok(false);
        };
        let in_place = runtime
            .table_raw_array_range(handle, slot + 1, slot + 1)
            .ok()
            .flatten()
            .and_then(<[Value]>::first)
            .is_some_and(|live| self.slot_type(live) == Some(ty));
        if !in_place {
            return Ok(false);
        }
        let field = self.field(
            &crate::string::LuaString::from(TABLE_ADDRESS_KEY),
            TraceType::Integer,
        )?;
        self.note_type_read(key.operand);
        self.note_type_read(value);
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::LaneArrayStore {
                table,
                field,
                key: key.operand,
                value,
                snapshot,
            },
            None,
        );
        self.note_slot_value(table, key.operand, value, ty);
        Ok(true)
    }

    /// The read of the field `text`, which the table of `receiver` lacks
    /// (`if math.pow then`, a feature test): nil, with a guard that the key
    /// is still absent when the trace runs. Only a bound table without a
    /// metatable: a missing key runs no `__index` there.
    fn read_absent_field(
        &mut self,
        pc: usize,
        (receiver, handle, runtime): (Receiver, TableHandle, &Runtime),
        text: &crate::string::LuaString,
    ) -> Result<u32, TraceAbortReason> {
        if !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None))
            || runtime.table_get_string_ref(handle, text).is_some()
        {
            return Err(TraceAbortReason::TableMiss);
        }
        if let (Receiver::Loaded(table), true) = (receiver, self.function.is_some()) {
            // A table met through its lane: the row says the key is absent
            // and the table has no metatable, and a function trace stores
            // nothing that could add the key.
            let field = self.field(text, TraceType::Nil)?;
            let snapshot = self.read_snapshot(pc);
            return Ok(self.emit(
                pc,
                RecKind::TableFieldLoad {
                    field,
                    table,
                    snapshot,
                },
                Some(TraceType::Nil),
            ));
        }
        let window = Self::bound_window(receiver).map_err(|_| TraceAbortReason::TableMiss)?;
        let slot = self.root_slot(text.as_bytes())?;
        let (key, _) = self.root_operand(pc, slot);
        let binding = self.binding(window, BindingKind::NoMetatable);
        let snapshot = self.read_snapshot(pc);
        Ok(self.emit(
            pc,
            RecKind::HashLoad {
                binding,
                table: Operand::Carried(window),
                key,
                expected: TraceType::Nil,
                snapshot,
            },
            Some(TraceType::Nil),
        ))
    }

    /// Reads the array slot `key` names of the bound table, for the guest
    /// instruction at `pc`. The slot must hold a native value now, which
    /// decides the type the read expects. Returns the operation and that
    /// type.
    ///
    /// With `holes`, a nil slot inside the array of a table without a
    /// metatable reads as nil: no `__index` can run for it.
    fn array_read(
        &mut self,
        pc: usize,
        (table, window): (Operand, usize),
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
        holes: bool,
    ) -> Result<(u32, TraceType), TraceAbortReason> {
        self.array_read_or_miss(pc, (table, window), (handle, runtime), key, (holes, false))
    }

    /// [`Self::array_read`] that, with `past_end` (which needs `holes`),
    /// also reads nil for a key outside the array part: see
    /// [`Self::array_miss`].
    fn array_read_or_miss(
        &mut self,
        pc: usize,
        (table, window): (Operand, usize),
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
        (holes, past_end): (bool, bool),
    ) -> Result<(u32, TraceType), TraceAbortReason> {
        let len = runtime
            .table_array_len(handle)
            .map_err(|_| TraceAbortReason::Divergence)?;
        let Some(slot) = Self::array_slot(key.live, len) else {
            if !holes {
                return Err(TraceAbortReason::TableMiss);
            }
            if past_end
                && let Ok(read) = self.array_miss(pc, (table, window), (handle, runtime), key)
            {
                return Ok(read);
            }
            return self.number_hash_read(pc, (table, window), (handle, runtime), key);
        };
        let live = runtime
            .table_raw_array_range(handle, slot + 1, slot + 1)
            .ok()
            .flatten()
            .and_then(<[Value]>::first)
            .ok_or(TraceAbortReason::Divergence)?;
        let plain = holes
            && *live == Value::Nil
            && matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None));
        let expected = if plain {
            self.binding(window, BindingKind::NoMetatable);
            TraceType::Nil
        } else {
            self.note_callable(live);
            self.array_slot_type(live).ok_or(if *live == Value::Nil {
                TraceAbortReason::TableMiss
            } else {
                TraceAbortReason::UnsupportedOperand
            })?
        };
        let number = matches!(expected, TraceType::Integer | TraceType::Float)
            && self.wants_number_load((handle, runtime), live);
        // A slot this iteration already read or stored is not read again
        // (see `slot_facts`).
        if !number
            && !plain
            && let Some(value) = self.forward_slot_read(pc, table, key.operand, expected)
        {
            return Ok((value, expected));
        }
        let first = self.first_slot_type(runtime, handle);
        let binding = self.binding(window, BindingKind::Array { first });
        let snapshot = self.read_snapshot(pc);
        if number {
            let value = self.emit_number_load(pc, binding, (table, key.operand), snapshot);
            return Ok((value, TraceType::Float));
        }
        let value = self.emit(
            pc,
            RecKind::ArrayLoad {
                binding,
                table,
                key: key.operand,
                expected,
                past_end: false,
                snapshot,
            },
            Some(expected),
        );
        if !plain {
            self.note_slot_read(table, key.operand, value, expected);
        }
        Ok((value, expected))
    }

    /// The read of a key outside the array part of the bound table (past its
    /// end, below 1 or fractional), which reads nil when the table has no
    /// metatable and an empty hash part: no `__index` can run and the key is
    /// not stored anywhere. The trace binds the table's header, whose word
    /// says again at run time that the hash part is empty. Any other table
    /// misses the way the recording always did.
    fn array_miss(
        &mut self,
        pc: usize,
        (table, window): (Operand, usize),
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
    ) -> Result<(u32, TraceType), TraceAbortReason> {
        let solid = runtime
            .table_native_header(handle)
            .is_ok_and(|header| header.solid);
        if !solid || !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None)) {
            return Err(TraceAbortReason::TableMiss);
        }
        self.binding(window, BindingKind::NoMetatable);
        self.binding(window, BindingKind::Header { strict: false });
        let first = self.first_slot_type(runtime, handle);
        let binding = self.binding(window, BindingKind::Array { first });
        let snapshot = self.read_snapshot(pc);
        let value = self.emit(
            pc,
            RecKind::ArrayLoad {
                binding,
                table,
                key: key.operand,
                expected: TraceType::Nil,
                past_end: true,
                snapshot,
            },
            Some(TraceType::Nil),
        );
        Ok((value, TraceType::Nil))
    }

    /// The read of a number key outside the array part of the bound table
    /// when its hash part is not empty (`t[0]`, a negative or sparse
    /// integer key, a fractional one): a raw lookup over both parts at run
    /// time ([`TraceOp::HashLoad`](crate::jit_ir::trace::TraceOp::HashLoad)
    /// with a number key), so a key that moves between the parts is still
    /// found. A key found needs a scalar value of the type the recording
    /// saw, and the lookup leaves unless it has that type. An absent key
    /// reads as nil only on a table without a metatable.
    fn number_hash_read(
        &mut self,
        pc: usize,
        (table, window): (Operand, usize),
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
    ) -> Result<(u32, TraceType), TraceAbortReason> {
        let plain = matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None));
        let live_key = match key.live {
            Live::Integer(index) => Value::Integer(index),
            Live::Float(value) => Value::Number(value),
        };
        let expected = match runtime.table_get_raw_value(handle, &live_key) {
            Ok(None | Some(Value::Nil)) => {
                if !plain {
                    return Err(TraceAbortReason::TableMiss);
                }
                TraceType::Nil
            }
            Ok(Some(live)) => match self.slot_type(live) {
                Some(ty) => ty,
                // A table, a function, a string or a userdata under a
                // sparse number key (the structure starts of Mineclonia's
                // `collect_structure_references`, `starts[hash]` at
                // structures.lua:856, keyed by `struct_hash`): no word of
                // the slot is a value native code can carry, so one lookup
                // call reads it as `rawget` does and gives its lane, and
                // leaves before the instruction unless the key still holds
                // a value of that type. A function trace may give one table
                // two lanes, so it keeps refusing.
                None if self.function.is_none()
                    && matches!(
                        live,
                        Value::Table(_) | Value::Closure(_) | Value::String(_) | Value::Userdata(_)
                    ) =>
                {
                    let key_type = self.operand_type(key.operand);
                    let value = self.lane_read(
                        pc,
                        (table, handle, runtime),
                        (key.operand, key_type, live_key),
                        false,
                    )?;
                    return Ok((value, self.operand_type(Operand::Op(value))));
                }
                None => {
                    self.note_callable(live);
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
            },
            Err(_) => return Err(TraceAbortReason::TableMiss),
        };
        let binding = if expected == TraceType::Nil {
            self.binding(window, BindingKind::NoMetatable)
        } else {
            self.binding(window, BindingKind::Hash)
        };
        let snapshot = self.read_snapshot(pc);
        let value = self.emit(
            pc,
            RecKind::HashLoad {
                binding,
                table,
                key: key.operand,
                expected,
                snapshot,
            },
            Some(expected),
        );
        Ok((value, expected))
    }

    /// Binds the header of the table in window register `window`, which lets
    /// native code append, pop and overwrite in place, for a set of a value
    /// of type `ty` under `key` into a list of `len` slots. A run refuses a
    /// table held in two registers. The header is strict when the store is
    /// at an end of a list that takes it in place now: a store that finds
    /// anything else leaves, and calls no VM.
    fn bind_header(
        &mut self,
        window: usize,
        (handle, runtime): (TableHandle, &Runtime),
        (key, ty): (Live, TraceType),
        len: usize,
    ) {
        let index = match key {
            Live::Integer(index) => usize::try_from(index).ok(),
            Live::Float(index) if index >= 0.0 && index.fract() == 0.0 =>
            {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Some(index as usize)
            }
            Live::Float(_) => None,
        };
        let at_the_end = index.is_some_and(|index| {
            index == len + 1 || (ty == TraceType::Nil && index == len && index >= 1)
        });
        let appendable = runtime
            .table_native_header(handle)
            .is_ok_and(|header| header.appendable);
        // A table stored over a table of a dense list is done in place
        // (the VM's store entry owns the reference counts and the barrier),
        // so a ring that is full keeps its sets in the trace like an append.
        let overwrites_table =
            ty == TraceType::Table && index.is_some_and(|index| (1..=len).contains(&index));
        let strict = (at_the_end || overwrites_table) && appendable;
        // A table that is no list yet (a ring still filling) turns into one:
        // a recording that cannot hold its set now is worth another try.
        self.set_waiting |= !strict && !appendable && index.is_some();
        self.binding(window, BindingKind::Header { strict });
    }

    /// Whether a register other than `window` holds `table` as it came in, so
    /// that a run binds the table in two registers. A register the iteration
    /// copied it into is the same register to the trace.
    fn table_is_aliased(&self, registers: &[Value], window: usize, table: TableHandle) -> bool {
        registers.iter().enumerate().any(|(register, held)| {
            register != window
                && matches!(held, Value::Table(other) if *other == table)
                && match self.frames.last() {
                    Some(frame) => frame.regs.get(register),
                    None => self.regs.get(register),
                }
                .is_some_and(|sym| matches!(sym, Sym::Carried))
        })
    }

    /// Whether the interpreter stores key `key` into `table` raw now: the key
    /// is no NaN, the table is writable, and it has no metatable or already
    /// holds the key (Lua runs `__newindex` only for an absent key). That is
    /// the condition under which [`TraceOp::ArraySet`] stores.
    fn stores_raw(runtime: &Runtime, table: TableHandle, key: Live) -> bool {
        let key = match key {
            Live::Integer(value) => Value::Integer(value),
            Live::Float(value) if value.is_nan() => return false,
            Live::Float(value) => Value::Number(value),
        };
        if runtime.table_is_readonly(table).unwrap_or(true) {
            return false;
        }
        match runtime.raw_metatable(&Value::Table(table)) {
            Ok(None) => true,
            Ok(Some(metatable)) => {
                runtime.metatable_lacks(metatable, crate::runtime::Metamethod::NewIndex)
                    || matches!(runtime.table_get_raw_value(table, &key), Ok(Some(_)))
            }
            Err(_) => false,
        }
    }

    /// Whether the metatable of `table` may have a `__newindex`.
    fn has_newindex(runtime: &Runtime, table: TableHandle) -> bool {
        matches!(
            runtime.raw_metatable(&Value::Table(table)),
            Ok(Some(metatable))
                if !runtime.metatable_lacks(metatable, crate::runtime::Metamethod::NewIndex)
        )
    }

    /// Checks that a store of `key` into `table` calls no function: the walk
    /// of `__newindex` ends in a table that holds the key or has no handler,
    /// which the runtime stores in. Any other end is a call the recorder does
    /// not make here, so the recording aborts, naming the native function or
    /// the metamethod among the aborts of the statistics.
    fn store_walk_is_raw(
        &mut self,
        runtime: &Runtime,
        table: TableHandle,
        key: &crate::string::LuaString,
    ) -> Result<(), TraceAbortReason> {
        match walk_handlers(
            runtime,
            table,
            &Value::String(key.clone()),
            crate::vm::arith::newindex_metamethod_key(),
        ) {
            Walk::Raw(_) | Walk::Holds => Ok(()),
            Walk::Native(function) => {
                self.note_callable(&function);
                Err(TraceAbortReason::UnsupportedOperand)
            }
            Walk::Closure(..) | Walk::Other => {
                self.abort_metamethod = Some("__newindex");
                Err(TraceAbortReason::UnsupportedOperand)
            }
        }
    }

    /// Records a `SetTable`, or with `field` a `SetField`.
    fn record_set_table(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        field: bool,
    ) -> Result<usize, TraceAbortReason> {
        // A function trace stores nothing but the fields of the tables it
        // made: an exit deep in a recursion would have to know which
        // activations stored into a table anyone else holds.
        if self.function.is_some() {
            return if field {
                self.record_made_store(at, instruction)
            } else {
                Err(TraceAbortReason::SelfCall)
            };
        }
        if self.reexec.is_some() {
            // A store is visible after the call starts over.
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let receiver = self.table_receiver(at, usize::from(instruction.a()))?;
        self.record_set_table_of(at, instruction, field, receiver)
    }

    /// [`Self::record_set_table`] on a table already found.
    fn record_set_table_of(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        field: bool,
        receiver: (Receiver, TableHandle, &Runtime),
    ) -> Result<usize, TraceAbortReason> {
        let key = if field {
            Self::field_key(at, instruction.b())?
        } else {
            self.table_key(at, usize::from(instruction.b()))?
        };
        self.record_set_keyed(at, usize::from(instruction.c()), key, receiver)
    }

    /// Records the store of register `c` under `key` into the table of
    /// `receiver`.
    fn record_set_keyed(
        &mut self,
        at: &HookPoint<'_>,
        c: usize,
        key: TableKey,
        receiver: (Receiver, TableHandle, &Runtime),
    ) -> Result<usize, TraceAbortReason> {
        self.record_set_keyed_as(at, c, key, receiver, false)
    }

    /// [`Self::record_set_keyed`], which with `raw` is `rawset`'s: the table
    /// itself takes the value whatever its metatable says.
    fn record_set_keyed_as(
        &mut self,
        at: &HookPoint<'_>,
        c: usize,
        key: TableKey,
        (receiver, handle, runtime): (Receiver, TableHandle, &Runtime),
        raw: bool,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        self.stores += 1;
        let (value, ty) = self.host_operand(pc, at.registers, c)?;
        if !matches!(
            ty,
            TraceType::Integer
                | TraceType::Float
                | TraceType::Boolean
                | TraceType::Nil
                | TraceType::String
                | TraceType::Table
                | TraceType::Closure
        ) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        match key {
            // A store under a table or function key: the interpreter makes
            // it.
            TableKey::Object(..) => return Err(TraceAbortReason::UnsupportedOperand),
            // A store through a run-time string key has no slot to find. A
            // helper makes the interpreter's raw store, or, under `rawset` or
            // a `__newindex`, the store the walk of `__newindex` ends in.
            TableKey::Hash(key, text) => {
                if raw || Self::has_newindex(runtime, handle) {
                    if self.function.is_some() {
                        return Err(TraceAbortReason::SelfCall);
                    }
                    if !raw {
                        self.store_walk_is_raw(runtime, handle, &text)?;
                    }
                    let key_ty = self.operand_type(key);
                    self.record_string_store(pc, receiver, (key, key_ty), (value, ty), raw);
                } else {
                    self.record_set_string(
                        pc,
                        (receiver, handle, runtime),
                        (key, &text),
                        (value, ty),
                    )?;
                }
            }
            TableKey::Field(text) => {
                let present = runtime.table_get_string_ref(handle, &text);
                // A key the table lacks that `rawset` adds, or that a
                // table-valued `__newindex` sends to another table (a
                // function is recorded as a call before this), is the store
                // the walk ends in.
                if present.is_none() && (raw || Self::has_newindex(runtime, handle)) {
                    if self.function.is_some() {
                        return Err(TraceAbortReason::TableMiss);
                    }
                    if !raw {
                        self.store_walk_is_raw(runtime, handle, &text)?;
                    }
                    let slot = self.root_slot(text.as_bytes())?;
                    let key = self.root_operand(pc, slot);
                    self.record_string_store(pc, receiver, key, (value, ty), raw);
                    return Ok(pc + 1);
                }
                // The slot must exist with the stored value's type, so the
                // store is an in-place payload overwrite that `__newindex`
                // and the table's bookkeeping never see. Any other store of
                // a field (nil, a key the table lacks, another type) is the
                // raw store of a helper.
                let in_place = present.is_some_and(|live| self.slot_type(live) == Some(ty));
                if !in_place || self.rowless {
                    let slot = self.root_slot(text.as_bytes())?;
                    let (key, _) = self.root_operand(pc, slot);
                    self.record_set_string(
                        pc,
                        (receiver, handle, runtime),
                        (key, &text),
                        (value, ty),
                    )?;
                    return Ok(pc + 1);
                }
                match self.field_access(receiver) {
                    FieldAccess::Bound(window) => {
                        let binding = self.binding(
                            window,
                            BindingKind::Field {
                                key: text.as_bytes().into(),
                                ty,
                                depth: 0,
                            },
                        );
                        self.emit(
                            pc,
                            RecKind::FieldStore {
                                binding,
                                table: Operand::Carried(window),
                                value,
                            },
                            None,
                        );
                    }
                    FieldAccess::Lane(table) => {
                        let field = self.field(&text, ty)?;
                        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
                        self.emit(
                            pc,
                            RecKind::TableFieldStore {
                                field,
                                table,
                                value,
                                snapshot,
                            },
                            None,
                        );
                    }
                }
            }
            TableKey::Index(key) => {
                // A table known only by lane has no array to bind: one call
                // stores through the VM's raw store, as the interpreter would
                // now, and leaves before the instruction where `__newindex`
                // applies at run time.
                if let Receiver::Loaded(table) = receiver {
                    if self.function.is_some() {
                        return Err(TraceAbortReason::SelfCall);
                    }
                    if !raw && !Self::stores_raw(runtime, handle, key.live) {
                        return Err(TraceAbortReason::UnsupportedOperand);
                    }
                    if self.lane_array_store(pc, table, (handle, runtime), key, (value, ty))? {
                        return Ok(pc + 1);
                    }
                    let key_type = match key.live {
                        Live::Integer(_) => TraceType::Integer,
                        Live::Float(_) => TraceType::Float,
                    };
                    self.note_type_read(key.operand);
                    self.note_type_read(value);
                    let raw = self.scalar(pc, TraceScalar::Integer(i64::from(raw)));
                    self.emit_helper_call(
                        pc,
                        HelperId::TABLE_SET_INDEX,
                        &[
                            (table, TraceType::Table),
                            (key.operand, key_type),
                            (value, ty),
                            (Operand::Op(raw), TraceType::Integer),
                        ],
                    );
                    return Ok(pc + 1);
                }
                let window = Self::bound_window(receiver)?;
                let len = runtime
                    .table_array_len(handle)
                    .map_err(|_| TraceAbortReason::Divergence)?;
                // An overwrite of a slot of the stored type keeps the
                // in-place store. Anything else (a hole, a key past the
                // array or in the hash part, nil) is a raw set, as long as
                // the interpreter would store raw now.
                let in_place = Self::array_slot(key.live, len)
                    .and_then(|slot| {
                        runtime
                            .table_raw_array_range(handle, slot + 1, slot + 1)
                            .ok()
                            .flatten()
                            .and_then(<[Value]>::first)
                    })
                    .is_some_and(|live| self.slot_type(live) == Some(ty));
                if !in_place && !Self::stores_raw(runtime, handle, key.live) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // A set fills a table whatever it holds, often a fresh one:
                // its slot 1 says nothing about the version to run.
                let first = if in_place {
                    self.first_slot_type(runtime, handle)
                } else {
                    None
                };
                let binding = self.binding(window, BindingKind::Array { first });
                if !in_place && !self.table_is_aliased(at.registers, window, handle) {
                    self.bind_header(window, (handle, runtime), (key.live, ty), len);
                }
                let table = Operand::Carried(window);
                // A slot this iteration already reached with a value of the
                // stored type cannot fail the store's guard: the store takes
                // none, and no snapshot (see `slot_facts`).
                if in_place && self.slot_fact(table, key.operand, ty).is_some() {
                    self.slot_overwrites += 1;
                    self.emit(
                        pc,
                        RecKind::ArrayOverwrite {
                            binding,
                            table,
                            key: key.operand,
                            value,
                        },
                        None,
                    );
                    self.note_slot_value(table, key.operand, value, ty);
                    return Ok(pc + 1);
                }
                let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
                self.emit(
                    pc,
                    RecKind::ArrayStore {
                        binding,
                        table,
                        key: key.operand,
                        value,
                        snapshot,
                        set: !in_place,
                    },
                    None,
                );
                if in_place {
                    self.note_slot_value(table, key.operand, value, ty);
                }
            }
        }
        Ok(pc + 1)
    }

    /// Records `table[key] = value` for the string `text`, which `key`
    /// holds, as the interpreter's raw store, made by a helper
    /// ([`HelperId::SET_STRING`]). It may add or delete the key, so it moves
    /// hash storage, and it misses where `__newindex` applies. The recording
    /// refuses a store the interpreter would not make raw now.
    fn record_set_string(
        &mut self,
        pc: usize,
        (receiver, handle, runtime): (Receiver, TableHandle, &Runtime),
        (key, text): (Operand, &crate::string::LuaString),
        (value, ty): (Operand, TraceType),
    ) -> Result<(), TraceAbortReason> {
        let stores = !runtime.table_is_readonly(handle).unwrap_or(true)
            && match runtime.raw_metatable(&Value::Table(handle)) {
                Ok(None) => true,
                Ok(Some(metatable)) => {
                    runtime.metatable_lacks(metatable, crate::runtime::Metamethod::NewIndex)
                        || runtime
                            .table_get_string_ref(handle, text)
                            .is_some_and(|live| !matches!(live, Value::Nil))
                }
                Err(_) => false,
            };
        if !stores {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let table = match receiver {
            Receiver::Bound(window) | Receiver::Around(window) => Operand::Carried(window),
            Receiver::Loaded(table) => table,
        };
        self.note_type_read(key);
        self.note_type_read(value);
        // A store that leaves the table's layout alone has helpers that keep
        // a native step of a traversal valid: an overwrite of a number or a
        // boolean with one of its type, the deletion of a key of a hashed
        // table, and the store of a number or a boolean in a key deleted that
        // way. Any other store may add a key or shift entries.
        //
        // A table the entry binds gets its string index for the three, which
        // native code stores through when the key and the value's type still
        // fit (the helpers answer every other run). A store into a deleted
        // key needs the table to have no metatable, since a nil counts as
        // absent for `__newindex`.
        let sample = match ty {
            TraceType::Integer => Some(Value::Integer(0)),
            TraceType::Float => Some(Value::Number(0.0)),
            TraceType::Boolean => Some(Value::Boolean(false)),
            _ => None,
        };
        let bound = Self::bound_window(receiver).ok();
        let (helper, args): (_, &[(Operand, TraceType)]) =
            if ty == TraceType::Nil && runtime.table_can_delete_string_in_place(handle, text) {
                if let Some(window) = bound {
                    self.binding(window, BindingKind::Hash);
                }
                (
                    HelperId::DELETE_STRING,
                    &[(table, TraceType::Table), (key, TraceType::String)],
                )
            } else if sample.as_ref().is_some_and(|sample| {
                runtime.table_can_overwrite_scalar_string(handle, text, sample)
            }) {
                if let Some(window) = bound {
                    self.binding(window, BindingKind::Hash);
                }
                (
                    HelperId::SET_PRESENT,
                    &[
                        (table, TraceType::Table),
                        (key, TraceType::String),
                        (value, ty),
                    ],
                )
            } else if sample.is_some() && runtime.table_can_revive_string(handle, text) {
                if let Some(window) = bound {
                    self.binding(window, BindingKind::NoMetatable);
                }
                (
                    HelperId::SET_REVIVE,
                    &[
                        (table, TraceType::Table),
                        (key, TraceType::String),
                        (value, ty),
                    ],
                )
            } else {
                (
                    HelperId::SET_STRING,
                    &[
                        (table, TraceType::Table),
                        (key, TraceType::String),
                        (value, ty),
                    ],
                )
            };
        self.emit_helper_call(pc, helper, args);
        Ok(())
    }

    fn record_length(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        self.check_register(a)?;
        if matches!(at.registers.get(b), Some(Value::String(_))) {
            // The length of a noted substring is read from its bounds.
            if let Some(next) = self.record_slice_length(at, (a, b))? {
                return Ok(next);
            }
            // The length of a string is its byte count, whatever the
            // string metatable holds.
            let sym = self.sym(b);
            let bound = if let Sym::Root(_) = sym {
                None
            } else {
                let (string, ty) = self.read(at.registers, b)?;
                if ty == TraceType::String {
                    // A string the loop replaces (a `gmatch` variable)
                    // is no string of the entry's.
                    self.bound_string(string).ok().filter(|window| {
                        !nest::written_within(
                            &self.chunk,
                            LoopSpan {
                                header: self.header,
                                latch: self.latch,
                            },
                            *window,
                        )
                    })
                } else {
                    None
                }
            };
            let Some(window) = bound else {
                return self.record_host_length(at, instruction);
            };
            let number = self.number_type();
            let binding = self.binding(window, BindingKind::Bytes { ty: number });
            let value = self.emit(
                pc,
                RecKind::StringLength {
                    binding,
                    string: Operand::Carried(window),
                },
                Some(number),
            );
            self.write(a, value);
            return Ok(pc + 1);
        }
        let (receiver, handle, runtime) = self.table_receiver(at, b)?;
        // A table with `__len` runs it in the interpreter, in the profiles
        // that ask for it (a Lua closure is recorded as a call before this).
        // The others take the border whatever the metatable holds.
        let raw = !self.honours_len;
        if let Receiver::Loaded(table) = receiver {
            // A table known only by lane has no border to bind: one call
            // reads it, and leaves for a table with a `__len`.
            if !raw
                && !matches!(
                    runtime.table_length_without_len_metamethod(handle),
                    Ok(Some(_))
                )
            {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            let flag = self.scalar(pc, TraceScalar::Integer(i64::from(raw)));
            let length = self.emit_helper_call(
                pc,
                HelperId::TABLE_LENGTH,
                &[
                    (table, TraceType::Table),
                    (Operand::Op(flag), TraceType::Integer),
                ],
            );
            let value = if self.integers {
                length
            } else {
                self.emit(
                    pc,
                    RecKind::IntToFloat(Operand::Op(length)),
                    Some(TraceType::Float),
                )
            };
            self.write(a, value);
            return Ok(pc + 1);
        }
        let window = Self::bound_window(receiver)?;
        if raw {
            runtime.table_length(handle).ok()
        } else {
            runtime
                .table_length_without_len_metamethod(handle)
                .ok()
                .flatten()
        }
        .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let ty = if self.integers {
            TraceType::Integer
        } else {
            TraceType::Float
        };
        let binding = self.binding(window, BindingKind::Length { ty, raw });
        let value = self.emit(
            pc,
            RecKind::TableLength {
                binding,
                table: Operand::Carried(window),
            },
            Some(ty),
        );
        self.write(a, value);
        Ok(pc + 1)
    }

    /// The pinned lane of the callable with this identity (see
    /// [`oximond::jit_ir::trace::pinned_native_identity`]), adding it on
    /// first use. A side trace starts from the tree's lanes and may add more
    /// after them (see [`oximond::jit_ir::trace::plan_tree`]).
    fn pin(&mut self, identity: u64) -> Result<u32, TraceAbortReason> {
        let position = match self.pinned.iter().position(|known| *known == identity) {
            Some(position) => position,
            None if self.pinned.len() >= MAX_TRACE_PINNED => {
                return Err(limit("pinned"));
            }
            None => {
                self.pinned.push(identity);
                self.pinned.len() - 1
            }
        };
        Ok(u32::try_from(position).expect("pinned closures fit u32"))
    }

    /// Takes the native or builtin function whose use aborted the recording,
    /// if one did.
    pub(crate) fn take_abort_callable(&mut self) -> Option<Value> {
        self.abort_callable.get_mut().take()
    }

    /// Whether the recording that aborted met a table set it would have kept
    /// had the table been a dense list (see [`Self::set_waiting`]).
    pub(crate) const fn waits_for_table(&self) -> bool {
        self.set_waiting
    }

    /// Takes the name of the metamethod whose access aborted the recording,
    /// if one did (see [`Self::abort_metamethod`]).
    pub(crate) fn take_abort_metamethod(&mut self) -> Option<&'static str> {
        self.abort_metamethod.take()
    }

    /// Remembers `value`, when it is a native or builtin function, as what
    /// the recording could not take. This is the abort path only: nothing
    /// here runs while a trace records successfully, and nothing reaches
    /// compiled code. The first function met is the one kept.
    fn note_callable(&self, value: &Value) {
        if matches!(value, Value::NativeFunction(_) | Value::BuiltinFunction(_)) {
            let mut held = self.abort_callable.borrow_mut();
            if held.is_none() {
                *held = Some(value.clone());
            }
        }
    }

    /// Notes the function an instruction that failed to record reads, for
    /// the instructions that only move or call one.
    fn note_failed_callable(&self, at: &HookPoint<'_>, instruction: Instruction) {
        let register = match instruction.opcode() {
            Ok(OpCode::Move) => instruction.b(),
            Ok(OpCode::ForInCall) => instruction.a(),
            _ => return,
        };
        if let Some(value) = at.registers.get(usize::from(register)) {
            self.note_callable(value);
        }
    }

    /// Records the call, and when the call of a native or builtin function
    /// is what the recorder cannot take, remembers the function for the
    /// statistics.
    fn record_call(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        let depth = self.frames.len();
        let region = self.open_region_for(at, instruction);
        // The open results the call takes are not in the registers: the
        // recorder reads the argument values from there, so put them where
        // the call's arguments would be.
        let flow = if instruction.opcode() == Ok(OpCode::CallOpen) && !at.open.is_empty() {
            let first = usize::from(instruction.a()) + 1 + usize::from(instruction.b());
            let mut registers = at.registers.to_vec();
            if registers.len() < first + at.open.len() {
                registers.resize(first + at.open.len(), Value::Nil);
            }
            registers[first..first + at.open.len()].clone_from_slice(at.open);
            let seen = HookPoint {
                chunk: at.chunk,
                depth: at.depth,
                top_level: at.top_level,
                pc: at.pc,
                registers: &registers,
                open: at.open,
                coroutine: at.coroutine,
                hooks_active: at.hooks_active,
                source: at.source,
                surroundings: at.surroundings,
            };
            if self
                .open
                .as_ref()
                .is_some_and(|open| open.len() != at.open.len())
            {
                Err(TraceAbortReason::Divergence)
            } else {
                self.record_call_inner(&seen, instruction)
            }
        } else {
            self.record_call_inner(at, instruction)
        };
        if region {
            self.reexec = None;
            self.open_region = false;
            self.site_override = None;
        }
        // The consumer is done with its note, taken or not.
        self.drop_hoisted(depth, at.pc);
        let callee = match instruction.opcode() {
            Ok(OpCode::CallOne) => instruction.b(),
            _ => instruction.a(),
        };
        if matches!(flow, Err(TraceAbortReason::UnsupportedOperand)) {
            // The function this call could not take is the one to blame,
            // not one an earlier field read met. A protected call blames the
            // native it was to call.
            let mut blamed = at.registers.get(usize::from(callee));
            if matches!(
                blamed,
                Some(Value::BuiltinFunction(
                    BuiltinFunction::ProtectedCall | BuiltinFunction::ExtendedProtectedCall
                ))
            ) && let Some(inner @ (Value::NativeFunction(_) | Value::BuiltinFunction(_))) =
                at.registers.get(usize::from(callee) + 1)
            {
                blamed = Some(inner);
            }
            if let Some(value) = blamed
                && matches!(value, Value::NativeFunction(_) | Value::BuiltinFunction(_))
            {
                *self.abort_callable.get_mut() = Some(value.clone());
            }
        } else if flow.is_ok() {
            self.abort_callable.get_mut().take();
        }
        flow
    }

    /// Opens the region of a call that takes open results of a native (a
    /// `CallOpen`): while it is recorded, every guard leaves through the
    /// snapshot from before the call that made the results, so the
    /// interpreter makes both calls again. That is exact only where nothing
    /// observable happened since, so a recording that stored, or advanced
    /// the generator, in between has no region and the call keeps the guard
    /// it had before the results were made. Returns whether a region is
    /// open.
    fn open_region_for(&mut self, at: &HookPoint<'_>, instruction: Instruction) -> bool {
        if instruction.opcode() != Ok(OpCode::CallOpen) || self.reexec.is_some() {
            return false;
        }
        let Some((snapshot, from, producer)) = self
            .hoisted
            .iter()
            .rev()
            .find(|note| note.consumer == (self.frames.len(), at.pc))
            .and_then(|note| note.start)
        else {
            return false;
        };
        let callee = at.registers.get(usize::from(instruction.a()));
        let takes_guards = match callee {
            Some(Value::BuiltinFunction(
                BuiltinFunction::ToString | BuiltinFunction::StringFormat,
            )) => true,
            Some(Value::NativeFunction(function)) => {
                !Self::native_kind(at, *function).is_some_and(Self::is_symbolic_kind)
            }
            _ => false,
        };
        if !takes_guards
            || self.open.is_none()
            || self.ops[from..].iter().any(|op| op.kind.commits_effect())
        {
            return false;
        }
        self.reexec = Some(snapshot);
        self.open_region = true;
        self.site_override = Some(producer);
        true
    }

    /// A call of a Lua closure, with fixed or open arguments and results. The
    /// interpreter makes the call and the recorder follows it into the
    /// callee, so a trace runs the callee's operations in place.
    ///
    /// The callee is pinned by identity: a guard compares the live callee
    /// with the recorded closure before the call, with a `Before` snapshot,
    /// so any other callee runs in the interpreter. The identity (not the
    /// chunk) matters, since two closures of one prototype capture other
    /// cells, unless the callee takes turns between closures: the guard
    /// then checks the prototype, and the cells are the live closure's (see
    /// the module docs). A fresh activation holds its arguments and nil,
    /// which becomes the callee's symbolic registers.
    ///
    /// A callee with varargs is inlined too. The values past its parameters
    /// are symbolic like any other (the call site has a fixed argument
    /// count), and `Vararg` reads them. The interpreter's frame for such a
    /// callee has storage the snapshots cannot describe, so the whole call
    /// is a re-execution region (see [`Recorder::reexec`]): every guard in
    /// it leaves through the snapshot before the call. `CallVararg`
    /// forwards the running frame's varargs, and `TailCallVararg` replaces
    /// the running frame, which only a re-execution region may do since the
    /// restored stack would have to show the tail call.
    ///
    /// A `TailCall` replaces the running inlined frame: the callee runs where
    /// it was, returns to the same caller, and the rebuilt frame of a snapshot
    /// inside it counts the tail calls the interpreter would (see
    /// [`InlinedFrame::tail_calls`]). A tail call of a native function is a
    /// call with open results that the `ReturnOpen` after it returns.
    ///
    /// Open results (a call whose count is `C = 0`) and the call or return
    /// that takes them (`CallOpen`, `TailCallOpen`, `ReturnOpen`) are followed
    /// where the recording knows how many values there are: the callee is a
    /// native with a numeric form, or an inlined closure whose return the
    /// recording has seen. Along the recorded path that count is fixed, since
    /// the guards pin the callee and every branch. The interpreter keeps the
    /// values in the frame's open-results slot, which no snapshot rebuilds,
    /// so no guard may leave between the call that makes them and the
    /// instruction that takes them. The consumer's callee is therefore
    /// guarded before the producer starts (see [`Recorder::hoist_callee`]).
    ///
    /// Anything else aborts: a native without a numeric form or with varargs
    /// or the wrong argument types, a callee that is already being inlined
    /// (recursion), and a callee that would nest too deeply.
    #[allow(clippy::too_many_lines)]
    fn record_call_inner(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let opcode = instruction.opcode();
        let tail_opcode = matches!(
            opcode,
            Ok(OpCode::TailCall | OpCode::TailCallVararg | OpCode::TailCallOpen)
        );
        let with_varargs = matches!(opcode, Ok(OpCode::CallVararg | OpCode::TailCallVararg));
        let consumes_open = matches!(opcode, Ok(OpCode::CallOpen | OpCode::TailCallOpen));
        let (callee, first_argument, fixed, mut base, mut results) =
            if let Ok(OpCode::CallOne) = opcode {
                // `CallOne`: destination a, callee b, argument c.
                (
                    usize::from(instruction.b()),
                    usize::from(instruction.c()),
                    1,
                    usize::from(instruction.a()),
                    1,
                )
            } else {
                let base = usize::from(instruction.a());
                (
                    base,
                    base + 1,
                    usize::from(instruction.b()),
                    base,
                    usize::from(instruction.c()),
                )
            };
        self.check_register(callee)?;
        if fixed > 0 {
            self.check_register(first_argument + fixed - 1)?;
        }
        let (callee_operand, ty) = self.read(at.registers, callee)?;
        // A table called through its metatable's raw `__call`, a Lua closure:
        // the closure runs with the table as its first argument.
        let table_call = if ty == TraceType::Table {
            let runtime = at
                .surroundings
                .ok_or(TraceAbortReason::UnsupportedOperand)?
                .runtime;
            let handler = match runtime.raw_metatable(&at.registers[callee]) {
                Ok(Some(metatable)) => {
                    runtime.table_get_string_ref(metatable, call_metamethod_key())
                }
                _ => None,
            };
            match handler {
                Some(Value::Closure(handle)) => Some(*handle),
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            }
        } else {
            None
        };
        if table_call.is_some() && (with_varargs || consumes_open || tail_opcode || results == 0) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // `pairs (t)` of a table with a Lua closure as `__pairs` calls it.
        let pairs_call = if with_varargs || consumes_open || tail_opcode {
            None
        } else {
            self.pairs_handler(at, (callee, first_argument, fixed))
        };
        if table_call.is_none() && ty != TraceType::Closure {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A call of the function's own chunk in the root frame of a function
        // trace: a self call, or something the trace cannot hold.
        if self.frames.is_empty()
            && let Some(function) = &self.function
            && let Value::Closure(handle) = at.registers[callee]
            && at
                .surroundings
                .and_then(|around| around.runtime.closure_call_parts(handle))
                .is_some_and(|parts| Arc::ptr_eq(parts.chunk, &self.chunk))
        {
            // Open results feed a self call only from a self call: their
            // value is a native one, and the call that made them checked
            // the depth this one would.
            if handle != function.closure || with_varargs || (consumes_open && !self.open_from_self)
            {
                return Err(TraceAbortReason::SelfCall);
            }
            if tail_opcode {
                return self.record_self_tail_call(at, instruction, callee_operand);
            }
            return self.record_self_call(at, instruction, callee_operand);
        }
        // Open results of a self call go to a self call only.
        if consumes_open && self.open_from_self {
            return Err(TraceAbortReason::SelfCall);
        }
        let native = matches!(
            at.registers[callee],
            Value::NativeFunction(_) | Value::BuiltinFunction(_) | Value::GmatchIterator(_)
        );
        // A tail call returns what its callee returns, to the running
        // frame's caller. A native does not replace the frame: it is a call
        // with open results, and the `ReturnOpen` after it returns them.
        let tail = tail_opcode && !native;
        if tail && self.frames.is_empty() {
            // The root frame's tail call: the callee replaces the root frame.
            return self.record_root_tail_call(at, instruction, callee_operand);
        }
        if tail {
            let frame = self
                .frames
                .last()
                .ok_or(TraceAbortReason::UnsupportedOperand)?;
            (results, base) = (frame.results, frame.result_base);
            if with_varargs && self.reexec.is_none() {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        } else if tail_opcode {
            results = 0;
            let returns = at.chunk.code.get(pc + 1).copied();
            if !returns.is_some_and(|next| {
                next.opcode() == Ok(OpCode::ReturnOpen) && next.a() == instruction.a()
            }) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        }
        if !tail && results > 0 {
            self.check_register(base + results - 1)?;
        }
        if results == 0 && !tail {
            // Open results: this call is the producer. One that takes open
            // results itself would guard its consumer's callee with values
            // pending in the frame, so a chain of calls is no producer. A
            // return needs no guard.
            if consumes_open {
                let returns = at.chunk.code.get(pc + 1).copied();
                if returns.is_none_or(|next| next.opcode() != Ok(OpCode::ReturnOpen)) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
            } else {
                self.hoist_callee(at)?;
            }
        }
        if let Value::NativeFunction(function) = at.registers[callee]
            && let Some(flow) = self.record_special_native(
                at,
                (callee_operand, function),
                (first_argument, fixed),
                (base, results),
                (with_varargs, consumes_open, tail_opcode),
            )
        {
            return flow;
        }
        // The arguments: the fixed ones from registers, the open results a
        // `CallOpen` takes, then the running frame's varargs if the call
        // forwards them.
        let mut arguments = Vec::with_capacity(fixed);
        // A call the VM can answer reads its string constants too, and so
        // does a call the trace inlines.
        // `pcall` and `xpcall` call their first argument.
        let protection = match &at.registers[callee] {
            Value::BuiltinFunction(BuiltinFunction::ProtectedCall) if !consumes_open => Some(1),
            Value::BuiltinFunction(BuiltinFunction::ExtendedProtectedCall) if !consumes_open => {
                Some(2)
            }
            _ => None,
        };
        let probe = if protection.is_some() && fixed > 0 {
            first_argument
        } else {
            callee
        };
        // A call in a region may guard, so it reads and records as a call
        // with fixed arguments does.
        let open_unguarded = consumes_open && !self.open_region;
        let host = !with_varargs
            && !open_unguarded
            && (Self::is_host_callee(at, probe)
                || matches!(at.registers[probe], Value::Closure(_)));
        // The arguments of a native `pcall` and `xpcall` make are read by
        // the native's own tier, which knows which ones may be constants.
        let read_count = match (protection, &at.registers[probe]) {
            (Some(leading), Value::NativeFunction(_) | Value::BuiltinFunction(_)) => {
                fixed.min(leading)
            }
            _ => fixed,
        };
        for index in 0..read_count {
            arguments.push(if host {
                self.host_operand(pc, at.registers, first_argument + index)?
            } else {
                self.read(at.registers, first_argument + index)?
            });
        }
        let mut chars = Vec::new();
        if consumes_open {
            for value in self.open.take().ok_or(TraceAbortReason::Divergence)? {
                match value {
                    Open::Op(op) => {
                        let ty = self.ops[op as usize]
                            .ty
                            .expect("an operand operation defines a value");
                        arguments.push((Operand::Op(op), ty));
                    }
                    Open::Char(byte) => chars.push(byte),
                    Open::Root(slot) => arguments.push(self.root_operand(pc, slot)),
                }
            }
        }
        if !chars.is_empty() {
            // The one-byte string `sub` made exists nowhere: only
            // `string.byte` of it, as the sole argument, is a call the trace
            // can hold.
            let runtime = at
                .surroundings
                .ok_or(TraceAbortReason::UnsupportedOperand)?
                .runtime;
            return match (&at.registers[callee], chars.as_slice()) {
                (Value::NativeFunction(function), [byte])
                    if fixed == 0
                        && arguments.is_empty()
                        && !tail_opcode
                        && runtime.direct_native_kind(*function)
                            == Some(DirectNativeKind::StringByte) =>
                {
                    self.record_byte_of_char(at, callee_operand, *function, *byte, (base, results))
                }
                _ => Err(TraceAbortReason::UnsupportedOperand),
            };
        }
        let mut callee_operand = callee_operand;
        let mut protected = Protected::No;
        let handle = if let Some(handler) = pairs_call {
            // The interpreter calls the handler with the table alone, with
            // the results `pairs` would have had.
            let &(table, table_ty) = arguments.first().ok_or(TraceAbortReason::Divergence)?;
            self.guard_pairs_handler(
                pc,
                callee_operand,
                (table, table_ty),
                &at.registers[first_argument],
                handler,
            )?;
            arguments.truncate(1);
            let lane = self.pin(handler.native_identity())?;
            callee_operand = Operand::Op(self.emit(
                pc,
                RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
                Some(TraceType::Closure),
            ));
            handler
        } else if let Some(handler) = table_call {
            // The helper checks that the callee's `__call` is the pinned
            // closure, and leaves through the call's `Before` snapshot
            // otherwise. The closure then runs as the callee, with the table
            // as its first argument.
            let identity = handler.native_identity();
            let lane = self.pin(identity)?;
            let handler_id = self.scalar(pc, TraceScalar::Integer(identity.cast_signed()));
            self.emit_helper_call(
                pc,
                HelperId::CALL_HANDLER,
                &[
                    (callee_operand, ty),
                    (Operand::Op(handler_id), TraceType::Integer),
                ],
            );
            arguments.insert(0, (callee_operand, ty));
            callee_operand = Operand::Op(self.emit(
                pc,
                RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
                Some(TraceType::Closure),
            ));
            handler
        } else {
            match at.registers[callee] {
                Value::Closure(handle) => handle,
                Value::BuiltinFunction(
                    builtin @ (BuiltinFunction::ProtectedCall
                    | BuiltinFunction::ExtendedProtectedCall),
                ) if protection.is_some() => {
                    match self.record_protected_call(
                        at,
                        (builtin, callee_operand),
                        &mut arguments,
                        (first_argument, fixed, with_varargs),
                        (base, results),
                    )? {
                        ProtectedTarget::Done(flow) => return Ok(flow),
                        ProtectedTarget::Closure {
                            handle,
                            target,
                            protected: kind,
                        } => {
                            callee_operand = target;
                            protected = kind;
                            handle
                        }
                    }
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineResume)
                    if !with_varargs && !consumes_open && !tail_opcode && !open_unguarded =>
                {
                    return self.record_resume(at, callee_operand, &arguments, (base, results));
                }
                Value::BuiltinFunction(BuiltinFunction::CoroutineYield) => {
                    return Err(self.record_yield(at));
                }
                Value::CoroutineWrapper(_)
                    if !with_varargs && !consumes_open && !tail_opcode && !open_unguarded =>
                {
                    return self.record_wrapped_resume(
                        at,
                        callee_operand,
                        &arguments,
                        (base, results),
                    );
                }
                Value::NativeFunction(function) if !with_varargs => {
                    return self.record_native_call(
                        at,
                        callee_operand,
                        function,
                        (&arguments, first_argument),
                        (base, results),
                        open_unguarded,
                    );
                }
                Value::BuiltinFunction(
                    builtin @ (BuiltinFunction::ToString | BuiltinFunction::StringFormat),
                ) if !with_varargs && !open_unguarded => {
                    return self.record_host_builtin(
                        at,
                        builtin,
                        callee_operand,
                        &arguments,
                        (base, results),
                    );
                }
                Value::BuiltinFunction(BuiltinFunction::StringGsub)
                    if !with_varargs && !open_unguarded && !tail_opcode && results > 0 =>
                {
                    return self.record_gsub(
                        at,
                        callee_operand,
                        (&arguments, first_argument),
                        (base, results),
                    );
                }
                Value::GmatchIterator(iterator)
                    if !with_varargs && !consumes_open && !tail_opcode && results > 0 =>
                {
                    return self.record_gmatch_step(at, callee_operand, iterator, (base, results));
                }
                Value::BuiltinFunction(BuiltinFunction::IPairsIterator)
                    if !with_varargs && !consumes_open && !tail_opcode && results > 0 =>
                {
                    return self.record_ipairs_step(
                        at,
                        callee_operand,
                        (first_argument, fixed),
                        (base, results),
                    );
                }
                Value::BuiltinFunction(BuiltinFunction::Next)
                    if !with_varargs && !consumes_open && !tail_opcode =>
                {
                    if std::mem::take(&mut self.boxed_call) {
                        return self.record_boxed_next(at, callee_operand, (base, results));
                    }
                    return self.record_next_step(
                        at,
                        callee_operand,
                        (&arguments, first_argument),
                        (base, results),
                    );
                }
                Value::BuiltinFunction(BuiltinFunction::Pairs)
                    if !with_varargs && !consumes_open && !tail_opcode =>
                {
                    return self.record_pairs(
                        at,
                        callee_operand,
                        (&arguments, first_argument),
                        (base, results),
                    );
                }
                Value::BuiltinFunction(BuiltinFunction::IPairs)
                    if !with_varargs && !consumes_open && !tail_opcode =>
                {
                    return self.record_ipairs(at, callee_operand, &arguments, (base, results));
                }
                Value::NativeFunction(_) | Value::BuiltinFunction(_) | Value::GmatchIterator(_) => {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                _ => return Err(TraceAbortReason::Divergence),
            }
        };
        let surroundings = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let chunk = surroundings
            .runtime
            .closure_call_parts(handle)
            .map(|parts| Arc::clone(parts.chunk))
            .ok_or(TraceAbortReason::Divergence)?;
        if chunk.arg_register.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if with_varargs {
            for value in self.current_varargs(at)? {
                let Sym::Op(op) = value else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                let ty = self.ops[op as usize]
                    .ty
                    .expect("an operand operation defines a value");
                arguments.push((Operand::Op(op), ty));
            }
        }
        // A tail call ends the running frame, so what recursion and depth
        // see is the stack without it.
        let outer = self.frames.len() - usize::from(tail);
        // A callee on the stack of inlined calls is recursion. The trace has
        // no way to bound it.
        if Arc::ptr_eq(&chunk, &self.chunk)
            || self
                .activation
                .as_ref()
                .is_some_and(|activation| Arc::ptr_eq(&activation.chunk, &chunk))
            || self.frames[..outer]
                .iter()
                .any(|frame| Arc::ptr_eq(&frame.chunk, &chunk))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // `MAX_TRACE_FRAME_DEPTH` counts the root frame.
        if outer + 2 > MAX_TRACE_FRAME_DEPTH || self.inlined.len() >= MAX_TRACE_INLINED_CALLS {
            return Err(limit("inlined calls/depth"));
        }
        let lane = self.pin(handle.native_identity())?;
        // A call site whose callee takes turns between closures (of one
        // prototype, or of several) is guarded on the prototype, and the
        // callee's cells are read and written through the live closure.
        let dynamic_call = !tail
            && !consumes_open
            && results > 0
            && protected == Protected::No
            && pairs_call.is_none()
            && table_call.is_none()
            && !chunk.is_vararg
            && (self.callee_varies(callee_operand) || self.poly_call(pc))
            && self.may_use_dynamic_cells()
            && !self.shares_window_cell(at, handle, &chunk);
        // The call that takes open results has no snapshot to leave through
        // (the interpreter's open-results slot holds them): its callee was
        // guarded before the call that made them.
        let before = if consumes_open {
            if !self.take_hoisted(callee_operand) {
                return Err(TraceAbortReason::Divergence);
            }
            None
        } else if dynamic_call {
            Some(self.guard_proto(pc, callee_operand, lane))
        } else if self.frames.is_empty() || chunk.is_vararg {
            // The callee's guards may leave through this snapshot, or a
            // region starts over at it: the call keeps its own.
            Some(self.guard_callee_always(pc, callee_operand, lane))
        } else {
            self.guard_callee(pc, callee_operand, lane)
        };
        // The frame's closure, which its snapshots carry.
        let dynamic = dynamic_call.then(|| {
            self.dynamic_cells = true;
            self.emit(pc, RecKind::Move(callee_operand), Some(TraceType::Closure))
        });
        if self.frames.is_empty()
            && let Some(before) = before
        {
            self.call_site = Some((before, self.stores));
            self.call_ops = self.ops.len();
            self.call_guest = self.guest;
            self.call_cell_writes = self.cell_writes;
        }
        // A callee with varargs starts a region that re-executes the call.
        let opens_region = chunk.is_vararg && self.reexec.is_none();
        if opens_region {
            self.reexec = Some(before.ok_or(TraceAbortReason::UnsupportedOperand)?);
        }

        let register_count = usize::from(chunk.register_count);
        let parameters = usize::from(chunk.parameter_count).min(register_count);
        let passed = parameters.min(arguments.len());
        let mut regs = Vec::with_capacity(register_count);
        for (source, ty) in &arguments[..passed] {
            let value = self.emit(pc, RecKind::Move(*source), Some(*ty));
            regs.push(Sym::Op(value));
        }
        let mut varargs = Vec::new();
        if chunk.is_vararg {
            for (source, ty) in arguments.iter().skip(parameters) {
                let value = self.emit(pc, RecKind::Move(*source), Some(*ty));
                varargs.push(Sym::Op(value));
            }
        }
        if passed < register_count {
            // A missing argument or an ordinary local starts as nil.
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            regs.resize(register_count, Sym::Op(nil));
        }

        let mut after = Pending::Nothing;
        let (parent, call_pc, owns_reexec, tail_calls, tail_site, entry) = if tail {
            // The new frame takes the place of the running one: the same
            // caller, the same result window, the same region, and the same
            // protected call.
            let mut replaced = self.frames.pop().expect("a tail call has a frame");
            if replaced.handler {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            protected = replaced.protected;
            after = std::mem::take(&mut replaced.after);
            (
                FrameId(self.frame_id()),
                replaced.call_pc,
                replaced.owns_reexec || opens_region,
                replaced.tail_calls.saturating_add(1),
                Some((replaced.closure, pc)),
                // A metamethod's tail call returns where the metamethod did.
                replaced.entry,
            )
        } else {
            // The interpreter marks the call as the caller's last instruction.
            self.set_prev(PrevPc::At(pc));
            (
                FrameId(self.frame_id()),
                pc,
                opens_region,
                0,
                None,
                Entry::Call,
            )
        };
        self.inlined.push(TraceFrameInfo {
            register_count: u16::try_from(register_count + usize::from(dynamic.is_some()))
                .expect("register count fits u16"),
            code_len: chunk.code.len(),
            inlined: Some(InlinedFrame {
                closure: lane,
                parent,
                call_pc,
                result_base: u8::try_from(base).expect("register index fits u8"),
                results: u8::try_from(results).expect("result count fits u8"),
                reexec: owns_reexec,
                tail_calls,
                tail_site,
                protected,
                entry,
                dynamic: dynamic.is_some(),
            }),
            activation: None,
        });
        self.frames.push(Inlined {
            chunk,
            closure: lane,
            frame: u32::try_from(self.inlined.len()).expect("frame count fits u32"),
            regs,
            prev: PrevPc::Unset,
            call_pc,
            result_base: base,
            results,
            varargs,
            owns_reexec,
            tail_calls,
            protected,
            handler: false,
            entry,
            after,
            dynamic,
        });
        Ok(if tail { Flow::Replace } else { Flow::Enter })
    }

    /// A `TailCall` of a Lua closure in the root frame. The interpreter
    /// replaces the root frame in place: the callee runs in it from pc 0 with
    /// the arguments and nil in its registers, the frame counts the tail call
    /// and is named after it. The recording follows the callee in the root
    /// frame as a later activation of it (see [`Activation`]), whose upvalues
    /// and globals read through the pinned callee.
    ///
    /// A tail call back into the closure the recording started with, in the
    /// root chunk, comes back to the loop header (pc 0): it closes a loop
    /// through tail calls, whose callee guard compares the callee with the
    /// closure the root frame runs at entry ([`ExternalSlot::RootClosure`]).
    /// Self tail recursion is that loop with no activation between.
    ///
    /// The frame's tail call count is a window slot
    /// ([`ExternalSlot::TailCalls`]) each such call adds one to, so every
    /// exit restores the count and the name the interpreter would have.
    ///
    /// Refused ([`TraceAbortReason::RootTailCall`]): a side recording or a
    /// loop whose header is not the chunk's first instruction, vararg or open
    /// arguments, a vararg callee, a root frame with captured locals, a
    /// callee with more registers than the root's window, and another closure
    /// of the root chunk.
    #[allow(clippy::too_many_lines)]
    fn record_root_tail_call(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        callee: Operand,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if self.function.is_some() {
            return Err(TraceAbortReason::SelfCall);
        }
        if self.side.is_some()
            || self.header != 0
            || instruction.opcode() != Ok(OpCode::TailCall)
            || !self.root_captured.is_empty()
            || self.loop_tail.is_some()
        {
            return Err(TraceAbortReason::RootTailCall);
        }
        let base = usize::from(instruction.a());
        let fixed = usize::from(instruction.b());
        let Value::Closure(handle) = at.registers[base] else {
            return Err(TraceAbortReason::Divergence);
        };
        let chunk = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime
            .closure_call_parts(handle)
            .map(|parts| Arc::clone(parts.chunk))
            .ok_or(TraceAbortReason::Divergence)?;
        if chunk.is_vararg
            || chunk.arg_register.is_some()
            || usize::from(chunk.register_count) > self.register_count
        {
            return Err(TraceAbortReason::RootTailCall);
        }
        let closes = Arc::ptr_eq(&chunk, &self.chunk);
        if closes && self.home_closure != Some(handle) {
            return Err(TraceAbortReason::RootTailCall);
        }
        if !closes && self.inlined.len() >= MAX_TRACE_INLINED_CALLS {
            return Err(limit("inlined calls"));
        }
        let mut arguments = Vec::with_capacity(fixed);
        for index in 0..fixed {
            arguments.push(self.read(at.registers, base + 1 + index)?);
        }
        let site = TailSite {
            closure: self
                .activation
                .as_ref()
                .map(|activation| activation.closure),
            pc,
        };
        // The callee guard, with a snapshot before the call: another callee
        // runs in the interpreter.
        let lane = if closes {
            let home = self.root_closure_register()?;
            let (home, _) = self.read_window(home)?;
            self.guard_same(pc, callee, home);
            None
        } else {
            let lane = self.pin(handle.native_identity())?;
            self.guard_callee(pc, callee, lane);
            Some(lane)
        };
        // The frame counts the call.
        let count = self.tail_count_register()?;
        let (counted, _) = self.read_window(count)?;
        let one = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Integer(1)),
            Some(TraceType::Integer),
        );
        let next = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Add, counted, Operand::Op(one)),
            Some(TraceType::Integer),
        );
        self.write_external(count, next);
        // A fresh activation: the arguments, then nil, in every guest
        // register of the window. The ones past the callee's registers are
        // not the frame's and restore nothing.
        let parameters = usize::from(chunk.parameter_count).min(usize::from(chunk.register_count));
        let passed = parameters.min(arguments.len());
        let mut regs = Vec::with_capacity(self.register_count);
        for (source, ty) in &arguments[..passed] {
            let value = self.emit(pc, RecKind::Move(*source), Some(*ty));
            regs.push(Sym::Op(value));
        }
        if passed < self.register_count {
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            regs.resize(self.register_count, Sym::Op(nil));
        }
        for (register, sym) in regs.into_iter().enumerate() {
            self.regs[register] = sym;
            self.written[register] = true;
        }
        // The interpreter keeps no previous pc across a tail call.
        self.prev = PrevPc::Unset;
        match lane {
            None => {
                self.activation = None;
                self.loop_tail = Some(site);
            }
            Some(lane) => {
                self.inlined.push(TraceFrameInfo {
                    // The root's window, known when the trace closes.
                    register_count: 0,
                    code_len: chunk.code.len(),
                    inlined: None,
                    activation: Some(RootActivation {
                        closure: lane,
                        site,
                    }),
                });
                let frame = u32::try_from(self.inlined.len()).expect("frame count fits u32");
                self.activation_chunks.push((frame, Arc::clone(&chunk)));
                self.entered_activation = true;
                self.activation = Some(Activation {
                    chunk,
                    closure: lane,
                    frame,
                });
            }
        }
        Ok(Flow::ReplaceRoot)
    }

    /// The chunks a root tail-call cycle recorded so far lands in, besides the
    /// root's own.
    pub(crate) fn tail_call_chunks(&self) -> impl Iterator<Item = &Arc<Chunk>> {
        self.activation_chunks.iter().map(|(_, chunk)| chunk)
    }

    /// Whether the step recorded since the last call entered a new root
    /// activation (see [`Self::tail_call_chunks`]), and forgets it.
    pub(crate) fn take_entered_activation(&mut self) -> bool {
        std::mem::take(&mut self.entered_activation)
    }

    /// Sets up a function-entry recording at its first hook: the parameters'
    /// types are the entry's, and the rest of a fresh frame is nil. A side
    /// of a function trace takes the root's parameter types.
    fn start_function(&mut self) -> Result<(), TraceAbortReason> {
        let Some(function) = &self.function else {
            return Ok(());
        };
        if self.chunk.is_vararg || self.chunk.arg_register.is_some() {
            return Err(TraceAbortReason::SelfCall);
        }
        let parameters = usize::from(self.chunk.parameter_count).min(self.register_count);
        if self.side.is_some() {
            return if function.parameters.len() == parameters {
                Ok(())
            } else {
                Err(TraceAbortReason::Divergence)
            };
        }
        if !self.root_captured.is_empty() {
            return Err(TraceAbortReason::SelfCall);
        }
        let types = self.entry[..parameters]
            .iter()
            .map(|kind| kind.trace_type())
            .collect::<Option<Vec<_>>>()
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        if self.entry[parameters..self.register_count]
            .iter()
            .any(|kind| *kind != Kind::Nil)
        {
            return Err(TraceAbortReason::Divergence);
        }
        if let Some(function) = &mut self.function {
            function.parameters = types;
        }
        // The registers past the parameters hold nil in every activation:
        // a constant, not an entry input.
        if parameters < self.register_count {
            let nil = self.emit(0, RecKind::Constant(TraceScalar::Nil), Some(TraceType::Nil));
            for register in parameters..self.register_count {
                self.regs[register] = Sym::Op(nil);
            }
        }
        Ok(())
    }

    /// A self call of a function trace's closure in its root frame: the
    /// callee's parameters, with the entry's types, and the call's `Before`
    /// snapshot. The recorder then sleeps until the interpreter is back at
    /// the instruction after the call (see [`Self::wake`]).
    fn record_self_call(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        callee: Operand,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let (first_argument, fixed, base, results) = match instruction.opcode() {
            Ok(OpCode::CallOne) => (
                usize::from(instruction.c()),
                1,
                usize::from(instruction.a()),
                1,
            ),
            Ok(OpCode::Call | OpCode::CallOpen) => {
                let base = usize::from(instruction.a());
                (
                    base + 1,
                    usize::from(instruction.b()),
                    base,
                    usize::from(instruction.c()),
                )
            }
            _ => return Err(TraceAbortReason::SelfCall),
        };
        // Open results go to the self call or tail call that takes them,
        // after at most some moves.
        if results > 0 {
            self.check_register(base + results - 1)?;
        }
        let register = self.self_callee_register(callee)?;
        let open = self.take_self_open(instruction)?;
        let args = self.self_call_arguments(at, register, (first_argument, fixed), &open)?;
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        let op = self.emit(pc, RecKind::SelfCall { args, snapshot }, None);
        self.set_prev(PrevPc::At(pc));
        self.sleeping = Some(SelfCallWait {
            op,
            depth: at.depth,
            pc,
            base,
            results,
            returned: None,
        });
        Ok(Flow::Sleep)
    }

    /// The callee register and arguments of a self call or self tail call at
    /// `at.pc`: the callee comes from a cell or global the function leaves
    /// alone, which the entry checks holds the frame's closure, and the
    /// arguments (nil past the `fixed` ones from `first`) have the entry's
    /// parameter types, so a new activation never mismatches.
    fn self_call_arguments(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
        (first, fixed): (usize, usize),
        open: &[u32],
    ) -> Result<Vec<Operand>, TraceAbortReason> {
        let register = register_u16(register);
        if !self.callee_registers.contains(&register) {
            self.callee_registers.push(register);
        }
        let parameters = self
            .function
            .as_ref()
            .map(|function| function.parameters.clone())
            .unwrap_or_default();
        let mut args = Vec::with_capacity(parameters.len());
        for (index, wanted) in parameters.iter().enumerate() {
            let (operand, ty) = if index < fixed {
                self.read(at.registers, first + index)?
            } else if let Some(op) = open.get(index - fixed) {
                let ty = self.ops[*op as usize]
                    .ty
                    .ok_or(TraceAbortReason::Divergence)?;
                (Operand::Op(*op), ty)
            } else {
                let nil = self.emit(
                    at.pc,
                    RecKind::Constant(TraceScalar::Nil),
                    Some(TraceType::Nil),
                );
                (Operand::Op(nil), TraceType::Nil)
            };
            // A table this trace made has a lane but no row in the callee's
            // run: its fields are never read natively, so it cannot cross a
            // self call. A table a self call returned is the same: the
            // callee made it, and nothing filled its row for this run.
            if let Operand::Op(op) = operand
                && (self.is_made_table(op) || self.is_self_call_table(op))
            {
                return Err(TraceAbortReason::SelfCall);
            }
            // A new activation must take the entry's types.
            if ty != *wanted {
                return Err(TraceAbortReason::TypeUnstable);
            }
            args.push(operand);
        }
        Ok(args)
    }

    /// The open results of a self call that a `CallOpen` or `TailCallOpen`
    /// takes, as operations; none for another call.
    fn take_self_open(&mut self, instruction: Instruction) -> Result<Vec<u32>, TraceAbortReason> {
        if !matches!(
            instruction.opcode(),
            Ok(OpCode::CallOpen | OpCode::TailCallOpen)
        ) {
            return Ok(Vec::new());
        }
        if !std::mem::take(&mut self.open_from_self) {
            return Err(TraceAbortReason::SelfCall);
        }
        self.open
            .take()
            .ok_or(TraceAbortReason::Divergence)?
            .into_iter()
            .map(|open| match open {
                Open::Op(op) => Ok(op),
                Open::Char(_) | Open::Root(_) => Err(TraceAbortReason::SelfCall),
            })
            .collect()
    }

    /// The window register a self call's callee comes from: a cell or global
    /// of the window the function never writes.
    fn self_callee_register(&self, callee: Operand) -> Result<usize, TraceAbortReason> {
        let Operand::Carried(register) = self.origin(callee) else {
            return Err(TraceAbortReason::SelfCall);
        };
        let slot = register
            .checked_sub(self.register_count)
            .and_then(|slot| self.externals.get(slot));
        if !matches!(
            slot,
            Some(ExternalSlot::Upvalue(_) | ExternalSlot::Global(_))
        ) || self.written[register]
        {
            return Err(TraceAbortReason::SelfCall);
        }
        Ok(register)
    }

    /// A `TailCall` of the function's own closure in its root frame (`return
    /// f(args)`): the frame starts over with the arguments, as the
    /// interpreter's in-place tail call does, which closes the trace with a
    /// [`TraceTerminator::TailCall`]. The activation's tail call count is a
    /// window slot of its own ([`ExternalSlot::TailCalls`]).
    fn record_self_tail_call(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        callee: Operand,
    ) -> Result<Flow, TraceAbortReason> {
        if !matches!(
            instruction.opcode(),
            Ok(OpCode::TailCall | OpCode::TailCallOpen)
        ) {
            return Err(TraceAbortReason::SelfCall);
        }
        let base = usize::from(instruction.a());
        let fixed = usize::from(instruction.b());
        let register = self.self_callee_register(callee)?;
        let open = self.take_self_open(instruction)?;
        let args = self.self_call_arguments(at, register, (base + 1, fixed), &open)?;
        self.special_register(ExternalSlot::TailCalls, Kind::Integer)?;
        self.tail = Some((args, at.pc));
        Ok(Flow::Return)
    }

    /// The interpreter came back from the callee of the self call the
    /// recorder slept through: the call's result register holds what the
    /// function returned, whose type every return of the trace gives.
    fn wake(&mut self, at: &HookPoint<'_>, wait: SelfCallWait) -> Result<(), TraceAbortReason> {
        if at.depth < wait.depth {
            return Err(TraceAbortReason::DepthChange);
        }
        if !Arc::ptr_eq(at.chunk, &self.chunk) || at.pc != wait.pc + 1 {
            return Err(TraceAbortReason::Divergence);
        }
        self.sleeping = None;
        // An open result count leaves the value in no register: it is what
        // the callee's frame returned.
        let kind = if wait.results == 0 {
            wait.returned.ok_or(TraceAbortReason::SelfCall)?
        } else {
            Kind::of(&at.registers[wait.base], self.integers)
        };
        let ty = match kind {
            Kind::Nil => TraceType::Nil,
            Kind::Boolean => TraceType::Boolean,
            Kind::Integer => TraceType::Integer,
            Kind::Float => TraceType::Float,
            Kind::Table => TraceType::Table,
            _ => return Err(TraceAbortReason::SelfCall),
        };
        let function = self.function.as_mut().ok_or(TraceAbortReason::Divergence)?;
        match function.result {
            Some(known) if known != ty => return Err(TraceAbortReason::TypeUnstable),
            _ => function.result = Some(ty),
        }
        self.ops[wait.op as usize].ty = Some(ty);
        if wait.results == 0 {
            // The value goes on to the self call or tail call that takes it.
            self.open = Some(vec![Open::Op(wait.op)]);
            self.open_from_self = true;
            return Ok(());
        }
        self.write(wait.base, Sym::Op(wait.op));
        if wait.results > 1 {
            // The function returns one value.
            let nil = self.emit(
                wait.pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            for register in wait.base + 1..wait.base + wait.results {
                self.write(register, Sym::Op(nil));
            }
        }
        Ok(())
    }

    /// The `Return` of a function trace's activation, which closes it. The
    /// function returns one number, boolean, nil or table.
    fn record_function_return(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        if instruction.opcode() != Ok(OpCode::Return) || instruction.b() != 1 {
            return Err(TraceAbortReason::SelfCall);
        }
        let a = usize::from(instruction.a());
        if matches!(self.sym(a), Sym::Root(_)) {
            return Err(TraceAbortReason::SelfCall);
        }
        let (operand, ty) = self.read(at.registers, a)?;
        if !matches!(
            ty,
            TraceType::Nil
                | TraceType::Boolean
                | TraceType::Integer
                | TraceType::Float
                | TraceType::Table
        ) {
            return Err(TraceAbortReason::SelfCall);
        }
        let function = self.function.as_mut().ok_or(TraceAbortReason::Divergence)?;
        match function.result {
            Some(known) if known != ty => return Err(TraceAbortReason::TypeUnstable),
            _ => function.result = Some(ty),
        }
        self.returned = Some(operand);
        Ok(Flow::Return)
    }

    /// The window register of the root frame's tail call count, naming it on
    /// first use. It is an integer in every profile: the count is the
    /// frame's, not a Lua value.
    fn tail_count_register(&mut self) -> Result<usize, TraceAbortReason> {
        self.special_register(ExternalSlot::TailCalls, Kind::Integer)
    }

    /// The window register of the closure the root frame runs at entry,
    /// naming it on first use.
    fn root_closure_register(&mut self) -> Result<usize, TraceAbortReason> {
        if self.home_closure.is_none() {
            return Err(TraceAbortReason::RootTailCall);
        }
        self.special_register(ExternalSlot::RootClosure, Kind::Closure)
    }

    /// A window slot the VM fills from the root frame itself, not from a cell
    /// or global, with its entry kind.
    fn special_register(
        &mut self,
        slot: ExternalSlot,
        kind: Kind,
    ) -> Result<usize, TraceAbortReason> {
        if let Some(position) = self.externals.iter().position(|known| *known == slot) {
            return Ok(self.register_count + position);
        }
        if self.externals.len() >= MAX_TRACE_EXTERNALS {
            return Err(limit("externals"));
        }
        self.externals.push(slot);
        self.external_cells.push(None);
        self.entry.push(kind);
        self.regs.push(Sym::Carried);
        self.written.push(false);
        Ok(self.regs.len() - 1)
    }

    /// The symbolic value of a root window register, with no check against a
    /// live value: for the slots the recorder fills itself.
    fn read_window(&self, register: usize) -> Result<(Operand, TraceType), TraceAbortReason> {
        let (operand, kind) = match self.regs[register] {
            Sym::Carried => (Operand::Carried(register), self.entry[register]),
            Sym::Op(index) => {
                let ty = self.ops[index as usize]
                    .ty
                    .expect("an operand operation defines a value");
                (Operand::Op(index), Kind::from_type(ty))
            }
            Sym::Root(_) | Sym::Virtual(_) => return Err(TraceAbortReason::Divergence),
        };
        let ty = kind.trace_type().ok_or(TraceAbortReason::Divergence)?;
        Ok((operand, ty))
    }

    /// Emits the guard that leaves, through a `Before` snapshot of the call
    /// at `pc`, unless the callee in `callee` is the pinned closure of lane
    /// `lane`. Returns the snapshot.
    ///
    /// No guard is emitted when an earlier one of the recording compared the
    /// same value with that closure (see [`Self::callee_guarded`]); there is
    /// no snapshot then.
    fn guard_callee(&mut self, pc: usize, callee: Operand, lane: u32) -> Option<u32> {
        if self.callee_guarded(callee, lane) {
            return None;
        }
        Some(self.guard_callee_always(pc, callee, lane))
    }

    /// [`Self::guard_callee`] that always emits the guard and its snapshot.
    fn guard_callee_always(&mut self, pc: usize, callee: Operand, lane: u32) -> u32 {
        let pinned = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        let before = self.guard_same(pc, callee, Operand::Op(pinned));
        let guard = u32::try_from(self.ops.len() - 1).expect("op count fits u32");
        self.callee_guards
            .push((self.value_origin(callee), lane, guard));
        before
    }

    /// The value `operand` copies, past its moves.
    fn value_origin(&self, operand: Operand) -> Operand {
        let mut origin = operand;
        while let Operand::Op(index) = origin {
            match self.ops[index as usize].kind {
                RecKind::Move(source) => origin = source,
                _ => break,
            }
        }
        origin
    }

    /// Whether an earlier guard of the recording compared the value `callee`
    /// copies with the closure of lane `lane` and went on.
    ///
    /// The recording is one path, so that guard ran before this call on
    /// every run that reaches it, and the operands are values, not
    /// registers: an upvalue the trace reads from a window slot it never
    /// writes is the slot's entry value (`Operand::Carried`) at every read.
    /// Mineclonia's `load`-made ULL helpers are such upvalues: the RNG step
    /// of `get_structure_starts` (structures.lua:744) calls them 254 times
    /// in one recording, and each call took a guard and a snapshot of its
    /// own. A guard a value select took away (`select.rs` truncates the
    /// operations) is no longer at its index, which the check sees.
    fn callee_guarded(&self, callee: Operand, lane: u32) -> bool {
        let origin = self.value_origin(callee);
        self.callee_guards.iter().any(|&(held, held_lane, guard)| {
            held == origin
                && held_lane == lane
                && self.ops.get(guard as usize).is_some_and(|op| {
                    let RecKind::GuardBranch {
                        condition: Operand::Op(same),
                        expected: true,
                        ..
                    } = op.kind
                    else {
                        return false;
                    };
                    matches!(
                        self.ops.get(same as usize).map(|op| &op.kind),
                        Some(RecKind::Compare(Comparison::Equal, value, Operand::Op(pinned)))
                            if self.value_origin(*value) == origin
                                && matches!(
                                    self.ops.get(*pinned as usize).map(|op| &op.kind),
                                    Some(RecKind::Constant(TraceScalar::Closure(held)))
                                        if *held == u64::from(lane)
                                )
                    )
                })
        })
    }

    /// Emits the guard that leaves, through a `Before` snapshot of the call
    /// at `pc`, unless `callee` is the closure `expected` holds. Returns the
    /// snapshot.
    fn guard_same(&mut self, pc: usize, callee: Operand, expected: Operand) -> u32 {
        let same = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, callee, expected),
            Some(TraceType::Boolean),
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(same),
                expected: true,
                snapshot: before,
            },
            None,
        );
        before
    }

    /// Emits the guard that leaves, through a `Before` snapshot of the call
    /// at `pc`, unless `callee` runs the prototype of the pinned closure of
    /// lane `lane`. Returns the snapshot.
    fn guard_proto(&mut self, pc: usize, callee: Operand, lane: u32) -> u32 {
        let pinned = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        let same = self.emit_helper_call(
            pc,
            HelperId::CLOSURE_SAME_PROTO,
            &[
                (callee, TraceType::Closure),
                (Operand::Op(pinned), TraceType::Closure),
            ],
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(same),
                expected: true,
                snapshot: before,
            },
            None,
        );
        before
    }

    /// Whether the last write of `register` in front of the instruction being
    /// recorded, back to the loop's header, reloads the same value on every
    /// iteration: a `GetUpvalue` of a cell that no `SetUpvalue` of the loop
    /// writes, or a `Move` of a register that is such a reload itself or is
    /// not written in the loop at all (the call base that takes `f` from a
    /// local). The results of the previous iteration's call land in that
    /// register too, and the reload overwrites them before the call. The
    /// callee is then one closure in every iteration, as it is when the
    /// recording itself ran the load. A side that starts at the call finds the
    /// register carried and must reach the same answer. The answer only picks
    /// the guard (identity or prototype) of the call, which leaves to the
    /// interpreter when the callee is another one.
    fn reloaded_invariant(&self, register: u8, before: usize, depth: u32) -> bool {
        let code = &self.chunk.code;
        let last = self.latch.min(code.len().saturating_sub(1));
        let Some(pc) = (self.header..before.min(last + 1))
            .rev()
            .find(|pc| self.chunk.writes_register_between(*pc, *pc + 1, register))
        else {
            return false;
        };
        // A branch in front of the reload that jumps past it leaves another
        // path with the carried value, so the reload is not the callee on
        // every path.
        if (self.header..pc).any(|branch| {
            nest::jump_target(branch, code[branch])
                .is_some_and(|target| target > pc && target <= before)
        }) {
            return false;
        }
        let instruction = code[pc];
        match instruction.opcode() {
            Ok(OpCode::GetUpvalue) if instruction.a() == register => {
                !code[self.header..=last].iter().any(|other| {
                    other.opcode() == Ok(OpCode::SetUpvalue) && other.d() == instruction.d()
                })
            }
            Ok(OpCode::Move) if instruction.a() == register && depth < 4 => {
                !self
                    .chunk
                    .writes_register_between(self.header, last + 1, instruction.b())
                    || self.reloaded_invariant(instruction.b(), pc, depth + 1)
            }
            _ => false,
        }
    }

    /// Whether the closure `callee` names is not the same one in every
    /// iteration: a register of the root frame that the loop writes (the
    /// locals of a rotating set of closures), or a closure read out of an
    /// array. A closure that is one in every iteration stays pinned by
    /// identity, and its cells live in window registers.
    fn callee_varies(&self, callee: Operand) -> bool {
        let mut origin = callee;
        while let Operand::Op(index) = origin {
            match self.ops[index as usize].kind {
                RecKind::Move(source) => origin = source,
                RecKind::ArrayLoad { .. } => return true,
                RecKind::CallHelper {
                    helper: HelperId::CELL_GET_CALLABLE,
                    ..
                } => return self.instance_callees.contains(&index),
                _ => return false,
            }
        }
        match origin {
            // A guest register of the root frame: a window slot past them is a
            // cell or a global, which the loop's bytecode does not write.
            Operand::Carried(register)
                if self.frames.is_empty()
                    && self.function.is_none()
                    && self.activation.is_none()
                    && register < self.register_count
                    && self.boxed.is_none_or(|boxed| boxed != register) =>
            {
                u8::try_from(register).is_ok_and(|register| {
                    self.chunk
                        .writes_register_between(self.header, self.latch + 1, register)
                        && !(self.side.is_some()
                            && self.reloaded_invariant(register, self.current_pc, 0))
                })
            }
            // A callee kept in an upvalue of the root closure (the loop runs
            // in a function over locals of its enclosing one): the loop
            // rewrites it if it has a `SetUpvalue` of it.
            Operand::Carried(register)
                if self.frames.is_empty()
                    && self.function.is_none()
                    && self.activation.is_none()
                    && register >= self.register_count =>
            {
                matches!(
                    self.externals.get(register - self.register_count),
                    Some(ExternalSlot::Upvalue(index))
                        if self.chunk.code[self.header..=self.latch.min(self.chunk.code.len() - 1)]
                            .iter()
                            .any(|instruction| {
                                instruction.opcode() == Ok(OpCode::SetUpvalue)
                                    && instruction.d() == *index
                            })
                )
            }
            Operand::Carried(_) | Operand::Op(_) => false,
        }
    }

    /// Whether the recording may read and write a callee's cells through the
    /// live closure. A table a window slot reads a global from (an
    /// `UpvalueField`) is read once at entry, and a write to the cell it was
    /// read through would leave it stale. A cell the trace holds in a window
    /// slot as a variable is no obstacle: a run refuses to call through the
    /// prototype guard a closure that shares one (see
    /// `RunTables::held_cells`).
    fn may_use_dynamic_cells(&self) -> bool {
        self.function.is_none()
            && self.activation.is_none()
            && self.reexec.is_none()
            && !self.externals.iter().any(|slot| {
                matches!(
                    slot,
                    ExternalSlot::UpvalueField { .. } | ExternalSlot::ClosureUpvalueField { .. }
                )
            })
    }

    /// Whether the closure `handle` has a cell that the recording holds as a
    /// window variable (a window slot, or a captured local of the root
    /// frame) and that either side writes: the window wrote it, or the
    /// callee's chunk has a `SetUpvalue` of it. A run zeroes the row of a
    /// closure that shares a cell with a slot some member of the tree writes
    /// (`written_cell_slots` in `native`), so the prototype guard would miss
    /// on it every time and every miss would ask for a side that misses the
    /// same way. It stays behind the identity guard, which reads the cell as
    /// the window variable it is. A callee that indexes such a cell with
    /// `GetTabUp` or `SetTabUp` (`_ENV` in Lua 5.2 and later) stays behind it
    /// too: a frame that runs whichever closure its call had cannot index
    /// its cells that way.
    ///
    /// A cell that neither side writes holds the value it held at entry for
    /// the whole run, which the closure's own cell holds too: every
    /// `next_integer` closure of Mineclonia's xoroshiro shares `bor`, and
    /// every `chunk_test` closure `floor` and `set_region_seed`, with the
    /// window slots of the closure the recording pinned for another call.
    /// A write through the live closure ([`HelperId::CELL_SET`]) to such a
    /// cell would leave the slot stale, so a tree that makes one counts
    /// every cell slot as written.
    fn shares_window_cell(&self, at: &HookPoint<'_>, handle: ClosureHandle, chunk: &Chunk) -> bool {
        let Some(surroundings) = at.surroundings else {
            return true;
        };
        let Some(parts) = surroundings.runtime.closure_call_parts(handle) else {
            return true;
        };
        let written = |register: usize| self.written.get(register).copied().unwrap_or(true);
        // How the callee's own code uses its upvalue `index`.
        let callee_uses = |index: usize| {
            chunk
                .code
                .iter()
                .any(|instruction| match instruction.opcode() {
                    Ok(OpCode::SetUpvalue) => usize::from(instruction.d()) == index,
                    Ok(OpCode::GetTabUp) => usize::from(instruction.b()) == index,
                    Ok(OpCode::SetTabUp) => usize::from(instruction.a()) == index,
                    _ => false,
                })
        };
        let shared = |index: usize, cell: &CellHandle| {
            let forced = self.dynamic_cell_writes || callee_uses(index);
            surroundings.upvalues.contains(cell) && forced
                || self
                    .external_cells
                    .iter()
                    .enumerate()
                    .any(|(position, held)| {
                        held.is_none_or(|held| held == *cell)
                            && (forced || written(self.register_count + position))
                    })
                || self.root_captured.iter().any(|(register, captured)| {
                    captured == cell && (forced || written(usize::from(*register)))
                })
        };
        parts
            .cells
            .iter()
            .enumerate()
            .any(|(index, cell)| shared(index, cell))
    }

    /// The operation holding the closure of the running frame, when the frame
    /// runs whichever closure of its prototype the call had.
    fn dynamic_closure(&self) -> Option<u32> {
        self.frames.last().and_then(|frame| frame.dynamic)
    }

    /// Before the first global read of a frame that runs whichever closure
    /// of a prototype its call had: guards, through a `Before` snapshot of
    /// the read at `pc`, that the live closure has the environment of the
    /// closure the recording pinned for the frame, whose globals the trace
    /// then reads. Does nothing in any other frame.
    fn guard_dynamic_environment(&mut self, pc: usize) -> Result<(), TraceAbortReason> {
        let Some(frame) = self.frames.last() else {
            return Ok(());
        };
        let (Some(live), lane, id) = (frame.dynamic, frame.closure, frame.frame) else {
            return Ok(());
        };
        if self.env_guarded.contains(&id) {
            return Ok(());
        }
        if self.function.is_some() || self.activation.is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let pinned = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        let same = self.emit_helper_call(
            pc,
            HelperId::CLOSURE_SAME_ENV,
            &[
                (Operand::Op(live), TraceType::Closure),
                (Operand::Op(pinned), TraceType::Closure),
            ],
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(same),
                expected: true,
                snapshot: before,
            },
            None,
        );
        self.env_guarded.push(id);
        Ok(())
    }

    /// The call at `at.pc` leaves open results, which a later call or return
    /// takes after at most some moves. If it is a call, guards its callee now,
    /// before this call runs, so that no guard has to leave while the
    /// interpreter holds the results in the frame's open-results slot. The
    /// callee is a register this call does not write, or one that moves
    /// copy into place. Anything between the two other than moves aborts the
    /// recording.
    fn hoist_callee(&mut self, at: &HookPoint<'_>) -> Result<(), TraceAbortReason> {
        let pc = at.pc;
        let mut end = pc + 1;
        let consumer = loop {
            let next = at
                .chunk
                .code
                .get(end)
                .copied()
                .ok_or(TraceAbortReason::Divergence)?;
            match next.opcode() {
                Ok(OpCode::Move) if end - pc <= MAX_OPEN_MOVES => end += 1,
                Ok(OpCode::ReturnOpen | OpCode::CallOpen | OpCode::TailCallOpen) => break next,
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            }
        };
        if consumer.opcode() == Ok(OpCode::ReturnOpen) {
            return Ok(());
        }
        // Where the callee was when this call started: back through the moves.
        let mut callee = usize::from(consumer.a());
        for between in at.chunk.code[pc + 1..end].iter().rev() {
            if usize::from(between.a()) == callee {
                callee = usize::from(between.b());
            }
        }
        self.check_register(callee)?;
        let (operand, ty) = self.read(at.registers, callee)?;
        if ty != TraceType::Closure {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let identity = match at.registers[callee] {
            Value::Closure(handle) => handle.native_identity(),
            Value::NativeFunction(function) => pinned_native_identity(function.index()),
            Value::BuiltinFunction(BuiltinFunction::ToString) => {
                pinned_builtin_identity(PINNED_BUILTIN_TOSTRING)
            }
            Value::BuiltinFunction(BuiltinFunction::StringFormat) => {
                pinned_builtin_identity(PINNED_BUILTIN_STRING_FORMAT)
            }
            Value::BuiltinFunction(BuiltinFunction::StringGsub) => {
                pinned_builtin_identity(PINNED_BUILTIN_STRING_GSUB)
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let lane = self.pin(identity)?;
        self.guard_callee(pc, operand, lane);
        let start = (consumer.opcode() == Ok(OpCode::CallOpen)).then(|| {
            let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            (before, self.ops.len(), pc)
        });
        self.hoisted.push(Hoisted {
            callee: operand,
            consumer: (self.frames.len(), end),
            start,
        });
        Ok(())
    }

    /// Records a call of a native function whose arguments the recorder reads
    /// as symbols: `setmetatable` of a virtual table, `rawequal` of strings,
    /// `select`, `assert` and the raw table accesses. `None` if `function`
    /// is none of them for this call. The arguments start at `first_argument`.
    fn record_special_native(
        &mut self,
        at: &HookPoint<'_>,
        function: (Operand, NativeFunctionHandle),
        (first_argument, fixed): (usize, usize),
        (base, results): (usize, usize),
        (with_varargs, consumes_open, tail_opcode): (bool, bool, bool),
    ) -> Option<Result<Flow, TraceAbortReason>> {
        let function_handle = function.1;
        let kind = Self::native_kind(at, function_handle);
        // The metatable of a table that already escaped belongs to the real
        // table, which no exit or helper would give it from the model.
        if kind == Some(DirectNativeKind::SetMetatable)
            && !with_varargs
            && !consumes_open
            && !tail_opcode
            && fixed == 2
            && self
                .virtual_in(first_argument)
                .is_some_and(|id| self.virtuals[id as usize].real.is_some())
        {
            return Some(Err(TraceAbortReason::VirtualEscape));
        }
        // `setmetatable` of a table the trace never made keeps the table
        // virtual: the exit sets the metatable.
        if kind == Some(DirectNativeKind::SetMetatable)
            && !with_varargs
            && !consumes_open
            && !tail_opcode
            && fixed == 2
            && self.virtual_in(first_argument).is_some()
        {
            return Some(self.record_virtual_set_metatable(
                at,
                function,
                first_argument,
                (base, results),
            ));
        }
        // `rawequal` of strings compares their bytes.
        if kind == Some(DirectNativeKind::RawEqual)
            && !with_varargs
            && !consumes_open
            && fixed >= 2
            && (0..2).all(|index| matches!(at.registers[first_argument + index], Value::String(_)))
        {
            return Some(self.record_rawequal_strings(
                at,
                function,
                first_argument,
                (base, results),
            ));
        }
        // `error` inside a protected call of the trace ends in a caught exit.
        if kind == Some(DirectNativeKind::Error) {
            if with_varargs || consumes_open {
                return Some(Err(TraceAbortReason::UnsupportedOperand));
            }
            return Some(self.record_error(at, function, (first_argument, fixed)));
        }
        // `select` and `assert` take their arguments as symbols: a string
        // constant is one, and the varargs of the running frame are values.
        let kind = kind.filter(|kind| Self::is_symbolic_kind(*kind))?;
        if kind == DirectNativeKind::Select {
            return Some(self.record_select(
                at,
                function,
                (first_argument, fixed),
                (with_varargs, consumes_open),
                (base, results),
            ));
        }
        // A guard on the first argument could not leave with open results
        // pending.
        if consumes_open {
            return Some(Err(TraceAbortReason::UnsupportedOperand));
        }
        if Self::is_raw_access_kind(kind) {
            if with_varargs {
                return Some(Err(TraceAbortReason::UnsupportedOperand));
            }
            return Some(self.record_raw_access(
                at,
                kind,
                function,
                (first_argument, fixed),
                (base, results),
            ));
        }
        Some(self.record_assert(
            at,
            function,
            (first_argument, fixed),
            with_varargs,
            (base, results),
        ))
    }

    /// A call of a native function that has a pure numeric form: one or two
    /// floats in, one float out (`math.sin`, `math.max`, `tonumber` of a
    /// number). The interpreter runs the call as it always does and the trace
    /// calls the function itself. A guard compares the live callee with the
    /// pinned native one, so another function runs in the interpreter. The
    /// call that takes open results has that guard before the call that made
    /// them (`hoisted`). The one float is the call's only result, which an
    /// open call leaves for its consumer.
    fn record_native_call(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        function: NativeFunctionHandle,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let kind = runtime.direct_native_kind(function);
        if let Some(kind) = kind
            && Self::is_inline_kind(kind)
        {
            return self.record_inline_native(
                at,
                kind,
                (callee, function),
                (arguments, first_argument),
                (base, results),
                hoisted,
            );
        }
        if let Some(kind) = kind
            && Self::is_bit_kind(kind)
            && !arguments.iter().any(|(_, ty)| ty.is_wide())
        {
            return self.record_bit_native(
                at,
                kind,
                (callee, function),
                (arguments, first_argument),
                (base, results),
                hoisted,
            );
        }
        if let Some(kind) = kind
            && Self::is_table_write_kind(kind)
        {
            return self.record_table_write(
                at,
                kind,
                (callee, function),
                (arguments, first_argument),
                (base, results),
                hoisted,
            );
        }
        if let Some(kind) = kind
            && Self::is_table_kind(kind)
        {
            return self.record_table_native(
                at,
                kind,
                (callee, function),
                (arguments, first_argument),
                (base, results),
                hoisted,
            );
        }
        if let Some(kind) = kind
            && !hoisted
            && !arguments.iter().any(|(_, ty)| ty.is_wide())
            && let Some(flow) = self.record_host_native(
                at,
                (kind, function),
                callee,
                (arguments, first_argument),
                (base, results),
            )?
        {
            return Ok(flow);
        }
        if let Some(kind) = kind
            && arguments.iter().any(|(_, ty)| ty.is_wide())
        {
            return self.record_wide_native(
                pc,
                kind,
                callee,
                function,
                arguments,
                (base, results),
                hoisted,
            );
        }
        if let Some(kind @ (DirectNativeKind::StringByte | DirectNativeKind::StringSlice)) = kind {
            // A snapshot between the call that made open results and the
            // one that takes them cannot hold them, so these calls leave
            // before themselves only as the call that makes the results.
            if hoisted {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            return self.record_string_native(
                at,
                kind,
                (callee, function),
                (arguments, first_argument),
                (base, results),
            );
        }
        if matches!(
            kind,
            Some(
                DirectNativeKind::MathLog
                    | DirectNativeKind::MathLogBase
                    | DirectNativeKind::MathPow
            )
        ) {
            return self.record_math_float(
                at,
                kind,
                (callee, function),
                arguments,
                (base, results),
                hoisted,
            );
        }
        // The numeric forms take exactly the argument types the scalar
        // callback is documented to agree with: `Value::Number`.
        let mut operands = [Operand::Op(0); 2];
        let wanted = match (
            runtime.native_numeric_unary(function),
            runtime.native_numeric_binary(function),
        ) {
            (Some(_), _) => 1,
            (None, Some(_)) => 2,
            (None, None) => {
                if let Some(flow) = self.record_vm_call(at, function, hoisted, (base, results))? {
                    return Ok(flow);
                }
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        };
        // `math.max` and `math.min` of more than two numbers fold the
        // binary form from the left, as the native's loop does
        // (`math_extreme`): `mathmax (0, a, b)` at structures.lua:1271.
        let folds = wanted == 2
            && arguments.len() > 2
            && matches!(
                kind,
                Some(DirectNativeKind::MathMax | DirectNativeKind::MathMin)
            );
        if arguments.len() != wanted && !folds {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if arguments.iter().any(|(_, ty)| *ty != TraceType::Float) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        for (slot, (operand, _)) in operands.iter_mut().zip(arguments) {
            *slot = *operand;
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        if hoisted {
            if !self.take_hoisted(callee) {
                return Err(TraceAbortReason::Divergence);
            }
        } else {
            self.guard_callee(pc, callee, lane);
        }
        let kind = match (
            runtime.native_numeric_unary(function),
            runtime.native_numeric_binary(function),
        ) {
            // `tonumber` of a number is the number: no call.
            (Some(_), _) if kind == Some(DirectNativeKind::Tonumber) => RecKind::Move(operands[0]),
            (Some(unary), _) => RecKind::NativeUnary(NativeUnary(unary), operands[0]),
            (None, Some(binary)) => {
                let mut value = self.emit(
                    pc,
                    RecKind::NativeBinary(NativeBinary(binary), operands[0], operands[1]),
                    Some(TraceType::Float),
                );
                for (operand, _) in &arguments[2..] {
                    value = self.emit(
                        pc,
                        RecKind::NativeBinary(NativeBinary(binary), Operand::Op(value), *operand),
                        Some(TraceType::Float),
                    );
                }
                return Ok(self.native_result(pc, value, (base, results)));
            }
            (None, None) => return Err(TraceAbortReason::Divergence),
        };
        let value = self.emit(pc, kind, Some(TraceType::Float));
        Ok(self.native_result(pc, value, (base, results)))
    }

    /// A call of `math.log` or `math.pow` on numbers. The operations are the
    /// native's own pure functions, as for the other float natives, with an
    /// integer argument converted to a float first (the native converts it
    /// the same way). The logarithm in a base is the profile's own pure
    /// function of the number and the base (`log2` and `log10` for the bases
    /// Lua 5.3 and later special-case, `LuaJIT`'s reciprocal form).
    fn record_math_float(
        &mut self,
        at: &HookPoint<'_>,
        kind: Option<DirectNativeKind>,
        (callee, function): (Operand, NativeFunctionHandle),
        arguments: &[(Operand, TraceType)],
        (base, results): (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let wanted = if kind == Some(DirectNativeKind::MathLog) {
            // Lua 5.1 and LuaJIT ignore the arguments after the first.
            1..=2
        } else if kind == Some(DirectNativeKind::MathLogBase) {
            1..=2
        } else {
            2..=2
        };
        if !wanted.contains(&arguments.len()) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        if hoisted {
            if !self.take_hoisted(callee) {
                return Err(TraceAbortReason::Divergence);
            }
        } else {
            self.guard_callee(pc, callee, lane);
        }
        let mut floats = Vec::with_capacity(2);
        for (operand, ty) in arguments {
            floats.push(match ty {
                TraceType::Float => *operand,
                TraceType::Integer => Operand::Op(self.emit(
                    pc,
                    RecKind::IntToFloat(*operand),
                    Some(TraceType::Float),
                )),
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            });
        }
        let value = if kind == Some(DirectNativeKind::MathPow) {
            let binary = runtime
                .native_numeric_binary(function)
                .ok_or(TraceAbortReason::UnsupportedOperand)?;
            RecKind::NativeBinary(NativeBinary(binary), floats[0], floats[1])
        } else {
            let unary = runtime
                .native_numeric_unary(function)
                .ok_or(TraceAbortReason::UnsupportedOperand)?;
            if kind == Some(DirectNativeKind::MathLogBase) && floats.len() == 2 {
                let with_base = runtime
                    .native_log_base(function)
                    .ok_or(TraceAbortReason::UnsupportedOperand)?;
                RecKind::NativeBinary(NativeBinary(with_base), floats[0], floats[1])
            } else {
                RecKind::NativeUnary(NativeUnary(unary), floats[0])
            }
        };
        let value = self.emit(pc, value, Some(TraceType::Float));
        Ok(self.native_result(pc, value, (base, results)))
    }

    /// Gives a native call its one result: the call's register, with nil for
    /// the others, or the open results of the call for the instruction after.
    fn native_result(&mut self, pc: usize, value: u32, (base, results): (usize, usize)) -> Flow {
        if results == 0 {
            self.open = Some(vec![Open::Op(value)]);
            return Flow::Next(pc + 1);
        }
        self.write(base, value);
        if results > 1 {
            // The call's other results are nil.
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            for register in base + 1..base + results {
                self.write(register, nil);
            }
        }
        Flow::Next(pc + 1)
    }

    /// Whether a `string.sub` call is the one-byte form that
    /// [`Self::record_string_native`] records: the same index twice, on a
    /// string bound at entry, for a call whose results are open and so go to
    /// `string.byte`.
    fn is_one_byte_slice(&self, args: &[(Operand, TraceType)], results: usize) -> bool {
        results == 0
            && args.len() == 3
            && args[0].1 == TraceType::String
            && self.origin(args[1].0) == self.origin(args[2].0)
            && self.bound_string(args[0].0).is_ok()
    }

    /// The type of a number: what `#s` and `string.byte` give.
    const fn number_type(&self) -> TraceType {
        if self.integers {
            TraceType::Integer
        } else {
            TraceType::Float
        }
    }

    /// Whether the guard of the callee `callee` was made before the call that
    /// made the open results this call takes. Takes the note.
    fn take_hoisted(&mut self, callee: Operand) -> bool {
        let key = (self.frames.len(), self.current_pc);
        let Some(index) = self.hoisted.iter().rposition(|note| note.consumer == key) else {
            return false;
        };
        let note = self.hoisted[index].callee;
        self.hoisted.truncate(index);
        self.origin(note) == self.origin(callee)
    }

    /// Forgets the note of the consumer at `pc` of the frame at `depth` and
    /// the notes of calls that never got one.
    fn drop_hoisted(&mut self, depth: usize, pc: usize) {
        if let Some(index) = self
            .hoisted
            .iter()
            .rposition(|note| note.consumer == (depth, pc))
        {
            self.hoisted.truncate(index);
        }
    }

    /// The operand a chain of moves copies `operand` from.
    fn origin(&self, mut operand: Operand) -> Operand {
        while let Operand::Op(index) = operand
            && let RecKind::Move(source) = self.ops[index as usize].kind
        {
            operand = source;
        }
        operand
    }

    /// Whether operation `index` holds the lane of a table the trace made
    /// ([`HelperId::TABLE_NEW`]), whatever moves it went through.
    fn is_made_table(&self, index: u32) -> bool {
        match self.origin(Operand::Op(index)) {
            Operand::Op(made) => matches!(
                &self.ops[made as usize].kind,
                RecKind::CallHelper { helper, .. } if *helper == HelperId::TABLE_NEW
            ),
            Operand::Carried(_) => false,
        }
    }

    /// Whether operation `index` holds a table a self call returned,
    /// whatever moves it went through.
    fn is_self_call_table(&self, index: u32) -> bool {
        match self.origin(Operand::Op(index)) {
            Operand::Op(call) => {
                let op = &self.ops[call as usize];
                matches!(op.kind, RecKind::SelfCall { .. }) && op.ty == Some(TraceType::Table)
            }
            Operand::Carried(_) => false,
        }
    }

    /// The window register of the string `operand` stands for, which the
    /// entry binds: the string must be the same one in every iteration, so
    /// it comes, through moves, from a register the iteration never writes
    /// ([`Self::finalize`] checks that).
    fn bound_string(&self, mut operand: Operand) -> Result<usize, TraceAbortReason> {
        loop {
            match operand {
                Operand::Carried(window) => return Ok(window),
                Operand::Op(index) => match self.ops[index as usize].kind {
                    RecKind::Move(source) => operand = source,
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                },
            }
        }
    }

    /// A call of `string.byte` with a string and at most an index, or of
    /// `string.sub` with a string and one index twice. The string is bound at
    /// entry. A byte reads from its bytes, and leaves through a `Before`
    /// snapshot unless the index is an integral number inside the string, so
    /// the interpreter makes every other call (an empty result, a negative
    /// index, an error). `sub` of one byte is not a string the trace makes:
    /// it leaves the byte as an open result that only `string.byte` takes
    /// (see [`Open::Char`]).
    fn record_string_native(
        &mut self,
        at: &HookPoint<'_>,
        kind: DirectNativeKind,
        (callee, function): (Operand, NativeFunctionHandle),
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let number = self.number_type();
        if let (
            DirectNativeKind::StringByte,
            [
                (string, TraceType::String),
                (first, first_ty),
                (last, last_ty),
            ],
        ) = (kind, arguments)
            && is_number(*first_ty)
            && is_number(*last_ty)
        {
            // A run of bytes: `string.byte (s, i, j)`.
            let lane = self.pin(pinned_native_identity(function.index()))?;
            self.guard_callee(pc, callee, lane);
            return self.record_byte_run(
                at,
                (*string, (*first, *first_ty), (*last, *last_ty)),
                (
                    &at.registers[first_argument],
                    &at.registers[first_argument + 1],
                    &at.registers[first_argument + 2],
                ),
                (base, results),
            );
        }
        let (string, index) = match (kind, arguments) {
            (DirectNativeKind::StringByte, [(string, TraceType::String)]) => (*string, None),
            (DirectNativeKind::StringByte, [(string, TraceType::String), (index, ty)])
                if is_number(*ty) =>
            {
                (*string, Some(*index))
            }
            (
                DirectNativeKind::StringSlice,
                [
                    (string, TraceType::String),
                    (first, first_ty),
                    (last, last_ty),
                ],
            ) if self.origin(*first) == self.origin(*last)
                && is_number(*first_ty)
                && is_number(*last_ty) =>
            {
                (*string, Some(*first))
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        if kind == DirectNativeKind::StringSlice && results != 0 {
            // The byte of a one-byte string is only for the call that takes it.
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let window = self.bound_string(string)?;
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let key = if let Some(index) = index {
            index
        } else {
            let one = if self.integers {
                TraceScalar::Integer(1)
            } else {
                TraceScalar::Float(1.0)
            };
            Operand::Op(self.emit(pc, RecKind::Constant(one), Some(number)))
        };
        let binding = self.binding(window, BindingKind::Bytes { ty: number });
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        let byte = self.emit(
            pc,
            RecKind::StringByte {
                binding,
                string: Operand::Carried(window),
                key,
                snapshot,
            },
            Some(number),
        );
        if kind == DirectNativeKind::StringSlice {
            self.open = Some(vec![Open::Char(byte)]);
            return Ok(Flow::Next(pc + 1));
        }
        Ok(self.native_result(pc, byte, (base, results)))
    }

    /// `string.byte` of the one-byte string a `string.sub` call left, which
    /// is the byte that call read. The callee was guarded before the `sub`
    /// call (see [`Recorder::hoist_callee`]).
    fn record_byte_of_char(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        function: NativeFunctionHandle,
        byte: u32,
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        self.pin(pinned_native_identity(function.index()))?;
        if !self.take_hoisted(callee) {
            return Err(TraceAbortReason::Divergence);
        }
        Ok(self.native_result(at.pc, byte, (base, results)))
    }

    /// At the header of an `ipairs` loop over strings, reads the element
    /// register again from the table, as the step that the interpreter ran
    /// just before did: `t[i]`, raw. The step reads a string out of the
    /// array as a borrow ([`TraceType::StringRef`]), which no loop parameter
    /// may be, while the entry finds an owned string in the register: carried
    /// across the loop-back the register made the loop type-unstable, and
    /// `indexof` (`for i, v in ipairs (list) do if v == val then return i
    /// end end`) never compiled. Read again, the register is written before
    /// anything reads it, and is no parameter. Only a loop whose iterator is
    /// the `ipairs` step (a guard checks it) holds `v == t[i]` at its header.
    fn reload_ipairs_element(&mut self, at: &HookPoint<'_>) -> Result<(), TraceAbortReason> {
        let chunk = Arc::clone(at.chunk);
        let Some(latch) = chunk.code.get(self.latch).copied() else {
            return Ok(());
        };
        if latch.opcode() != Ok(OpCode::ForInLoop)
            || self.target(self.latch, short_offset(latch)).ok() != Some(self.header)
        {
            return Ok(());
        }
        let base = usize::from(latch.a());
        let call = self
            .latch
            .checked_sub(1)
            .and_then(|pc| chunk.code.get(pc).copied());
        if !call.is_some_and(|call| {
            call.opcode() == Ok(OpCode::ForInCall) && usize::from(call.a()) == base && call.b() >= 2
        }) {
            return Ok(());
        }
        let element = base + 4;
        if element >= usize::from(chunk.register_count)
            || !matches!(
                at.registers.get(base),
                Some(Value::BuiltinFunction(BuiltinFunction::IPairsIterator))
            )
            || !matches!(at.registers.get(element), Some(Value::String(_)))
            || !matches!(at.registers.get(base + 1), Some(Value::Table(_)))
        {
            return Ok(());
        }
        // The reads stand for the step that the interpreter ran last.
        let pc = self.latch - 1;
        let (callee, _) = self.read(at.registers, base)?;
        let (receiver, handle, runtime) = self.table_receiver(at, base + 1)?;
        if !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None)) {
            return Ok(());
        }
        let key = self.read_number(at.registers, base + 2)?;
        // A guard here leaves in front of the step, with the control value
        // one back: the interpreter steps to this element again. The element
        // register is dead there, so no snapshot names the value the
        // previous trip left in it, and it is no parameter.
        let one = match key.live {
            Live::Integer(_) => TraceScalar::Integer(1),
            Live::Float(_) => TraceScalar::Float(1.0),
        };
        let one = self.emit(pc, RecKind::Constant(one), Some(key.ty()));
        let back = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Subtract, key.operand, Operand::Op(one)),
            Some(key.ty()),
        );
        let step = self.snapshot(SnapshotPhase::Before { pc }, &[(base + 2, Sym::Op(back))]);
        let reexec = self.reexec.replace(step);
        let read = self.reload_element(pc, (callee, receiver), (handle, runtime), key);
        self.reexec = reexec;
        let (read, ty) = read?;
        if ty == TraceType::Nil {
            return Err(TraceAbortReason::Divergence);
        }
        self.write(element, read);
        Ok(())
    }

    /// Gives each string the inlined frames hold as a borrow of an array slot
    /// a lane of its own, at the end of a recording in front of a loop in a
    /// call (see "Loops in calls" in the module docs of `trace`). The loop
    /// side starts in those frames and takes their registers as frame inputs,
    /// which a borrow cannot be: `val` of `indexof (list, val)`, read out of
    /// an array by the caller. The element register of an `ipairs` loop at
    /// `latch` is left alone, as the loop side reads it again
    /// ([`Self::reload_ipairs_element`]).
    fn own_frame_borrows(&mut self, at: &HookPoint<'_>, latch: usize) {
        let element = at
            .chunk
            .code
            .get(latch)
            .filter(|instruction| instruction.opcode() == Ok(OpCode::ForInLoop))
            .map(|instruction| usize::from(instruction.a()) + 4);
        let innermost = self.frames.len();
        let source = |ops: &[RecOp], mut index: u32| {
            while let RecKind::Move(Operand::Op(from)) = ops[index as usize].kind {
                index = from;
            }
            index
        };
        let borrowed = |ops: &[RecOp], position: usize, register: usize, sym: &Sym| match sym {
            Sym::Op(index)
                if (position + 1 != innermost || Some(register) != element)
                    && ops[*index as usize].ty.is_some_and(TraceType::is_borrowed) =>
            {
                Some(source(ops, *index))
            }
            _ => None,
        };
        let mut found: Vec<u32> = Vec::new();
        for (position, frame) in self.frames.iter().enumerate() {
            for (register, sym) in frame.regs.iter().enumerate() {
                if let Some(index) = borrowed(&self.ops, position, register, sym)
                    && !found.contains(&index)
                {
                    found.push(index);
                }
            }
        }
        let work = self.host_work;
        for old in found {
            let owned = self.emit_helper_call(
                latch,
                HelperId::STRING_OWN,
                &[(Operand::Op(old), TraceType::StringRef)],
            );
            let ops = &self.ops;
            for (position, frame) in self.frames.iter_mut().enumerate() {
                for (register, sym) in frame.regs.iter_mut().enumerate() {
                    if borrowed(ops, position, register, sym) == Some(old) {
                        *sym = Sym::Op(owned);
                    }
                }
            }
        }
        self.host_work = work;
    }

    /// The guarded read of [`Self::reload_ipairs_element`], whose snapshots
    /// are all the one in front of the step.
    fn reload_element(
        &mut self,
        pc: usize,
        (callee, receiver): (Operand, Receiver),
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
    ) -> Result<(u32, TraceType), TraceAbortReason> {
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_IPAIRS_ITERATOR))?;
        let pinned = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        let same = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, callee, Operand::Op(pinned)),
            Some(TraceType::Boolean),
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(same),
                expected: true,
                snapshot: before,
            },
            None,
        );
        let (read, ty) = match receiver {
            Receiver::Loaded(table) => {
                self.lane_ipairs_read(pc, (table, None), (handle, runtime), key, false)?
            }
            Receiver::Bound(window) | Receiver::Around(window) => self.lane_ipairs_read(
                pc,
                (Operand::Carried(window), Some(window)),
                (handle, runtime),
                key,
                false,
            )?,
        };
        Ok((read, ty))
    }

    /// One step of an `ipairs` loop: the call of the iterator with the table
    /// and the control value, which returns the next index and the element.
    /// The trace reads that element from the array part of the table, behind
    /// the usual array guard, so a step that ends the loop (a nil element, the
    /// end of the array, a metamethod) leaves before the call and the
    /// interpreter runs it.
    fn record_ipairs_step(
        &mut self,
        at: &HookPoint<'_>,
        callee: Operand,
        (first_argument, arguments): (usize, usize),
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        if arguments != 2 || results != 2 {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let (receiver, handle, runtime) = self.table_receiver(at, first_argument)?;
        let control = self.read_number(at.registers, first_argument + 1)?;
        let lane = self.pin(pinned_builtin_identity(PINNED_BUILTIN_IPAIRS_ITERATOR))?;
        let pinned = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Closure(u64::from(lane))),
            Some(TraceType::Closure),
        );
        let same = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, callee, Operand::Op(pinned)),
            Some(TraceType::Boolean),
        );
        // Inside a call that has stored nothing, a step of another iterator
        // starts the call over, as a table read there does.
        let before = self.read_snapshot(pc);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(same),
                expected: true,
                snapshot: before,
            },
            None,
        );
        // The next index is the control value plus one, in its own type.
        let (one, next_live) = match control.live {
            Live::Integer(index) => (
                TraceScalar::Integer(1),
                Live::Integer(index.wrapping_add(1)),
            ),
            Live::Float(index) => (TraceScalar::Float(1.0), Live::Float(index + 1.0)),
        };
        let one = self.emit(pc, RecKind::Constant(one), Some(control.ty()));
        let next = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Add, control.operand, Operand::Op(one)),
            Some(control.ty()),
        );
        let key = Number {
            operand: Operand::Op(next),
            live: next_live,
        };
        // A nil element ends the iteration. A root or a side that stays in
        // the root's loop cannot follow that (its recording would not
        // close), so the read misses. An outer side starts at the end and
        // runs on through the loops around the root: it reads the nil past
        // the array's end (or a hole) as `ipairs` does, raw, which only a
        // table without a metatable allows (`__index` would run in Lua 5.3).
        // So does a recording that unrolls the loop (a short one in an
        // inlined call, see `Self::may_unroll_while`): the step that ends
        // it is a guard like the others.
        //
        // The first step of a loop the recording's own loop holds (not its
        // header) that ends at once runs no trip: the recording goes on
        // after it as after an `if`, and closes. The structure starts of
        // Mineclonia's `collect_structure_references` (structures.lua:857)
        // are mostly empty lists, and the outer loop aborted there with
        // `TableMiss` until the site was rejected.
        let first = match control.live {
            Live::Integer(index) => index == 0,
            Live::Float(index) => index == 0.0,
        };
        let skips_inner = first && self.function.is_none() && self.ends_inner_generic_for(at, pc);
        // A side that starts at the step of a loop in a call, from the guard
        // of a loop side of that loop (see "Loops in calls" in the module
        // docs of `trace`), records the step that ends it and runs on out
        // of the call.
        let leaves_callee_loop = self.guest == 0
            && self.side.as_ref().is_some_and(|seed| {
                !seed.frames.is_empty()
                    && seed.start_pc == pc
                    && self.frames.len() == seed.frames.len()
            });
        let unrolls = leaves_callee_loop
            || (!self.outer_side && (skips_inner || self.unrolls_generic_for(at, pc)));
        let (element, ty) = match receiver {
            Receiver::Loaded(table) => {
                self.lane_ipairs_read(pc, (table, None), (handle, runtime), key, unrolls)?
            }
            Receiver::Bound(window) | Receiver::Around(window) => {
                let table = Operand::Carried(window);
                if self.outer_side {
                    self.array_read_or_miss(
                        pc,
                        (table, window),
                        (handle, runtime),
                        key,
                        (true, true),
                    )?
                } else {
                    self.lane_ipairs_read(
                        pc,
                        (table, Some(window)),
                        (handle, runtime),
                        key,
                        unrolls,
                    )?
                }
            }
        };
        if ty == TraceType::Nil {
            // The step returns nothing: the control variable is nil too.
            let nil = self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            );
            self.write(base, nil);
        } else {
            self.write(base, next);
        }
        self.write(base + 1, element);
        Ok(Flow::Next(pc + 1))
    }

    /// The element read of an `ipairs` step: of a table the entry binds
    /// (`window`), the bound array read; of a table the recording met by
    /// lane (a table it read out of another, say `ipairs (t.list)`), the read
    /// `GetIndex` makes of such a table ([`Self::lane_array_read`] or
    /// [`Self::lane_read`]). For the nil that ends the loop, when the
    /// recording follows the end (`ends`, an unrolled loop), a
    /// [`HelperId::TABLE_ABSENT_INDEX`] call, which leaves unless the key is
    /// still absent from a table without a metatable. A bound table takes
    /// the call too, not a read past the array's end: the read has no lane
    /// form, and a side that binds a table its tree's root rewrites (the
    /// list passed to the call it starts at) must give its bindings up for
    /// lane reads (`plan_tree`).
    fn lane_ipairs_read(
        &mut self,
        pc: usize,
        (table, window): (Operand, Option<usize>),
        (handle, runtime): (TableHandle, &Runtime),
        key: Number,
        ends: bool,
    ) -> Result<(u32, TraceType), TraceAbortReason> {
        let (key_type, live) = match key.live {
            Live::Integer(index) => (TraceType::Integer, Value::Integer(index)),
            Live::Float(number) => (TraceType::Float, Value::Number(number)),
        };
        let present = !matches!(
            runtime.table_get_raw_value(handle, &live),
            Ok(None | Some(Value::Nil))
        );
        if let Some(window) = window
            && (present || !ends)
        {
            return self.array_read(pc, (table, window), (handle, runtime), key, false);
        }
        if present {
            if let Some(read) = self.lane_array_read(pc, table, (handle, runtime), key)? {
                return Ok((read, self.operand_type(Operand::Op(read))));
            }
            let read = self.lane_read(
                pc,
                (table, handle, runtime),
                (key.operand, key_type, live),
                false,
            )?;
            return Ok((read, self.operand_type(Operand::Op(read))));
        }
        if !ends || !matches!(runtime.raw_metatable(&Value::Table(handle)), Ok(None)) {
            return Err(TraceAbortReason::TableMiss);
        }
        self.emit_helper_call(
            pc,
            HelperId::TABLE_ABSENT_INDEX,
            &[(table, TraceType::Table), (key.operand, key_type)],
        );
        let nil = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Nil),
            Some(TraceType::Nil),
        );
        Ok((nil, TraceType::Nil))
    }

    /// A `Return`, `ReturnOpen` or `Halt` of an inlined call: its first values
    /// become the caller's registers, and the rest of the caller's result
    /// window is nil. A call with open results leaves all of them for the
    /// instruction after it instead. `ReturnOpen` returns the open results of
    /// the call before it after its own fixed values.
    fn record_return(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let a = usize::from(instruction.a());
        let opcode = instruction.opcode();
        let count = if opcode == Ok(OpCode::Halt) {
            0
        } else {
            match usize::from(instruction.b()) {
                // Open results.
                0 if opcode == Ok(OpCode::Return) => {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                count => count,
            }
        };
        if self.frames.last().is_some_and(|frame| frame.handler) {
            return self.return_from_handler(at, instruction, count);
        }
        if self.frames.last().is_some_and(|frame| {
            matches!(
                (frame.entry, &frame.after),
                (Entry::Compare { .. } | Entry::Branch { .. }, _) | (_, Pending::Concat(_))
            )
        }) {
            return self.return_to_operator(at, instruction, count);
        }
        if self
            .frames
            .last()
            .is_some_and(|frame| matches!(frame.after, Pending::ToString))
        {
            return self.return_from_tostring(at, instruction, count);
        }
        let (results, base, protected, entry) = {
            let frame = self.frames.last().expect("an inlined frame returns");
            (
                frame.results,
                frame.result_base,
                frame.protected,
                frame.entry,
            )
        };
        if count > 0 {
            self.check_register(a + count - 1)?;
        }
        if entry == Entry::NewIndex {
            // The store ran a handler: whatever it returns is dropped, the
            // open results of a `ReturnOpen` included.
            if opcode == Ok(OpCode::ReturnOpen) {
                self.open.take().ok_or(TraceAbortReason::Divergence)?;
            }
            let callee = self.frames.pop().expect("an inlined frame returns");
            if callee.owns_reexec {
                self.reexec = None;
            }
            if self.frames.is_empty() {
                self.call_site = None;
            }
            return Ok(Flow::Leave {
                resume: callee.call_pc + 1,
            });
        }
        // A protected call delivers `true` ahead of the callee's values, so
        // it keeps one value fewer of them.
        let open_results = results == 0;
        let kept = if protected != Protected::No && !open_results {
            results - 1
        } else {
            results
        };
        let mut values = Vec::with_capacity(kept.max(count));
        let wanted = if open_results { count } else { count.min(kept) };
        for index in 0..wanted {
            let sym = match self.sym(a + index) {
                sym @ Sym::Root(_) => sym,
                _ => match self.read(at.registers, a + index)?.0 {
                    Operand::Op(op) => Sym::Op(op),
                    Operand::Carried(_) => return Err(TraceAbortReason::Divergence),
                },
            };
            values.push(sym);
        }
        let mut open = Vec::new();
        if opcode == Ok(OpCode::ReturnOpen) {
            for value in self.open.take().ok_or(TraceAbortReason::Divergence)? {
                open.push(match value {
                    Open::Op(op) => Sym::Op(op),
                    Open::Root(slot) => Sym::Root(slot),
                    Open::Char(_) => return Err(TraceAbortReason::UnsupportedOperand),
                });
            }
        }
        let yes = (protected != Protected::No).then(|| {
            self.emit(
                pc,
                RecKind::Constant(TraceScalar::Boolean(true)),
                Some(TraceType::Boolean),
            )
        });
        if open_results {
            // The caller's call has open results: all the values go on to
            // the instruction that takes them.
            let mut all = Vec::with_capacity(values.len() + open.len() + 1);
            all.extend(yes.map(Open::Op));
            for sym in values.iter().chain(&open) {
                all.push(match *sym {
                    Sym::Op(op) => Open::Op(op),
                    Sym::Root(slot) => Open::Root(slot),
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                });
            }
            self.open = Some(all);
        } else {
            // A fixed count of the open values may follow the fixed ones.
            for sym in open {
                if values.len() >= kept {
                    break;
                }
                values.push(sym);
            }
            if values.len() < kept {
                let nil = self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Nil),
                    Some(TraceType::Nil),
                );
                values.resize(kept, Sym::Op(nil));
            }
            values.splice(..0, yes.map(Sym::Op));
        }
        let callee = self.frames.pop().expect("an inlined frame returns");
        if callee.owns_reexec {
            self.reexec = None;
        }
        if self.frames.is_empty() {
            self.call_site = None;
        }
        // The caller is the running frame again.
        if results > 0 {
            self.check_register(base + results - 1)?;
            for (index, value) in values.into_iter().enumerate() {
                self.write(base + index, value);
            }
        }
        Ok(Flow::Leave {
            resume: callee.call_pc + 1,
        })
    }

    /// A compare-and-branch test and the `Jump` word after it, recorded as
    /// the comparison, one guard on its outcome, and the path the outcome
    /// takes: the jump's target when the outcome equals `expected`, else the
    /// instruction after the word. The word is not hooked on its own, so this
    /// follows it.
    fn record_test(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let registers = at.registers;
        let opcode = instruction
            .opcode()
            .map_err(|_| TraceAbortReason::UnsupportedOpcode)?;
        let comparison_opcode = opcode
            .comparison_form()
            .ok_or(TraceAbortReason::UnsupportedOpcode)?;
        let comparison = match comparison_opcode {
            OpCode::Equal => Comparison::Equal,
            OpCode::LessThan => Comparison::LessThan,
            _ => Comparison::LessEqual,
        };
        let word = *at
            .chunk
            .code
            .get(pc + 1)
            .ok_or(TraceAbortReason::Divergence)?;
        if word.opcode() != Ok(OpCode::Jump) {
            return Err(TraceAbortReason::Divergence);
        }
        let target = self.target(pc + 1, word.jump_offset())?;
        let a = usize::from(instruction.a());
        let expected = instruction.test_expected();
        let constant_outcome = if comparison == Comparison::Equal {
            match self.string_constants_equal(at, instruction, opcode) {
                Some(outcome) => Some(outcome),
                None => self.slice_equals_test(at, instruction, opcode)?,
            }
        } else {
            None
        };
        if let Some(outcome) = constant_outcome {
            // Two string constants, or a substring against one that its
            // guards decided: the outcome is the recording's.
            return if outcome == expected {
                self.check_branch(pc + 1, target)?;
                Ok(target)
            } else {
                Ok(pc + 2)
            };
        }
        // The live operands in comparison order, with the recorded ones.
        let (value, outcome) = if let Some(Constant::String(text)) = at
            .chunk
            .constants
            .get(usize::from(instruction.b()))
            .filter(|_| opcode.is_test_constant() && comparison == Comparison::Equal)
        {
            // A string constant is a root string: the comparison with it
            // compares bytes (see `record_comparison`), or it is false for
            // a register of another type.
            let slot = self.root_slot(text.as_bytes())?;
            let constant = self.root_operand(pc, slot);
            let live_constant = Value::String(text.clone());
            let register = self.read_compared(pc, registers, a)?;
            let outcome = self.live_comparison(comparison_opcode, &registers[a], &live_constant)?;
            let value = self.record_comparison(pc, comparison, register, constant, None)?;
            (value, outcome)
        } else if opcode.is_test_constant() {
            // A constant operand records as the constant loaded into a
            // register, then the register form.
            let scalar = self.constant_at(at.chunk, usize::from(instruction.b()))?;
            // An integer compared with a number of run-time subtype is
            // compared as the float it is, which is exact.
            let compared = match scalar {
                TraceScalar::Integer(value)
                    if self.number_pair(a).is_some() && value.abs() <= 1 << 53 =>
                {
                    #[allow(clippy::cast_precision_loss)]
                    TraceScalar::Float(value as f64)
                }
                other => other,
            };
            let constant =
                Operand::Op(self.emit(pc, RecKind::Constant(compared), Some(compared.ty())));
            let live_constant = match scalar {
                TraceScalar::Nil => Value::Nil,
                TraceScalar::Boolean(flag) => Value::Boolean(flag),
                TraceScalar::Integer(value) => Value::Integer(value),
                TraceScalar::Float(value) => Value::Number(value),
                TraceScalar::Int64(value) => Value::Int64(value),
                TraceScalar::UInt64(value) => Value::UInt64(value),
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            };
            let register = self.read_for_comparison(registers, a)?;
            let register_first = !instruction.test_constant_left();
            let (left, right, live_left, live_right) = if register_first {
                (
                    register,
                    (constant, compared.ty()),
                    &registers[a],
                    &live_constant,
                )
            } else {
                (
                    (constant, compared.ty()),
                    register,
                    &live_constant,
                    &registers[a],
                )
            };
            let outcome = self.live_comparison(comparison_opcode, live_left, live_right)?;
            let value = self.record_comparison(pc, comparison, left, right, None)?;
            (value, outcome)
        } else {
            let b = usize::from(instruction.b());
            let (left, right) = if comparison == Comparison::Equal {
                (
                    self.read_compared(pc, registers, a)?,
                    self.read_compared(pc, registers, b)?,
                )
            } else {
                (
                    self.read_for_comparison(registers, a)?,
                    self.read_for_comparison(registers, b)?,
                )
            };
            let outcome = self.live_comparison(comparison_opcode, &registers[a], &registers[b])?;
            let value = self.record_comparison(
                pc,
                comparison,
                left,
                right,
                Some((&registers[a], &registers[b])),
            )?;
            (value, outcome)
        };
        let taken = outcome == expected;
        let next = if taken {
            self.check_branch(pc + 1, target)?;
            target
        } else {
            pc + 2
        };
        // Two pure arms that join run both, and a select picks.
        let successors = if expected {
            (target, pc + 2)
        } else {
            (pc + 2, target)
        };
        if let Some(next) = self.value_select(at, Operand::Op(value), successors, outcome) {
            return Ok(next);
        }
        // A branch writes nothing: the post-state is the current state,
        // resumed at the successor the recorded direction did not take. The
        // snapshot belongs to the test, where the guard sits.
        let alternative = if taken { pc + 2 } else { target };
        let snapshot = self.branch_snapshot(pc, alternative);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(value),
                expected: outcome,
                snapshot,
            },
            None,
        );
        Ok(next)
    }

    /// The outcome of the comparison `opcode` on live operands, for a
    /// comparison the recorder follows: no metamethod can run for it.
    fn live_comparison(
        &self,
        opcode: OpCode,
        left: &Value,
        right: &Value,
    ) -> Result<bool, TraceAbortReason> {
        match numeric_scalar_result(opcode, left, right, self.float_arith) {
            Ok(Some(Value::Boolean(outcome))) => Ok(outcome),
            _ if opcode == OpCode::Equal => Ok(Runtime::raw_equal(left, right)),
            _ => Err(TraceAbortReason::UnsupportedOperand),
        }
    }

    /// The snapshot of a guard on a table read at `pc`: inside a call that
    /// has stored nothing yet, the one before the call, so the interpreter
    /// makes the call itself and the read that went the other way gets a side
    /// trace in the root frame (see [`Self::call_site`]). Otherwise the one
    /// before the read.
    fn read_snapshot(&mut self, pc: usize) -> u32 {
        match self.starts_call_over() {
            // No effect may have been committed (a store, the generator of
            // `math.random`): the call would commit it twice.
            Some(before_call)
                if !self.ops[self.call_ops..]
                    .iter()
                    .any(|op| op.kind.commits_effect()) =>
            {
                before_call
            }
            _ => self.snapshot(SnapshotPhase::Before { pc }, &[]),
        }
    }

    /// The snapshot before the outermost inlined call, when a guard inside
    /// the call may leave through it: the call has stored nothing yet, so the
    /// interpreter can make it again, and the other arm gets a side that
    /// starts at the call, in the root frame.
    ///
    /// A call that wrote an upvalue cell, or ran more than
    /// [`BEFORE_CALL_MAX_PREFIX`] guest instructions, before the guard leaves
    /// through a snapshot of its frames instead, when a side can start in
    /// them (see "Sides from exits inside inlined frames" in the module
    /// docs). Starting the call over is exact there too (the snapshot names
    /// the cell's value before the call), but the exit would run the prefix
    /// again in the interpreter, the cell write included, and the side would
    /// record it once more for each guard of the call: a body of 27 inlined
    /// calls, each with a data-dependent branch after an upvalue write, gave
    /// every side the whole body again.
    fn starts_call_over(&self) -> Option<u32> {
        let (before_call, stores) = self.call_site?;
        if self.frames.is_empty() || stores != self.stores || self.rebuild_callee_frames {
            return None;
        }
        let long = self.guest.saturating_sub(self.call_guest) > BEFORE_CALL_MAX_PREFIX
            || self.cell_writes != self.call_cell_writes;
        (!long || !self.frames_seedable()).then_some(before_call)
    }

    /// Whether a side could start inside the inlined frames now on the
    /// stack (the rules of `ExitRestore::seedable`): at most
    /// [`MAX_SEED_FRAMES`](super::policy::MAX_SEED_FRAMES) plain calls of
    /// the frame below, each of a fixed result count, of the pinned closure,
    /// without varargs or tail calls, in a loop trace's root frame.
    fn frames_seedable(&self) -> bool {
        self.frames.len() <= super::policy::MAX_SEED_FRAMES
            && self.function.is_none()
            && self.activation.is_none()
            && self.reexec.is_none()
            && self.loop_tail.is_none()
            && self.frames.iter().all(|frame| {
                frame.entry == Entry::Call
                    && frame.protected == Protected::No
                    && frame.results > 0
                    && frame.tail_calls == 0
                    && frame.dynamic.is_none()
                    && !frame.handler
                    && !frame.owns_reexec
                    && !frame.chunk.is_vararg
                    && matches!(frame.after, Pending::Nothing)
            })
    }

    /// The snapshot of a branch guard at `pc`, resuming at `alternative`.
    fn branch_snapshot(&mut self, pc: usize, alternative: usize) -> u32 {
        match self.starts_call_over() {
            // Inside a call that has stored nothing yet, start the call
            // over instead of rebuilding its frames.
            Some(before_call) => before_call,
            _ => self.snapshot(
                SnapshotPhase::After {
                    pc,
                    successor: alternative,
                },
                &[],
            ),
        }
    }

    /// `JumpIfFalse`: a boolean condition guards its direction, a nil or
    /// float condition has a fixed direction.
    fn jump_if_false(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let target = self.target(pc, short_offset(instruction))?;
        let (condition, ty) = self.read(at.registers, usize::from(instruction.a()))?;
        let truthy = at.registers[usize::from(instruction.a())].is_truthy();
        let next = if truthy { pc + 1 } else { target };
        if !truthy {
            self.check_branch(pc, target)?;
        }
        if let Operand::Carried(register) = condition
            && ty != TraceType::Boolean
        {
            self.condition_reads.push(register);
        }
        if ty == TraceType::Boolean {
            // Two pure arms that join run both, and a select picks.
            if let Some(next) = self.value_select(at, condition, (pc + 1, target), truthy) {
                return Ok(next);
            }
            // A branch writes nothing: the post-state is the current state,
            // resumed at the successor the recorded direction did not take.
            let alternative = if truthy { target } else { pc + 1 };
            let snapshot = match self.starts_call_over() {
                // Inside a call that has stored nothing yet, start the call
                // over instead of rebuilding its frames.
                Some(before_call) => before_call,
                _ => self.snapshot(
                    SnapshotPhase::After {
                        pc,
                        successor: alternative,
                    },
                    &[],
                ),
            };
            self.emit(
                pc,
                RecKind::GuardBranch {
                    condition,
                    expected: truthy,
                    snapshot,
                },
                None,
            );
        }
        Ok(next)
    }

    /// `NumericForPrep`. Three integers start an integer loop that enters
    /// the body directly. Anything else is a float loop (the interpreter
    /// converts every operand to a float), whose prep jumps to the latch.
    fn numeric_for_prep(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let base = usize::from(instruction.a());
        let target = self.target(pc, short_offset(instruction))?;
        self.check_register(base + 3)?;
        let control = self.read_number(at.registers, base)?;
        let limit = self.read_number(at.registers, base + 1)?;
        let step = self.read_number(at.registers, base + 2)?;
        if let (
            Live::Integer(control_value),
            Live::Integer(limit_value),
            Live::Integer(step_value),
        ) = (control.live, limit.live, step.live)
        {
            if step_value == 0 {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            let positive = step_value > 0;
            let continues = if positive {
                control_value <= limit_value
            } else {
                control_value >= limit_value
            };
            self.guard_step_sign(pc, step.operand, positive);
            if !positive {
                // `0 < step` failing does not exclude a zero step.
                self.guard_nonzero(pc, step.operand);
            }
            let test = self.emit(
                pc,
                RecKind::Compare(
                    Comparison::LessEqual,
                    if positive {
                        control.operand
                    } else {
                        limit.operand
                    },
                    if positive {
                        limit.operand
                    } else {
                        control.operand
                    },
                ),
                Some(TraceType::Boolean),
            );
            let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
            self.emit(
                pc,
                RecKind::GuardBranch {
                    condition: Operand::Op(test),
                    expected: continues,
                    snapshot: before,
                },
                None,
            );
            if continues {
                // The body runs once before its latch advances the control.
                let visible =
                    self.emit(pc, RecKind::Move(control.operand), Some(TraceType::Integer));
                self.write(base + 3, visible);
                return Ok(pc + 1);
            }
            // An empty range skips the latch too.
            return Ok(target + 1);
        }
        // A float loop, whatever mix of numbers started it. The interpreter
        // writes all three registers back as floats.
        let control_float = self.float_operand(pc, control);
        let limit_float = self.float_operand(pc, limit);
        let step_float = self.float_operand(pc, step);
        for (offset, number, converted) in [(1, limit, limit_float), (2, step, step_float)] {
            if number.ty() == TraceType::Integer {
                let Operand::Op(index) = converted else {
                    unreachable!("a conversion defines an operation");
                };
                self.write(base + offset, index);
            }
        }
        let value = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Subtract, control_float, step_float),
            Some(TraceType::Float),
        );
        self.write(base, value);
        Ok(target)
    }

    /// The iterations of the loop whose latch at `pc` continues with
    /// `remaining` left in its run that the program will still run, when they
    /// can be known: the frame runs a top-level chunk of the main thread,
    /// which runs once ([`HookPoint::top_level`]; a function a host calls
    /// again and again does not qualify), and every loop around this one in
    /// the chunk is a numeric for loop. Each of those starts this loop again on each of its
    /// remaining iterations, for as many iterations as a run of this loop
    /// has, which is estimated from the counters ([`run_trips`]). `None` when
    /// any of it is unknown.
    fn remaining_work(
        &self,
        at: &HookPoint<'_>,
        pc: usize,
        remaining: f64,
        counters: (Live, Live, Live),
    ) -> Option<f64> {
        if !at.top_level || self.coroutine.is_some() {
            return None;
        }
        let around = nest::enclosing_loops(
            &self.chunk,
            LoopSpan {
                header: self.header,
                latch: pc,
            },
        );
        let mut work = remaining;
        let mut per_run = run_trips(counters)?;
        for (level, outer) in around.iter().enumerate() {
            let instruction = self.chunk.code.get(outer.latch)?;
            if instruction.opcode() != Ok(OpCode::NumericForLoop) {
                return None;
            }
            let base = usize::from(instruction.a());
            let counter = |register: usize| match at.registers.get(register)? {
                Value::Integer(value) => Some(Live::Integer(*value)),
                Value::Number(value) => Some(Live::Float(*value)),
                _ => None,
            };
            let outer_counters = (counter(base)?, counter(base + 1)?, counter(base + 2)?);
            let (control, limit, step) = outer_counters;
            work += per_run * remaining_trips(control, limit, step);
            if level + 1 < around.len() {
                per_run *= run_trips(outer_counters)?;
            }
        }
        Some(work)
    }

    /// The nest around the short numeric for loop that closes at `pc`: the
    /// loops around it up to the first that is not short, when each of them
    /// is a numeric for loop with nothing else inside, the frame is not in a
    /// coroutine and the whole nest unrolled below that loop fits one
    /// recording. The counters of those loops are the frame's registers, so
    /// a loop recording in a function frame qualifies like one in a
    /// top-level chunk: the iterations the target has left are those of this
    /// call, and the repay check in `TracePolicy::defers_to_outer` judges
    /// them. A function-entry recording, or a side of a function trace
    /// (`function` is set), does not qualify: its loops close no root loop to
    /// give way to. `None` otherwise.
    fn nest_plan(
        &self,
        at: &HookPoint<'_>,
        pc: usize,
        counters: (Live, Live, Live),
    ) -> Option<NestPlan> {
        if self.coroutine.is_some() || self.function.is_some() {
            return None;
        }
        let root = LoopSpan {
            header: self.header,
            latch: pc,
        };
        let around = nest::enclosing_loops(&self.chunk, root);
        if around.is_empty() {
            return None;
        }
        // `run_trips` counts from zero: a loop from one has one fewer.
        let root_trips = (run_trips(counters)? - 1.0).max(1.0);
        #[allow(clippy::cast_precision_loss)]
        let span = |span: LoopSpan| (span.latch - span.header + 1) as f64;
        // What one run of the loop below, unrolled, costs in guest
        // instructions and in backward branches.
        let mut guest = span(root) * root_trips;
        let mut backedges = root_trips - 1.0;
        let mut below = root;
        for (level, outer) in around.iter().enumerate() {
            // Nothing but the loops of this chain inside it.
            if nest::inner_loops(&self.chunk, *outer).len() != level + 1 {
                return None;
            }
            let instruction = self.chunk.code.get(outer.latch)?;
            if instruction.opcode() != Ok(OpCode::NumericForLoop) {
                return None;
            }
            let base = usize::from(instruction.a());
            let counter = |register: usize| match at.registers.get(register)? {
                Value::Integer(value) => Some(Live::Integer(*value)),
                Value::Number(value) => Some(Live::Float(*value)),
                _ => None,
            };
            let outer_counters = (counter(base)?, counter(base + 1)?, counter(base + 2)?);
            let (control, limit, step) = outer_counters;
            let own = span(*outer) - span(below);
            let trips = (run_trips(outer_counters)? - 1.0).max(1.0);
            if trips > MAX_UNROLLED_LOOP_TRIPS {
                // A loop body that calls is bound by the call, not by its
                // loop (a string library call spends thousands of
                // instructions where the loop spends tens), and every
                // unrolled trip would be another call site to compile. The
                // calls of the loop a recording starts from are made once
                // per iteration of it, whatever the short loops do.
                let calls = self.chunk.code[below.header..=below.latch]
                    .iter()
                    .any(|instruction| {
                        matches!(
                            instruction.opcode(),
                            Ok(OpCode::Call
                                | OpCode::CallOne
                                | OpCode::CallOpen
                                | OpCode::CallVararg
                                | OpCode::TailCall
                                | OpCode::TailCallOpen
                                | OpCode::TailCallVararg)
                        )
                    });
                if calls {
                    return None;
                }
                // The loop a recording starts from.
                return Some(NestPlan {
                    target: *outer,
                    target_remaining: remaining_trips(control, limit, step),
                    root_trips,
                    guest_instructions: own + guest,
                    backedges,
                });
            }
            guest = trips * (own + guest);
            backedges = trips * backedges + (trips - 1.0);
            below = *outer;
        }
        None
    }

    /// `NumericForLoop`: advance, test, then jump back or fall through.
    ///
    /// The sign of the step picks the comparison, so it is guarded first,
    /// with a `Before` snapshot so the interpreter re-executes the latch from
    /// its untouched state. An integer loop also guards that the advance does
    /// not overflow. The interpreter ends the loop without writing anything
    /// when `checked_add` fails, which is a `Before` exit as well. The test
    /// is `control > MAX - step` for a positive step and
    /// `control < MIN - step` for a negative one, and neither bound can wrap.
    fn numeric_for_loop(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let base = usize::from(instruction.a());
        let target = self.target(pc, short_offset(instruction))?;
        self.check_register(base + 3)?;
        let control = self.read_number(at.registers, base)?;
        let limit = self.read_number(at.registers, base + 1)?;
        let step = self.read_number(at.registers, base + 2)?;
        let ty = control.ty();
        if limit.ty() != ty || step.ty() != ty {
            // The interpreter raises an error for the odd one out.
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let (positive, continues) = for_direction(control.live, limit.live, step.live)?;
        if continues {
            let remaining = remaining_trips(control.live, limit.live, step.live);
            let counters = [control.operand, limit.operand, step.operand];
            if self.links_loop(pc, target) {
                // The side rejoins the loop side it descends from.
            } else if self.wants_loop_side_at(
                pc,
                target,
                LoopKind::Numeric {
                    remaining,
                    counters,
                },
            ) {
                // The recording ends in front of the loop's next trip: the
                // loop is a side of its own.
                self.enter_loop = Some((target, pc));
                self.enter_depth = self.frames.len();
            } else {
                match self.check_branch(pc, target) {
                    Err(TraceAbortReason::InnerLoop)
                        if self.may_unroll_loop(remaining, counters) =>
                    {
                        self.unrolled_loops += 1;
                    }
                    // A loop the policy found short leaves early (a `break`)
                    // whatever its range: it unrolls the trips this
                    // iteration takes, as a `while` loop does.
                    Err(TraceAbortReason::InnerLoop) if self.may_unroll_short(pc, target) => {
                        self.unrolled_loops += 1;
                        self.unrolled_whiles += 1;
                    }
                    Err(TraceAbortReason::InnerLoop) => {
                        self.met_long_loop = remaining > MAX_UNROLLED_LOOP_TRIPS
                            || (self.unroll_loops && self.is_short_loop(target));
                        return Err(TraceAbortReason::InnerLoop);
                    }
                    other => other?,
                }
            }
            if target == self.header && self.side.is_none() && self.frames.is_empty() {
                let remaining = remaining_trips(control.live, limit.live, step.live);
                self.remaining_trips = Some(remaining);
                self.run_trips = run_trips((control.live, limit.live, step.live));
                self.remaining_work =
                    self.remaining_work(at, pc, remaining, (control.live, limit.live, step.live));
                self.nest_plan = self.nest_plan(at, pc, (control.live, limit.live, step.live));
            }
        } else if self.is_header(target) && !self.outer_side {
            // The loop ended while recording. Nothing closes this iteration.
            return Err(TraceAbortReason::LoopNotClosed);
        }
        self.guard_step_sign(pc, step.operand, positive);
        if ty == TraceType::Integer {
            self.guard_headroom(pc, control.operand, step.operand, positive);
        }

        let next_control = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Add, control.operand, step.operand),
            Some(ty),
        );
        let (left, right) = if positive {
            (Operand::Op(next_control), limit.operand)
        } else {
            (limit.operand, Operand::Op(next_control))
        };
        let test = self.emit(
            pc,
            RecKind::Compare(Comparison::LessEqual, left, right),
            Some(TraceType::Boolean),
        );
        // The failure outcome is the one the recording did not take. Both
        // advance the control register. Only continuing sets the visible
        // variable.
        let (failure, overrides): (SnapshotPhase, Vec<(usize, Sym)>) = if continues {
            (
                SnapshotPhase::After {
                    pc,
                    successor: pc + 1,
                },
                vec![(base, Sym::Op(next_control))],
            )
        } else {
            (
                SnapshotPhase::After {
                    pc,
                    successor: target,
                },
                vec![
                    (base, Sym::Op(next_control)),
                    (base + 3, Sym::Op(next_control)),
                ],
            )
        };
        let after = self.snapshot(failure, &overrides);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(test),
                expected: continues,
                snapshot: after,
            },
            None,
        );
        self.write(base, Sym::Op(next_control));
        if continues {
            self.write(base + 3, Sym::Op(next_control));
            Ok(target)
        } else {
            Ok(pc + 1)
        }
    }

    /// Guards that `control + step` does not overflow, with a `Before`
    /// snapshot. See [`Self::numeric_for_loop`].
    fn guard_headroom(&mut self, pc: usize, control: Operand, step: Operand, positive: bool) {
        let bound = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Integer(if positive {
                i64::MAX
            } else {
                i64::MIN
            })),
            Some(TraceType::Integer),
        );
        let headroom = self.emit(
            pc,
            RecKind::Arithmetic(ArithmeticOp::Subtract, Operand::Op(bound), step),
            Some(TraceType::Integer),
        );
        let (left, right) = if positive {
            (Operand::Op(headroom), control)
        } else {
            (control, Operand::Op(headroom))
        };
        let overflows = self.emit(
            pc,
            RecKind::Compare(Comparison::LessThan, left, right),
            Some(TraceType::Boolean),
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(overflows),
                expected: false,
                snapshot: before,
            },
            None,
        );
    }

    /// Guards the sign of a for loop's step: `0 < step` must equal
    /// `positive`. Failing it re-executes the instruction from its
    /// pre-state.
    fn guard_step_sign(&mut self, pc: usize, step: Operand, positive: bool) {
        let zero = self.emit(
            pc,
            RecKind::Constant(self.zero(step)),
            Some(self.operand_type(step)),
        );
        let sign = self.emit(
            pc,
            RecKind::Compare(Comparison::LessThan, Operand::Op(zero), step),
            Some(TraceType::Boolean),
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(sign),
                expected: positive,
                snapshot: before,
            },
            None,
        );
    }

    /// Guards an integer divisor against zero, leaving through a `Before`
    /// snapshot so the interpreter runs the instruction and raises its error.
    fn guard_nonzero(&mut self, pc: usize, divisor: Operand) {
        let zero = self.emit(
            pc,
            RecKind::Constant(TraceScalar::Integer(0)),
            Some(TraceType::Integer),
        );
        let is_zero = self.emit(
            pc,
            RecKind::Compare(Comparison::Equal, divisor, Operand::Op(zero)),
            Some(TraceType::Boolean),
        );
        let before = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::GuardBranch {
                condition: Operand::Op(is_zero),
                expected: false,
                snapshot: before,
            },
            None,
        );
    }

    /// The type of an operand a recorded operation or a register defines.
    fn operand_type(&self, operand: Operand) -> TraceType {
        match operand {
            Operand::Op(index) => self.ops[index as usize]
                .ty
                .expect("an operand operation defines a value"),
            Operand::Carried(register) => self.entry[register]
                .trace_type()
                .expect("a read operand has a trace type"),
        }
    }

    /// Zero in the number type of `operand`.
    fn zero(&self, operand: Operand) -> TraceScalar {
        if self.operand_type(operand) == TraceType::Integer {
            TraceScalar::Integer(0)
        } else {
            TraceScalar::Float(0.0)
        }
    }

    /// `number` as a float operand, converting an integer.
    fn float_operand(&mut self, pc: usize, number: Number) -> Operand {
        if number.ty() == TraceType::Float {
            return number.operand;
        }
        Operand::Op(self.emit(
            pc,
            RecKind::IntToFloat(number.operand),
            Some(TraceType::Float),
        ))
    }

    /// The 64-bit type an operation on `types` works in: unsigned when any
    /// operand is `uint64_t`, as C's usual arithmetic conversions have it.
    fn wide_target(types: &[TraceType]) -> TraceType {
        if types.contains(&TraceType::UInt64) {
            TraceType::UInt64
        } else {
            TraceType::Int64
        }
    }

    /// `operand`, of type `ty` (a float or a 64-bit integer), as a `target`
    /// 64-bit integer. A float truncates, the other cdata keeps its bits.
    // `to_wide` is the IR's name for this conversion; the method emits, so
    // it holds `&mut self`.
    #[allow(clippy::wrong_self_convention)]
    fn to_wide(
        &mut self,
        pc: usize,
        operand: Operand,
        ty: TraceType,
        target: TraceType,
    ) -> Operand {
        if ty == target {
            return operand;
        }
        Operand::Op(self.emit(
            pc,
            RecKind::ToWide(operand, target, WideConversion::Truncate),
            Some(target),
        ))
    }

    /// A 64-bit integer constant of type `target`.
    fn wide_constant(&mut self, pc: usize, target: TraceType, bits: u64) -> Operand {
        let scalar = if target == TraceType::UInt64 {
            TraceScalar::UInt64(bits)
        } else {
            TraceScalar::Int64(bits.cast_signed())
        };
        Operand::Op(self.emit(pc, RecKind::Constant(scalar), Some(target)))
    }

    /// An arithmetic instruction with a 64-bit integer operand. The other
    /// operand is a float or one too, and both convert to the common type
    /// (see [`oximond::runtime::cint64::common_operands`]). The operations are
    /// total, so nothing needs a guard: a zero divisor gives the same result
    /// in the interpreter.
    fn record_wide_arithmetic(
        &mut self,
        registers: &[Value],
        opcode: OpCode,
        (b, c): (usize, usize),
        constant: Option<(Operand, TraceType)>,
        pc: usize,
    ) -> Result<u32, TraceAbortReason> {
        let (left, left_ty) = self.read(registers, b)?;
        let (right, right_ty) = match constant {
            Some(constant) => constant,
            None => self.read(registers, c)?,
        };
        let convertible = |ty: TraceType| ty.is_wide() || ty == TraceType::Float;
        if !convertible(left_ty) || !convertible(right_ty) {
            // A string, or a value the interpreter reports an error for.
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let op = match opcode {
            OpCode::Add => ArithmeticOp::Add,
            OpCode::Subtract => ArithmeticOp::Subtract,
            OpCode::Multiply => ArithmeticOp::Multiply,
            OpCode::Divide => ArithmeticOp::Divide,
            OpCode::Modulo => ArithmeticOp::Modulo,
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let target = Self::wide_target(&[left_ty, right_ty]);
        let left = self.to_wide(pc, left, left_ty, target);
        let right = self.to_wide(pc, right, right_ty, target);
        Ok(self.emit(pc, RecKind::Arithmetic(op, left, right), Some(target)))
    }

    /// A call of `tonumber` or a `bit` function with a 64-bit integer
    /// argument, behind the callee guard. The cases are the ones the
    /// interpreter's `wide_pair`, `wide_shift_arguments` and `tonumber`
    /// handle: a number operand converts as in arithmetic, a shift count
    /// converts as the `bit` library reads an operand, and the result has the
    /// common 64-bit type. Any other shape aborts.
    fn record_wide_native(
        &mut self,
        pc: usize,
        kind: DirectNativeKind,
        callee: Operand,
        function: NativeFunctionHandle,
        arguments: &[(Operand, TraceType)],
        results: (usize, usize),
        hoisted: bool,
    ) -> Result<Flow, TraceAbortReason> {
        let convertible = |ty: TraceType| ty.is_wide() || ty == TraceType::Float;
        if arguments.iter().any(|(_, ty)| !convertible(*ty)) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let lane = self.pin(pinned_native_identity(function.index()))?;
        let binary = |op: ArithmeticOp| (op, 2);
        let (op, wanted) = match kind {
            DirectNativeKind::BitBand => binary(ArithmeticOp::BitAnd),
            DirectNativeKind::BitBor => binary(ArithmeticOp::BitOr),
            DirectNativeKind::BitBxor => binary(ArithmeticOp::BitXor),
            DirectNativeKind::BitLshift => binary(ArithmeticOp::ShiftLeft),
            DirectNativeKind::BitRshift => binary(ArithmeticOp::ShiftRight),
            DirectNativeKind::BitArshift => binary(ArithmeticOp::ArithmeticShiftRight),
            DirectNativeKind::BitRol => binary(ArithmeticOp::RotateLeft),
            DirectNativeKind::BitRor => binary(ArithmeticOp::RotateRight),
            // The unary functions take their operation from the kind below.
            DirectNativeKind::Tonumber | DirectNativeKind::BitTobit | DirectNativeKind::BitBnot => {
                (ArithmeticOp::BitXor, 1)
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        if arguments.len() != wanted {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let shifts = matches!(
            kind,
            DirectNativeKind::BitLshift
                | DirectNativeKind::BitRshift
                | DirectNativeKind::BitArshift
                | DirectNativeKind::BitRol
                | DirectNativeKind::BitRor
        );
        if shifts && !arguments[0].1.is_wide() {
            // A float shifted by a 64-bit count is a 32-bit operation.
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if !hoisted {
            self.guard_callee(pc, callee, lane);
        } else if !self.take_hoisted(callee) {
            return Err(TraceAbortReason::Divergence);
        }
        let (first, first_ty) = arguments[0];
        let value = match kind {
            DirectNativeKind::Tonumber => {
                self.emit(pc, RecKind::IntToFloat(first), Some(TraceType::Float))
            }
            DirectNativeKind::BitTobit => {
                // The low 32 bits, sign extended, as a number: shift them to
                // the top and back.
                let first = self.to_wide(pc, first, first_ty, TraceType::Int64);
                let distance = self.wide_constant(pc, TraceType::Int64, 32);
                let high = self.emit(
                    pc,
                    RecKind::Arithmetic(ArithmeticOp::ShiftLeft, first, distance),
                    Some(TraceType::Int64),
                );
                let low = self.emit(
                    pc,
                    RecKind::Arithmetic(
                        ArithmeticOp::ArithmeticShiftRight,
                        Operand::Op(high),
                        distance,
                    ),
                    Some(TraceType::Int64),
                );
                self.emit(
                    pc,
                    RecKind::IntToFloat(Operand::Op(low)),
                    Some(TraceType::Float),
                )
            }
            DirectNativeKind::BitBnot => {
                let ones = self.wide_constant(pc, first_ty, u64::MAX);
                self.emit(
                    pc,
                    RecKind::Arithmetic(ArithmeticOp::BitXor, first, ones),
                    Some(first_ty),
                )
            }
            _ => {
                let (second, second_ty) = arguments[1];
                if shifts {
                    let count = match second_ty {
                        TraceType::Float => Operand::Op(self.emit(
                            pc,
                            RecKind::ToWide(second, TraceType::Int64, WideConversion::Bits32),
                            Some(TraceType::Int64),
                        )),
                        _ => second,
                    };
                    let count_ty = if second_ty == TraceType::Float {
                        TraceType::Int64
                    } else {
                        second_ty
                    };
                    let count = self.to_wide(pc, count, count_ty, first_ty);
                    self.emit(pc, RecKind::Arithmetic(op, first, count), Some(first_ty))
                } else {
                    let target = Self::wide_target(&[first_ty, second_ty]);
                    let left = self.to_wide(pc, first, first_ty, target);
                    let right = self.to_wide(pc, second, second_ty, target);
                    self.emit(pc, RecKind::Arithmetic(op, left, right), Some(target))
                }
            }
        };
        Ok(self.native_result(pc, value, results))
    }

    /// The chunk of the frame that is running: the root's, or the innermost
    /// inlined call's.
    fn running_chunk(&self) -> &Arc<Chunk> {
        self.frames
            .last()
            .map_or_else(|| self.root_chunk(), |frame| &frame.chunk)
    }

    /// The chunk the root frame runs now: the root chunk, or that of the
    /// activation a tail call put there.
    fn root_chunk(&self) -> &Arc<Chunk> {
        self.activation
            .as_ref()
            .map_or(&self.chunk, |activation| &activation.chunk)
    }

    /// Whether the root frame runs its own closure and no inlined call is in
    /// progress: the only place the loop header is.
    fn in_home(&self) -> bool {
        self.frames.is_empty() && self.activation.is_none()
    }

    /// The IR frame of the root frame's activation.
    fn root_frame_id(&self) -> u32 {
        self.activation
            .as_ref()
            .map_or(0, |activation| activation.frame)
    }

    /// The IR frame the running instruction belongs to.
    fn frame_id(&self) -> u32 {
        self.frames
            .last()
            .map_or_else(|| self.root_frame_id(), |frame| frame.frame)
    }

    /// Whether `pc` is this loop's header, which only the root frame has,
    /// or the loop frame of a loop side of a loop in a call.
    fn is_header(&self, pc: usize) -> bool {
        (self.in_home() || self.in_loop_frame()) && pc == self.header
    }

    /// Whether the running frame is the one a loop side of a loop in a call
    /// loops in (see "Loops in calls" in the module docs).
    fn in_loop_frame(&self) -> bool {
        self.loop_depth > 0 && self.frames.len() == self.loop_depth && !self.in_caller()
    }

    /// Notes the pc the running frame executed last.
    fn set_prev(&mut self, prev: PrevPc) {
        match self.frames.last_mut() {
            Some(frame) => frame.prev = prev,
            None => self.prev = prev,
        }
    }

    /// The symbolic value of a register of the running frame.
    fn sym(&self, register: usize) -> Sym {
        match self.frames.last() {
            Some(frame) => frame.regs[register],
            None => self.regs[register],
        }
    }

    fn check_register(&self, register: usize) -> Result<(), TraceAbortReason> {
        if register < usize::from(self.running_chunk().register_count) {
            Ok(())
        } else {
            Err(TraceAbortReason::UnsupportedOperand)
        }
    }

    /// The pc a jump at `pc` with `offset` lands on.
    fn target(&self, pc: usize, offset: i32) -> Result<usize, TraceAbortReason> {
        usize::try_from(i64::try_from(pc).unwrap_or(i64::MAX) + 1 + i64::from(offset))
            .ok()
            .filter(|target| *target < self.running_chunk().code.len())
            .ok_or(TraceAbortReason::Divergence)
    }

    /// A taken branch may only go backward to this loop's own header, in the
    /// root frame. An inlined call that loops is an inner loop. An outer side
    /// trace may also branch back to the header of a loop around the root's:
    /// its span contains the whole of the root's, so the recording stays on
    /// the path to the root's header.
    fn check_branch(&mut self, pc: usize, target: usize) -> Result<(), TraceAbortReason> {
        self.check_branch_of(pc, target, false)
    }

    /// [`Self::check_branch`], which with `unrollable` also follows the
    /// backward branch of a loop inside the recorded one, up to
    /// [`MAX_UNROLLED_BACKEDGES`] times in a recording. The loop is unrolled:
    /// each trip is recorded after the one before, and the step that ends the
    /// loop is a guard like any other, which a side trace continues for the
    /// loops of other lengths. Only the loop of a `string.gmatch` iterator is
    /// unrollable: its step is a helper that leaves when the iteration ends
    /// at another trip than the recorded one, and a few trips are all most
    /// of these loops run. A loop with more trips ends the recording as an
    /// inner loop.
    fn check_branch_of(
        &mut self,
        pc: usize,
        target: usize,
        unrollable: bool,
    ) -> Result<(), TraceAbortReason> {
        // A function trace has no loop: any backward branch is one inside it.
        if target > pc || (self.function.is_none() && self.is_header(target)) {
            return Ok(());
        }
        // In the root frame's caller, the loop that calls the root function
        // again is the one the recording follows.
        if self.running_caller() {
            return if self.caller_branch(pc, target) {
                Ok(())
            } else {
                Err(TraceAbortReason::InnerLoop)
            };
        }
        if unrollable
            && self.function.is_none()
            && self.frames.is_empty()
            && target >= self.header
            && pc <= self.latch
            && self.unrolled < MAX_UNROLLED_BACKEDGES
        {
            self.unrolled += 1;
            return Ok(());
        }
        if self.outer_side && self.frames.is_empty() && target < self.header && pc > self.latch {
            if !self.crossed.contains(&target) {
                self.crossed.push(target);
            }
            return Ok(());
        }
        // An outer side may also run through a numeric for loop beside the
        // root's, outside its span (the work after the nest): unrolled like
        // a `string.gmatch` loop, each trip's step a guard. A loop of
        // another length leaves at its step.
        if self.beyond
            && self.frames.is_empty()
            && (target > self.latch || pc < self.header)
            && self.unrolled < MAX_UNROLLED_BACKEDGES
            && self
                .running_chunk()
                .code
                .get(pc)
                .is_some_and(|instruction| instruction.opcode() == Ok(OpCode::NumericForLoop))
        {
            self.unrolled += 1;
            return Ok(());
        }
        Err(TraceAbortReason::InnerLoop)
    }

    /// Whether the loop whose closing branch at `pc` branches back to
    /// `target` is left to a loop side: the recording ends in front of the
    /// loop's next trip with a terminal exit, and the loop runs in place in
    /// a side of its own. That costs two copies of the loop's body in the
    /// tree where unrolling costs one per trip, and has no limit on the
    /// trips. `kind` is the loop's closing branch.
    ///
    /// - An outer side that went beyond the nest does so for the loops after
    ///   it, of any kind.
    /// - A root does so for a loop inside its span that no tree of its own
    ///   carries ([`Self::with_free_loops`]), and a side of such a root for
    ///   any loop inside the root's span ([`SideSeed::inner_loops`]).
    ///
    /// A numeric for loop needs [`MIN_LOOP_SIDE_TRIPS`] trips left. Inside
    /// the span it also needs more than unrolling takes, or counters that
    /// follow the root's: a shorter loop is unrolled by the retry after the
    /// `InnerLoop` abort, as before. A `while` loop that the policy found
    /// short ([`Self::with_short_loops`]) is unrolled too. A recording that
    /// counts instructions for a count hook never does: the hook's budget
    /// runs through the root's loop only.
    fn wants_loop_side_at(&self, pc: usize, target: usize, kind: LoopKind) -> bool {
        if self.ends_before_callee_loop(pc, target) {
            return true;
        }
        if self.enter_loop.is_some()
            || self.link_loop.is_some()
            || !self.frames.is_empty()
            || self.function.is_some()
            || self.counting
            || self.looping
            || target > pc
        {
            return false;
        }
        let beyond = self.loop_sides && (target > self.latch || pc < self.header);
        // A side of a root that unrolls the loops the policy found short
        // runs a variant of the iteration on which such a loop runs longer
        // than the root's recording saw: it leaves the trips to a loop side,
        // a numeric for loop with more than unrolling takes right away, a
        // `while` loop once the unrolling budget is spent.
        let short = self.side.is_some() && self.unroll_loops && self.is_short_loop(target);
        let inside = target > self.header
            && pc < self.latch
            && match self.side {
                Some(_) => self.inner_loop_sides || short,
                None => self.free_loops.contains(&target),
            };
        if !beyond && !inside {
            return false;
        }
        match kind {
            LoopKind::Numeric {
                remaining,
                counters,
            } => {
                remaining >= MIN_LOOP_SIDE_TRIPS
                    && (beyond
                        || remaining > MAX_UNROLLED_LOOP_TRIPS
                        || counters
                            .into_iter()
                            .any(|counter| self.depends_on_root_counter(counter)))
            }
            LoopKind::While => {
                !self.short_loops.contains(&target)
                    || (short && self.unrolled_whiles >= MAX_UNROLLED_WHILE_BACKEDGES)
            }
            LoopKind::Generic => true,
        }
    }

    /// Whether the backward branch at `pc` to `target` in the running
    /// inlined frame closes the loop in a call that the recording ends in
    /// front of ([`Self::with_callee_loop`]): a root's, or a side's of such a
    /// root ([`SideSeed::inner_loops`]).
    fn ends_before_callee_loop(&self, pc: usize, target: usize) -> bool {
        let Some(frame) = self.frames.last() else {
            return false;
        };
        target <= pc
            && self.enter_loop.is_none()
            && self.link_loop.is_none()
            && self.function.is_none()
            && self.activation.is_none()
            && !self.counting
            && !self.looping
            && !self.in_caller()
            && self.frames.len() <= super::policy::MAX_SEED_FRAMES
            && (self.side.is_none() || self.inner_loop_sides)
            && self
                .callee_loop
                .as_ref()
                .is_some_and(|callee| callee.is(&frame.chunk, pc))
    }

    /// The registers of a loop side's loop frame (see "Loops in calls" in
    /// the module docs) that the loop carries: live at the header, changed by
    /// the trip, and a frame input at the start, which the parameter stands
    /// for. With the frame input's operation, in the order of the
    /// operations. A changed register that held no frame input (a constant
    /// string of the roots) cannot be carried. An element the header reads
    /// again is no parameter.
    fn loop_frame_params(&self) -> Result<Vec<(usize, u32)>, TraceAbortReason> {
        if self.loop_depth == 0 {
            return Ok(Vec::new());
        }
        let frame = self
            .frames
            .last()
            .filter(|_| self.frames.len() == self.loop_depth)
            .ok_or(TraceAbortReason::Divergence)?;
        let live = observable_registers(&frame.chunk);
        let at_header = live.get(self.header);
        let mut params = Vec::new();
        for (register, start) in self.loop_frame_start.iter().enumerate() {
            let end = frame.regs.get(register).copied().unwrap_or(Sym::Carried);
            if end == *start || self.reloaded.iter().any(|(held, _)| *held == register) {
                continue;
            }
            let live = match (at_header, u8::try_from(register)) {
                (Some(set), Ok(narrow)) => register_set_contains(set, narrow),
                _ => true,
            };
            if !live {
                continue;
            }
            match start {
                Sym::Op(index)
                    if matches!(self.ops[*index as usize].kind, RecKind::FrameInput { .. })
                        && self
                            .loop_frame_start
                            .iter()
                            .filter(|other| **other == *start)
                            .count()
                            == 1 =>
                {
                    params.push((register, *index));
                }
                _ => return Err(TraceAbortReason::UnsupportedOperand),
            }
        }
        params.sort_by_key(|(_, index)| *index);
        Ok(params)
    }

    /// Where the root frame resumes at the end of the recording: the header,
    /// the instruction a side exits in front of, or, for a recording that
    /// ends inside the frames of a call (in front of a loop of the call, or
    /// a loop side of one), the instruction after the outermost call.
    fn root_resume(&self) -> usize {
        match self.frames.first() {
            Some(outermost) if self.enter_depth > 0 || self.loop_depth > 0 => outermost.call_pc + 1,
            _ => self.exit_at.unwrap_or(self.header),
        }
    }

    /// Whether the loop at `target` is one of the root frame's loops whose
    /// own root cannot pay (see [`Self::with_short_loops`]), met in the root
    /// frame.
    fn is_short_loop(&self, target: usize) -> bool {
        self.function.is_none() && self.frames.is_empty() && self.short_loops.contains(&target)
    }

    /// Whether the backward branch at `pc` of the root frame's short loop at
    /// `target` (a numeric for latch: see [`Self::is_short_loop`]) is
    /// followed as one more trip, within the budget the `while` loops share
    /// ([`MAX_UNROLLED_WHILE_BACKEDGES`]). Such a loop was given up because
    /// its runs end on the trip they start on, by a `break` or a `return`,
    /// so its range says nothing about its trips: `for dy = ytop, ybot, -1`
    /// in Mineclonia's `build_heightmap` (`mg_ersatz.lua:122`) has 81 trips
    /// ahead and breaks on the first under the ground.
    fn may_unroll_short(&self, pc: usize, target: usize) -> bool {
        self.unroll_loops
            && target <= pc
            && self.unrolled_whiles < MAX_UNROLLED_WHILE_BACKEDGES
            && self.is_short_loop(target)
    }

    /// Whether the backward branch at `pc` to `target` closes the loop of an
    /// ancestor loop side ([`SideSeed::loop_headers`]) in the root frame. The
    /// recording then ends at that loop's header, with a
    /// [`TraceTerminator::LinkLoop`] when the link binds.
    fn links_loop(&mut self, pc: usize, target: usize) -> bool {
        if self.side.is_none()
            || !self.frames.is_empty()
            || self.enter_loop.is_some()
            || self.link_loop.is_some()
        {
            return false;
        }
        let Some(position) = self
            .loop_headers
            .iter()
            .position(|held| held.header == target && held.latch == pc)
        else {
            return false;
        };
        self.link_loop = Some(position);
        true
    }

    /// Whether the numeric for loop whose latch continues with `remaining`
    /// iterations after this one is unrolled into the recording instead of
    /// ending it as an inner loop. Each trip is recorded after the one
    /// before, and the latch's test is a guard like any other, so a run of
    /// another length leaves the trace there. Only short loops qualify (at
    /// most [`MAX_UNROLLED_LOOP_TRIPS`] more trips, and
    /// [`MAX_UNROLLED_LOOP_BACKEDGES`] in a recording): a longer loop keeps
    /// a trace of its own.
    ///
    /// A loop whose counters depend on the counter of the recorded loop
    /// itself (the `j` of `for j = i + 1, n` inside `for i`) has a length
    /// that changes with every iteration of the recorded one, so the trace
    /// would leave at its latch on all but one of them: it is not unrolled.
    fn may_unroll_loop(&self, remaining: f64, counters: [Operand; 3]) -> bool {
        self.unroll_loops
            && self.function.is_none()
            && remaining <= MAX_UNROLLED_LOOP_TRIPS
            && self.unrolled_loops < MAX_UNROLLED_LOOP_BACKEDGES
            && !counters
                .into_iter()
                .any(|counter| self.depends_on_root_counter(counter))
    }

    /// Whether the backward jump at `pc` to `target` (the end of a `while`
    /// loop) is followed as one more trip of an unrolled loop. The loop's
    /// length is not known ahead, so the trips the recording sees are the
    /// ones it unrolls, up to [`MAX_UNROLLED_WHILE_BACKEDGES`] in all; a loop
    /// that runs longer ends the recording as an inner loop with its own
    /// trace. A loop in the root frame is unrolled only when its own root was
    /// retired as too short (`Site::retired_short`); until then it is the
    /// root of a nest, whose exit takes an outer side.
    /// Whether the generic for loop whose step is the `ForInCall` at `pc`
    /// is one this recording unrolls: its `ForInLoop` follows the step, and
    /// [`Self::may_unroll_while`] takes its latch.
    fn unrolls_generic_for(&self, at: &HookPoint<'_>, pc: usize) -> bool {
        at.chunk.code.get(pc + 1).is_some_and(|latch| {
            latch.opcode() == Ok(OpCode::ForInLoop)
                && super::nest::jump_target(pc + 1, *latch).is_some_and(|target| {
                    !self.is_header(target) && self.may_unroll_while(pc + 1, target)
                })
        })
    }

    /// Whether the generic `for` whose call is at `pc` is a loop other than
    /// the one the recording closes: its latch jumps to a body that is not
    /// the recording's header.
    fn ends_inner_generic_for(&self, at: &HookPoint<'_>, pc: usize) -> bool {
        at.chunk.code.get(pc + 1).is_some_and(|latch| {
            latch.opcode() == Ok(OpCode::ForInLoop)
                && super::nest::jump_target(pc + 1, *latch)
                    .is_some_and(|target| target <= pc && !self.is_header(target))
        })
    }

    fn may_unroll_while(&self, pc: usize, target: usize) -> bool {
        self.unroll_loops
            && self.function.is_none()
            && target <= pc
            && self.unrolled_whiles < MAX_UNROLLED_WHILE_BACKEDGES
            && (!self.frames.is_empty() || self.short_loops.contains(&target))
    }

    /// Whether `operand` is computed from the counter of the numeric for
    /// loop the recording closes (its control register, as the iteration
    /// found it) by arithmetic. Gives up, as `true`, on a long chain.
    fn depends_on_root_counter(&self, operand: Operand) -> bool {
        let Some(latch) = self.chunk.code.get(self.latch) else {
            return false;
        };
        if latch.opcode() != Ok(OpCode::NumericForLoop) {
            return false;
        }
        let base = usize::from(latch.a());
        let mut pending = vec![operand];
        let mut seen: Vec<u32> = Vec::new();
        while let Some(next) = pending.pop() {
            match next {
                Operand::Carried(register) => {
                    // A side starts from registers the root's iteration
                    // wrote before the guard it continues, which may well
                    // hold values computed from the counter. It cannot tell.
                    if register == base || register == base + 3 {
                        return true;
                    }
                    if self.side.is_some()
                        && nest::written_within(
                            &self.chunk,
                            LoopSpan {
                                header: self.header,
                                latch: self.latch,
                            },
                            register,
                        )
                    {
                        return true;
                    }
                }
                Operand::Op(index) => {
                    if seen.contains(&index) {
                        continue;
                    }
                    if seen.len() >= 256 {
                        return true;
                    }
                    seen.push(index);
                    match &self.ops[index as usize].kind {
                        RecKind::Move(a)
                        | RecKind::Negate(a)
                        | RecKind::IntToFloat(a)
                        | RecKind::Numeric(_, a) => pending.push(*a),
                        RecKind::Arithmetic(_, a, b) => {
                            pending.push(*a);
                            pending.push(*b);
                        }
                        _ => {}
                    }
                }
            }
        }
        false
    }

    fn constant(
        &self,
        chunk: &Chunk,
        instruction: Instruction,
    ) -> Result<TraceScalar, TraceAbortReason> {
        self.constant_at(chunk, usize::from(instruction.d()))
    }

    /// [`Self::constant`] for the constant pool entry `index`.
    fn constant_at(&self, chunk: &Chunk, index: usize) -> Result<TraceScalar, TraceAbortReason> {
        let constant = chunk
            .constants
            .get(index)
            .ok_or(TraceAbortReason::Divergence)?;
        Ok(match constant {
            Constant::Number(number) => TraceScalar::Float(*number),
            Constant::Integer(integer) if self.integers => TraceScalar::Integer(*integer),
            // Profiles without integers read integer literals as floats.
            Constant::Integer(integer) => TraceScalar::Float(integer_to_number(*integer)),
            Constant::Int64(value) => TraceScalar::Int64(*value),
            Constant::UInt64(value) => TraceScalar::UInt64(*value),
            Constant::Boolean(flag) => TraceScalar::Boolean(*flag),
            Constant::Nil => TraceScalar::Nil,
            Constant::String(_) => return Err(TraceAbortReason::UnsupportedOperand),
        })
    }

    /// The symbolic value of register `register` and its type. The register
    /// must still hold what the recorder believes, which catches a model that
    /// drifted from the interpreter.
    fn read(
        &self,
        registers: &[Value],
        register: usize,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        self.check_register(register)?;
        let mut exact = None;
        // A userdata an operation made (read out of a table): the register
        // must hold one now.
        if let Sym::Op(index) = self.sym(register)
            && self.ops[index as usize].ty == Some(TraceType::Userdata)
        {
            return if matches!(registers[register], Value::Userdata(_)) {
                Ok((Operand::Op(index), TraceType::Userdata))
            } else {
                Err(TraceAbortReason::Divergence)
            };
        }
        let (operand, kind) = match self.sym(register) {
            // An inlined call starts with no carried registers.
            Sym::Carried if !self.frames.is_empty() => {
                return Err(TraceAbortReason::Divergence);
            }
            Sym::Carried => (Operand::Carried(register), self.entry[register]),
            // A number of run-time subtype is read only by the instructions
            // that take one (see `number.rs`).
            Sym::Op(index) if self.number_tags.contains_key(&index) => {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            Sym::Op(index) => {
                let ty = self.ops[index as usize]
                    .ty
                    .expect("an operand operation defines a value");
                exact = Some(ty);
                (Operand::Op(index), Kind::from_type(ty))
            }
            Sym::Root(_) => return Err(TraceAbortReason::UnsupportedOperand),
            // Any use of the table that is not an access the recorder
            // resolves lets it escape, and an escaped table is the real one
            // the trace made.
            Sym::Virtual(id) => match self.virtuals[id as usize].real {
                Some(escaped) => {
                    exact = Some(TraceType::Table);
                    (Operand::Op(escaped.table), Kind::Table)
                }
                None => return Err(TraceAbortReason::VirtualEscape),
            },
        };
        // The kind loses a borrow: the interpreter holds a plain string.
        let ty = match exact {
            Some(ty) if ty.is_borrowed() => ty,
            _ => kind
                .trace_type()
                .ok_or(TraceAbortReason::UnsupportedOperand)?,
        };
        // A borrow is a plain string to the interpreter.
        let live = match Kind::of(&registers[register], self.integers) {
            Kind::String if kind == Kind::Borrowed => Kind::Borrowed,
            live => live,
        };
        if live != kind {
            return Err(TraceAbortReason::Divergence);
        }
        Ok((operand, ty))
    }

    /// Like [`Self::read`] for a number, also returning the live value.
    fn read_number(
        &self,
        registers: &[Value],
        register: usize,
    ) -> Result<Number, TraceAbortReason> {
        let (operand, ty) = self.read(registers, register)?;
        let live = match (&registers[register], ty) {
            (Value::Number(value), TraceType::Float) => Live::Float(*value),
            (Value::Integer(value), TraceType::Integer) => Live::Integer(*value),
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        Ok(Number { operand, live })
    }

    /// Like [`Self::read_number`] for an operand of arithmetic, which may be
    /// a string the interpreter converts. The conversion is a helper call
    /// that leaves, before the instruction, when a later string converts to
    /// another kind of number or not at all. `operator` is the index of the
    /// operation among the string metatable's arithmetic metamethods
    /// (Lua 5.4 and later), or `None` for an operation that has none and
    /// leaves strings to the interpreter.
    fn read_arithmetic(
        &mut self,
        at: &HookPoint<'_>,
        register: usize,
        operator: Option<i64>,
    ) -> Result<Number, TraceAbortReason> {
        if !matches!(at.registers.get(register), Some(Value::String(_))) {
            return self.read_number(at.registers, register);
        }
        let operator = operator.ok_or(TraceAbortReason::UnsupportedOperand)?;
        let pc = at.pc;
        let (operand, ty) = self.read(at.registers, register)?;
        if ty != TraceType::String && !ty.is_borrowed() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let mode = match self.string_number_mode {
            STRING_NUMBER_METAMETHOD_MODE => STRING_NUMBER_METAMETHOD_MODE + operator,
            mode => mode,
        };
        let runtime = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?
            .runtime;
        let (helper, live) = match string_number(runtime, mode, &at.registers[register]) {
            Some(Value::Integer(value)) => (HelperId::STRING_NUMBER_INTEGER, Live::Integer(value)),
            Some(Value::Number(value)) => (HelperId::STRING_NUMBER_FLOAT, Live::Float(value)),
            // The interpreter raises an error, or runs a metamethod.
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let mode = self.scalar(pc, TraceScalar::Integer(mode));
        let call = self.emit_helper_call(
            pc,
            helper,
            &[(Operand::Op(mode), TraceType::Integer), (operand, ty)],
        );
        Ok(Number {
            operand: Operand::Op(call),
            live,
        })
    }

    /// The root slot holding the string `text`, adding it on first use.
    fn root_slot(&mut self, text: &[u8]) -> Result<u32, TraceAbortReason> {
        let position = match self
            .root_constants
            .iter()
            .position(|known| **known == *text)
        {
            Some(position) => position,
            // A side grows the root bank with its own strings: the tree
            // takes them in when the side joins (see `take_queued_tree`).
            None if self.root_constants.len() >= MAX_TRACE_ROOTS => {
                return Err(limit("root constants"));
            }
            None => {
                self.root_constants.push(Arc::from(text));
                self.root_constants.len() - 1
            }
        };
        Ok(u32::try_from(position).expect("root slots fit u32"))
    }

    /// `GetTabUp` of a side's inlined callee: the read of a global through
    /// the callee's own environment, which the side cannot bind a table
    /// receiver for. It reads the global as a `ClosureGlobal` slot, which the
    /// root's guards cover like the callee's cells.
    fn record_side_global(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let a = usize::from(instruction.a());
        self.check_register(a)?;
        let TableKey::Field(_) = Self::field_key(at, instruction.c())? else {
            return Err(TraceAbortReason::UnsupportedOperand);
        };
        let (cell, runtime) = Self::upvalue_cell(at, u16::from(instruction.b()))?;
        let surroundings = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        // Only the callee's environment: another table is a field read.
        if !matches!(runtime.cell_get(cell), Ok(Value::Table(table)) if table == surroundings.environment)
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let slot = self.global_slot(u32::from(instruction.c()));
        let external = self.external(at, slot)?;
        let (source, ty) = self.read_external(at, external)?;
        let value = self.emit(at.pc, RecKind::Move(source), Some(ty));
        self.write(a, value);
        Ok(at.pc + 1)
    }

    /// `GetUpvalue` in a frame that runs whichever closure of a prototype the
    /// call had: the cell is read through the live closure by a helper that
    /// leaves, before the instruction, when the cell holds another type than
    /// the one the recording saw. The cell may be open (aliasing a stack slot
    /// of an enclosing frame) or closed: the runtime's own read tells.
    fn record_dynamic_get_upvalue(
        &mut self,
        at: &HookPoint<'_>,
        a: usize,
        index: u16,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let closure = self.dynamic_closure().ok_or(TraceAbortReason::Divergence)?;
        let (cell, runtime) = Self::upvalue_cell(at, index)?;
        let live = runtime
            .cell_get(cell)
            .map_err(|_| TraceAbortReason::Divergence)?;
        let helper = match Kind::of(&live, self.integers) {
            Kind::Integer => HelperId::CELL_GET_INTEGER,
            Kind::Float => HelperId::CELL_GET_FLOAT,
            Kind::Boolean => HelperId::CELL_GET_BOOLEAN,
            Kind::Table => HelperId::CELL_GET_TABLE,
            // A function comes back as its lane, which the guard of its call
            // compares with the lane of the callee it pins. The function is
            // pinned here, so a native has its lane in the bank when the
            // helper looks for it: `fn` in the `next_integer` closures of
            // Mineclonia's xoroshiro (random.lua:943), `bor` beside it.
            Kind::Closure => {
                let identity = match &live {
                    Value::Closure(handle) => handle.native_identity(),
                    Value::NativeFunction(function) => pinned_native_identity(function.index()),
                    _ => {
                        self.note_callable(&live);
                        return Err(TraceAbortReason::UnsupportedOperand);
                    }
                };
                self.pin(identity)?;
                HelperId::CELL_GET_CALLABLE
            }
            // A string comes back in a lane of the run's bank, a different
            // string in each closure: the `id` of Mineclonia's concentric
            // ring `chunk_test` (structures.lua:404), which indexes
            // `level.stronghold_starts` with it.
            Kind::String => HelperId::CELL_GET_STRING,
            _ => {
                self.note_callable(&live);
                return Err(TraceAbortReason::UnsupportedOperand);
            }
        };
        let instance = helper == HelperId::CELL_GET_CALLABLE
            && matches!(
                at.chunk
                    .upvalues
                    .get(usize::from(index))
                    .map(|upvalue| upvalue.source),
                Some(UpvalueSource::ParentRegister(_))
            );
        if let Some(known) = self.known_cell_read(closure, helper, index) {
            self.write(a, known);
            return Ok(pc + 1);
        }
        let index = self.scalar(pc, TraceScalar::Integer(i64::from(index)));
        let value = self.emit_helper_call(
            pc,
            helper,
            &[
                (Operand::Op(closure), TraceType::Closure),
                (Operand::Op(index), TraceType::Integer),
            ],
        );
        if instance {
            self.instance_callees.push(value);
        }
        self.write(a, value);
        Ok(pc + 1)
    }

    /// The earlier read of upvalue `index` of the frame's live closure
    /// `closure` (the operation that holds it) by `helper`, when the cell
    /// still holds what that read gave: nothing recorded since can have
    /// written a cell. `object.v = object.v + 1; return object.v` reads the
    /// cell of `object` three times, and each read was a check of the
    /// closure's row and a guard with a snapshot of its own.
    ///
    /// A cell changes under a run in two ways: a write through a live
    /// closure ([`HelperId::CELL_SET`]), and Lua or host code the trace
    /// calls out to. A cell the tree writes as a window slot never meets a
    /// read through a live closure: the run zeroes the row of a closure that
    /// shares one (see [`Self::shares_window_cell`]). So the search goes back
    /// over the operations that do neither and stops at the first that might:
    /// a helper that writes the heap, advances private state or resumes, a
    /// host call, a self call, and any operation not listed here. A table
    /// store is listed, since a cell is no table slot. Only the reads of a
    /// number, a boolean or a table are shared: a string read takes a lane of
    /// its own and a callable read is noted per operation
    /// (`instance_callees`).
    fn known_cell_read(&self, closure: u32, helper: HelperId, index: u16) -> Option<u32> {
        if !matches!(
            helper,
            HelperId::CELL_GET_INTEGER
                | HelperId::CELL_GET_FLOAT
                | HelperId::CELL_GET_BOOLEAN
                | HelperId::CELL_GET_TABLE
        ) {
            return None;
        }
        let wanted = TraceScalar::Integer(i64::from(index));
        let first = closure as usize + 1;
        for (at, op) in self.ops.iter().enumerate().skip(first).rev() {
            match &op.kind {
                RecKind::CallHelper {
                    helper: other,
                    args,
                    ..
                } => {
                    let read = other.is_cell_access() && *other != HelperId::CELL_SET;
                    if read {
                        if *other == helper
                            && let [Operand::Op(held), Operand::Op(position)] = args.as_slice()
                            && *held == closure
                            && matches!(
                                &self.ops[*position as usize].kind,
                                RecKind::Constant(scalar) if *scalar == wanted
                            )
                        {
                            return u32::try_from(at).ok();
                        }
                        continue;
                    }
                    let effects = other.effects();
                    if effects.contains(HelperEffects::WRITES_HEAP)
                        || effects.contains(HelperEffects::PRIVATE_STATE)
                        || effects.contains(HelperEffects::RESUMES)
                    {
                        return None;
                    }
                }
                RecKind::Constant(_)
                | RecKind::FrameInput { .. }
                | RecKind::Move(_)
                | RecKind::Negate(_)
                | RecKind::IntToFloat(_)
                | RecKind::Numeric(..)
                | RecKind::ToWide(..)
                | RecKind::NativeUnary(..)
                | RecKind::NativeBinary(..)
                | RecKind::Arithmetic(..)
                | RecKind::Compare(..)
                | RecKind::Not(_)
                | RecKind::Select(..)
                | RecKind::GuardBranch { .. }
                | RecKind::ArrayLoad { .. }
                | RecKind::LaneArrayLoad { .. }
                | RecKind::NumberLoad { .. }
                | RecKind::NumberTag(_)
                | RecKind::BoolAnd(..)
                | RecKind::NumberModulo(..)
                | RecKind::ArrayNext { .. }
                | RecKind::HashLoad { .. }
                | RecKind::ArrayStore { .. }
                | RecKind::ArrayOverwrite { .. }
                | RecKind::FieldLoad { .. }
                | RecKind::FieldStore { .. }
                | RecKind::TableLength { .. }
                | RecKind::StringLength { .. }
                | RecKind::StringByte { .. }
                | RecKind::TableFieldLoad { .. }
                | RecKind::TableFieldStore { .. }
                | RecKind::LaneArrayStore { .. }
                | RecKind::MetaGuard { .. }
                | RecKind::HelperResult { .. } => {}
                RecKind::PhantomTable { .. }
                | RecKind::HostCall { .. }
                | RecKind::SelfCall { .. } => return None,
            }
        }
        None
    }

    /// `SetUpvalue` in such a frame: the live closure's cell takes the value
    /// at once, through the runtime's own write (so an open cell writes the
    /// enclosing frame's stack slot, and a table or string stored into a cell
    /// reaches the collector's write barrier). The store is an effect: a
    /// guard after it may not start the call over.
    fn record_dynamic_set_upvalue(
        &mut self,
        at: &HookPoint<'_>,
        a: usize,
        index: u16,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let closure = self.dynamic_closure().ok_or(TraceAbortReason::Divergence)?;
        self.check_register(a)?;
        let (source, ty) = self.read(at.registers, a)?;
        if !HelperType::Value.accepts(ty) {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // The cell must exist, as the helper finds it.
        Self::upvalue_cell(at, index)?;
        self.dynamic_cell_writes = true;
        let index = self.scalar(pc, TraceScalar::Integer(i64::from(index)));
        self.emit_helper_call(
            pc,
            HelperId::CELL_SET,
            &[
                (Operand::Op(closure), TraceType::Closure),
                (Operand::Op(index), TraceType::Integer),
                (source, ty),
            ],
        );
        self.stores += 1;
        Ok(pc + 1)
    }

    /// The slot naming upvalue `index` of the running frame's closure.
    fn upvalue_slot(&self, index: u16) -> ExternalSlot {
        match (self.frames.last(), &self.activation) {
            (Some(Inlined { closure, .. }), _) | (None, Some(Activation { closure, .. })) => {
                ExternalSlot::ClosureUpvalue {
                    closure: *closure,
                    index,
                }
            }
            (None, None) => ExternalSlot::Upvalue(index),
        }
    }

    /// The name of the global at `constant` of the running frame's chunk.
    // Kept a method beside the other frame lookups, though the chunk alone
    // decides the answer.
    #[allow(clippy::unused_self)]
    fn global_name(
        &self,
        at: &HookPoint<'_>,
        constant: u32,
    ) -> Result<crate::string::LuaString, TraceAbortReason> {
        match usize::try_from(constant)
            .ok()
            .and_then(|index| at.chunk.constants.get(index))
        {
            Some(Constant::String(name)) => Ok(name.clone()),
            _ => Err(TraceAbortReason::Divergence),
        }
    }

    /// Whether the window holds the global `name` of the root frame's
    /// environment in a slot. A callee's global slot whose name cannot be
    /// resolved counts as one.
    fn holds_global_slot(&self, at: &HookPoint<'_>, name: &crate::string::LuaString) -> bool {
        self.externals.iter().any(|slot| match *slot {
            ExternalSlot::Global(constant) => self
                .chunk
                .constants
                .get(constant as usize)
                .is_none_or(|held| {
                    !matches!(held, Constant::String(held) if held.as_bytes() != name.as_bytes())
                }),
            ExternalSlot::ClosureGlobal { closure, constant } => {
                let Some(surroundings) = at.surroundings else {
                    return true;
                };
                let Some(parts) = self.pinned.get(closure as usize).and_then(|identity| {
                    surroundings
                        .runtime
                        .closure_call_parts(ClosureHandle::from_native_identity(*identity))
                }) else {
                    return true;
                };
                parts.environment == surroundings.environment
                    && parts.chunk.constants.get(constant as usize).is_none_or(|held| {
                        !matches!(held, Constant::String(held) if held.as_bytes() != name.as_bytes())
                    })
            }
            // The caller's environment is the root's in every profile that
            // runs a function in the environment it was made in: a name that
            // matches counts as held.
            ExternalSlot::CallerGlobal(constant) => self.caller_seed().is_none_or(|seed| {
                seed.chunk.constants.get(constant as usize).is_none_or(|held| {
                    !matches!(held, Constant::String(held) if held.as_bytes() != name.as_bytes())
                })
            }),
            _ => false,
        })
    }

    /// Whether the code the loop's trace tree can run in the root chunk
    /// stores the global at `constant`, or the running frame is a callee that
    /// cannot share the root's field read. Such a global is read and written
    /// through the environment table, so that a read after a store sees the
    /// stored value.
    ///
    /// That code is the loop's own span and, when the tree may grow outer
    /// sides (see [`Self::coverable`]), the spans of the loops around it. A
    /// global slot is read once at the tree's entry and every member shares
    /// it, so a store in an outer side has to rule the slot out for the root
    /// as well; every member therefore computes the same names.
    fn stores_global(&self, at: &HookPoint<'_>, constant: u32) -> bool {
        if self.function.is_some() {
            return false;
        }
        let Ok(name) = self.global_name(at, constant) else {
            return false;
        };
        self.stored_globals
            .get_or_init(|| {
                let root = LoopSpan {
                    header: self.header,
                    latch: self.latch,
                };
                let mut spans = vec![root];
                if self.coverable() {
                    spans.extend(nest::enclosing_loops(&self.chunk, root));
                }
                // A side that may return from the root frame runs the code
                // after the loops too.
                if self.caller_seed().is_some() {
                    spans.push(LoopSpan {
                        header: 0,
                        latch: self.chunk.code.len().saturating_sub(1),
                    });
                }
                let mut names: Vec<Box<[u8]>> = Vec::new();
                for span in spans {
                    for instruction in self
                        .chunk
                        .code
                        .get(span.header..=span.latch)
                        .unwrap_or_default()
                    {
                        if instruction.opcode().ok() == Some(OpCode::SetGlobal)
                            && let Some(Constant::String(name)) =
                                self.chunk.constants.get(usize::from(instruction.d()))
                            && !names.iter().any(|known| **known == *name.as_bytes())
                        {
                            names.push(name.as_bytes().into());
                        }
                    }
                }
                names
            })
            .iter()
            .any(|stored| **stored == *name.as_bytes())
    }

    /// The environment table of the root frame, which the entry binds.
    fn environment_receiver<'a>(
        &mut self,
        at: &HookPoint<'a>,
    ) -> Result<(Receiver, TableHandle, &'a Runtime), TraceAbortReason> {
        // A callee has its own environment, and the stores of the root
        // would not reach its reads.
        if !self.in_home() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let external = self.external(at, ExternalSlot::Environment)?;
        let (operand, ty) = self.read_external(at, external)?;
        if ty != TraceType::Table {
            return Err(TraceAbortReason::Divergence);
        }
        let Value::Table(handle) = self.external_value(at, ExternalSlot::Environment)? else {
            return Err(TraceAbortReason::Divergence);
        };
        self.receiver_of(at, operand, handle)
    }

    /// The slot naming the global at `constant` of the running frame's chunk.
    fn global_slot(&self, constant: u32) -> ExternalSlot {
        match (self.frames.last(), &self.activation) {
            // The root frame's caller runs no closure of the trace's: its
            // constants and its environment are named by the caller itself.
            (
                Some(Inlined {
                    closure: linked::NO_CLOSURE,
                    entry: Entry::Caller { .. },
                    ..
                }),
                _,
            ) => ExternalSlot::CallerGlobal(constant),
            (Some(Inlined { closure, .. }), _) | (None, Some(Activation { closure, .. })) => {
                ExternalSlot::ClosureGlobal {
                    closure: *closure,
                    constant,
                }
            }
            (None, None) => ExternalSlot::Global(constant),
        }
    }

    /// The cell an upvalue slot of the running frame names.
    fn upvalue_cell<'a>(
        at: &HookPoint<'a>,
        index: u16,
    ) -> Result<(CellHandle, &'a Runtime), TraceAbortReason> {
        let surroundings = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let cell = *surroundings
            .upvalues
            .get(usize::from(index))
            .ok_or(TraceAbortReason::Divergence)?;
        Ok((cell, surroundings.runtime))
    }

    /// The cells and the chunk of the pinned closure of lane `closure`.
    fn pinned_closure_parts<'a>(
        &self,
        at: &HookPoint<'a>,
        closure: u32,
    ) -> Result<(&'a [CellHandle], &'a Arc<Chunk>), TraceAbortReason> {
        let identity = *self
            .pinned
            .get(closure as usize)
            .ok_or(TraceAbortReason::Divergence)?;
        let surroundings = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        surroundings
            .runtime
            .closure_call_parts(ClosureHandle::from_native_identity(identity))
            .map(|parts| (&**parts.cells, parts.chunk))
            .ok_or(TraceAbortReason::Divergence)
    }

    /// What the interpreter holds right now for an upvalue cell or a global
    /// of the running frame, or of the pinned closure a slot names (a side
    /// recording starts in the root frame).
    fn external_value(
        &self,
        at: &HookPoint<'_>,
        slot: ExternalSlot,
    ) -> Result<Value, TraceAbortReason> {
        let surroundings = at
            .surroundings
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        match slot {
            // Not a Lua value: the recorder names these slots itself (see
            // `tail_count_register` and `root_closure_register`).
            ExternalSlot::TailCalls => Err(TraceAbortReason::Divergence),
            // A register of the root frame's caller: while the recording
            // runs in the caller its frame is the one the hook shows, and
            // before that it is the frame the recording started beside.
            ExternalSlot::CallerRegister(register) => {
                let register = usize::from(register);
                if self.in_caller() {
                    at.registers.get(register).cloned()
                } else {
                    self.caller_seed()
                        .and_then(|seed| seed.registers.get(register).cloned())
                }
                .ok_or(TraceAbortReason::Divergence)
            }
            ExternalSlot::RootClosure => self
                .home_closure
                .map(Value::Closure)
                .ok_or(TraceAbortReason::Divergence),
            ExternalSlot::Environment => Ok(Value::Table(surroundings.environment)),
            // Named by the recorder itself (see `root_varargs`).
            ExternalSlot::VarargCount | ExternalSlot::Vararg(_) => {
                Err(TraceAbortReason::Divergence)
            }
            ExternalSlot::Upvalue(index) => {
                let (cell, runtime) = Self::upvalue_cell(at, index)?;
                runtime
                    .cell_get(cell)
                    .map_err(|_| TraceAbortReason::Divergence)
            }
            // A cell of an inlined closure, which a side recording has not
            // entered: the pinned closure holds it.
            ExternalSlot::ClosureUpvalue { closure, index } => {
                let (cells, _) = self.pinned_closure_parts(at, closure)?;
                let cell = *cells
                    .get(usize::from(index))
                    .ok_or(TraceAbortReason::Divergence)?;
                surroundings
                    .runtime
                    .cell_get(cell)
                    .map_err(|_| TraceAbortReason::Divergence)
            }
            ExternalSlot::UpvalueField {
                upvalue, constant, ..
            }
            | ExternalSlot::ClosureUpvalueField {
                upvalue, constant, ..
            } => {
                let (cell, chunk) = match slot {
                    ExternalSlot::ClosureUpvalueField { closure, .. } => {
                        let (cells, chunk) = self.pinned_closure_parts(at, closure)?;
                        let cell = *cells
                            .get(usize::from(upvalue))
                            .ok_or(TraceAbortReason::Divergence)?;
                        (cell, chunk)
                    }
                    _ => (Self::upvalue_cell(at, upvalue)?.0, at.chunk),
                };
                let runtime = surroundings.runtime;
                let Some(Constant::String(name)) = usize::try_from(constant)
                    .ok()
                    .and_then(|index| chunk.constants.get(index))
                else {
                    return Err(TraceAbortReason::Divergence);
                };
                let Ok(Value::Table(table)) = runtime.cell_get(cell) else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                runtime
                    .table_get_string_ref(table, name)
                    .cloned()
                    .ok_or(TraceAbortReason::UnsupportedOperand)
            }
            ExternalSlot::Global(constant)
            | ExternalSlot::ClosureGlobal { constant, .. }
            | ExternalSlot::CallerGlobal(constant) => {
                // The name is a constant of the chunk of the function that
                // reads the global, which a side recording has not entered.
                let owner = match slot {
                    ExternalSlot::CallerGlobal(_) => self
                        .caller_seed()
                        .map(|seed| (&seed.chunk, seed.environment)),
                    ExternalSlot::ClosureGlobal { closure, .. } => self
                        .pinned
                        .get(closure as usize)
                        .and_then(|identity| {
                            surroundings
                                .runtime
                                .closure_call_parts(ClosureHandle::from_native_identity(*identity))
                        })
                        .map(|parts| (parts.chunk, parts.environment)),
                    _ => None,
                };
                let (chunk, environment) = owner.unwrap_or((at.chunk, surroundings.environment));
                let Some(Constant::String(name)) = usize::try_from(constant)
                    .ok()
                    .and_then(|index| chunk.constants.get(index))
                else {
                    return Err(TraceAbortReason::Divergence);
                };
                // `_ENV` names the environment itself in Lua 5.2 and later,
                // and a missing global runs `__index`. Only a present raw
                // value is a read the trace can repeat.
                if self.profile_has_env_name && name.as_bytes() == b"_ENV" {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                surroundings
                    .runtime
                    .table_get_string_ref(environment, name)
                    .cloned()
                    .ok_or(TraceAbortReason::UnsupportedOperand)
            }
        }
    }

    /// How a cell slot is named in an alias.
    const fn cell_source(slot: ExternalSlot) -> Option<CellSource> {
        match slot {
            ExternalSlot::Upvalue(index) => Some(CellSource::Upvalue(index)),
            ExternalSlot::ClosureUpvalue { closure, index } => {
                Some(CellSource::ClosureUpvalue { closure, index })
            }
            ExternalSlot::Global(_)
            | ExternalSlot::ClosureGlobal { .. }
            | ExternalSlot::CallerGlobal(_)
            | ExternalSlot::UpvalueField { .. }
            | ExternalSlot::ClosureUpvalueField { .. }
            | ExternalSlot::CallerRegister(_)
            | ExternalSlot::TailCalls
            | ExternalSlot::RootClosure
            | ExternalSlot::VarargCount
            | ExternalSlot::Vararg(_)
            | ExternalSlot::Environment => None,
        }
    }

    /// The window register standing for `slot`, naming it on first use. Its
    /// entry type is what the cell or global holds now: nothing in the
    /// iteration has written it before its first use.
    ///
    /// A cell the trace already holds under another name gets that name's
    /// register, so the variable has one value. The sharing is a promise the
    /// VM checks at entry: a callee often shares a variable with the root
    /// frame (a captured local of it) or with another callee.
    fn external(
        &mut self,
        at: &HookPoint<'_>,
        slot: ExternalSlot,
    ) -> Result<ExternalRef, TraceAbortReason> {
        // A frame that runs whichever closure of a prototype the call had has
        // no cell, global or environment of its own to name: another closure
        // of the prototype may have another one. And a recording that reads
        // and writes callee cells through the closure holds no cell as a
        // window slot, which the write would leave stale.
        let field_slot = matches!(
            slot,
            ExternalSlot::UpvalueField { .. } | ExternalSlot::ClosureUpvalueField { .. }
        );
        // A global of such a frame is the pinned closure's once a guard
        // checked that the live closure shares its environment.
        let env_guarded = self
            .frames
            .last()
            .is_some_and(|frame| self.env_guarded.contains(&frame.frame));
        if (self.dynamic_cells && field_slot)
            || (self.dynamic_closure().is_some()
                && match slot {
                    ExternalSlot::ClosureUpvalue { .. }
                    | ExternalSlot::ClosureUpvalueField { .. } => true,
                    ExternalSlot::ClosureGlobal { .. } => !env_guarded,
                    _ => false,
                })
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let name = Self::cell_source(slot);
        if let Some(name) = name
            && let Some((_, register)) = self.aliases.iter().find(|(alias, _)| alias.cell == name)
        {
            return Ok(ExternalRef {
                register: *register,
                slot,
            });
        }
        if let Some(position) = self.externals.iter().position(|known| *known == slot) {
            return Ok(ExternalRef {
                register: self.register_count + position,
                slot,
            });
        }
        let live = self.external_value(at, slot)?;
        let kind = Kind::of(&live, self.integers);
        // A nil global is a missing one, which the lookup above refused.
        if kind == Kind::Other {
            self.note_callable(&live);
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let cell = match slot {
            ExternalSlot::Upvalue(index) | ExternalSlot::ClosureUpvalue { index, .. } => {
                Some(Self::upvalue_cell(at, index)?.0)
            }
            ExternalSlot::Global(_)
            | ExternalSlot::ClosureGlobal { .. }
            | ExternalSlot::CallerGlobal(_)
            | ExternalSlot::UpvalueField { .. }
            | ExternalSlot::ClosureUpvalueField { .. }
            | ExternalSlot::CallerRegister(_)
            | ExternalSlot::TailCalls
            | ExternalSlot::RootClosure
            | ExternalSlot::VarargCount
            | ExternalSlot::Vararg(_)
            | ExternalSlot::Environment => None,
        };
        if let (Some(cell), Some(name)) = (cell, name) {
            let held = self
                .root_captured
                .iter()
                .find(|(_, captured)| *captured == cell)
                .map(|(register, _)| {
                    (
                        usize::from(*register),
                        CellSource::Captured(u16::from(*register)),
                    )
                })
                .or_else(|| {
                    let position = self.external_cells.iter().position(|c| *c == Some(cell))?;
                    Some((
                        self.register_count + position,
                        Self::cell_source(self.externals[position])?,
                    ))
                });
            if let Some((register, is)) = held {
                self.aliases.push((CellAlias { cell: name, is }, register));
                return Ok(ExternalRef { register, slot });
            }
        }
        if self.externals.len() >= MAX_TRACE_EXTERNALS {
            return Err(limit("externals"));
        }
        self.externals.push(slot);
        self.external_cells.push(cell);
        self.entry.push(kind);
        self.regs.push(Sym::Carried);
        self.written.push(false);
        Ok(ExternalRef {
            register: self.regs.len() - 1,
            slot,
        })
    }

    /// Like [`Self::read`] for an external window register: the symbolic
    /// value, and a check that the cell or global still holds what the
    /// recorder believes.
    fn read_external(
        &self,
        at: &HookPoint<'_>,
        external: ExternalRef,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        let register = external.register;
        let (operand, kind) = match self.regs[register] {
            Sym::Carried => (Operand::Carried(register), self.entry[register]),
            Sym::Op(index) => {
                let ty = self.ops[index as usize]
                    .ty
                    .expect("an operand operation defines a value");
                (Operand::Op(index), Kind::from_type(ty))
            }
            Sym::Root(_) | Sym::Virtual(_) => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let ty = kind
            .trace_type()
            .ok_or(TraceAbortReason::UnsupportedOperand)?;
        let live = self.external_value(at, external.slot)?;
        if Kind::of(&live, self.integers) != kind {
            return Err(TraceAbortReason::Divergence);
        }
        Ok((operand, ty))
    }

    fn emit(&mut self, pc: usize, kind: RecKind, ty: Option<TraceType>) -> u32 {
        // The lifetime contract of borrowed strings: an operation that ends
        // borrows may not be recorded while a register still holds one. The
        // recorder has no operation that owns a borrow first, so the
        // recording is rejected (see `record`).
        if kind.ends_borrows() && self.holds_borrow() {
            self.borrow_ended = true;
        }
        self.slot_facts_before(&kind);
        if let Some(snapshot) = kind.guard_snapshot() {
            let snapshot = snapshot as usize;
            if self.guarded.len() <= snapshot {
                self.guarded.resize(snapshot + 1, false);
            }
            if !self.guarded[snapshot] {
                self.guarded[snapshot] = true;
                self.guarded_snapshots += 1;
            }
        }
        let frame = self.frame_id();
        self.ops.push(RecOp {
            frame,
            pc: self.site_override.unwrap_or(pc),
            kind,
            ty,
        });
        u32::try_from(self.ops.len() - 1).expect("operation count fits u32")
    }

    /// Whether a register of any frame, or an open result, holds a borrowed
    /// string now. Snapshots taken after an operation that ends borrows would
    /// restore it from a slot the operation may have replaced.
    fn holds_borrow(&self) -> bool {
        let borrowed = |sym: &Sym| match sym {
            Sym::Op(index) => self.ops[*index as usize]
                .ty
                .is_some_and(TraceType::is_borrowed),
            Sym::Carried | Sym::Root(_) | Sym::Virtual(_) => false,
        };
        self.regs.iter().any(borrowed)
            || self
                .frames
                .iter()
                .any(|frame| frame.regs.iter().chain(&frame.varargs).any(borrowed))
            || self.open.iter().flatten().any(|open| match open {
                Open::Op(index) | Open::Char(index) => borrowed(&Sym::Op(*index)),
                Open::Root(_) => false,
            })
    }

    /// Writes a register of the running frame.
    fn write(&mut self, register: usize, value: impl Into<Sym>) {
        match self.frames.last_mut() {
            Some(frame) => frame.regs[register] = value.into(),
            None if self.boxed == Some(register) => self.store_control(register, value.into()),
            None => {
                self.regs[register] = value.into();
                self.written[register] = true;
            }
        }
    }

    /// A write to the control register of the boxed `next` loop (`pairs (t)`
    /// gives nil, say): the value goes into the loop's lane, and the register
    /// stays the entry input that stands for the lane.
    fn store_control(&mut self, register: usize, value: Sym) {
        let ty = match value {
            Sym::Op(index) => self.ops[index as usize].ty,
            _ => None,
        };
        let (Sym::Op(index), Some(ty)) = (value, ty) else {
            self.deferred = Some(TraceAbortReason::UnsupportedOperand);
            return;
        };
        // A borrowed string has no lane to store.
        if !HelperType::Value.accepts(ty) || ty.is_borrowed() {
            self.deferred = Some(TraceAbortReason::UnsupportedOperand);
            return;
        }
        self.emit_helper_call(
            self.current_pc,
            HelperId::BOX_SET,
            &[
                (Operand::Carried(register), TraceType::String),
                (Operand::Op(index), ty),
            ],
        );
    }

    /// Writes a window register beyond the guest registers: an upvalue cell.
    /// They belong to the root window whichever frame runs.
    fn write_external(&mut self, register: usize, value: impl Into<Sym>) {
        self.regs[register] = value.into();
        self.written[register] = true;
        self.cell_writes += 1;
    }

    /// Takes the state the recording starts in for the type guards of the
    /// registers of either type (see [`Self::with_lazy`]), if any of them
    /// holds nil or a number now. A side that starts inside inlined calls, a
    /// function trace and a counted one take none.
    fn start_lazy(&mut self, pc: usize) {
        let lazy = match &self.side {
            Some(seed) if !seed.frames.is_empty() => return,
            Some(seed) => seed.lazy.clone(),
            None => self.lazy.clone(),
        };
        if lazy.is_empty() || self.function.is_some() || self.counting {
            return;
        }
        let guest = usize::from(self.chunk.register_count).min(self.register_count);
        let mut registers: Vec<usize> = lazy
            .iter()
            .map(|register| usize::from(*register))
            .filter(|register| {
                *register < guest
                    && matches!(
                        self.entry.get(*register),
                        Some(Kind::Nil | Kind::Float | Kind::Integer)
                    )
            })
            .collect();
        registers.sort_unstable();
        registers.dedup();
        if registers.is_empty() {
            return;
        }
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.lazy_start = Some((snapshot, registers));
    }

    /// Takes a snapshot of the current state with `overrides` applied to the
    /// running frame's registers.
    fn snapshot(&mut self, phase: SnapshotPhase, overrides: &[(usize, Sym)]) -> u32 {
        if let Some(before_call) = self.reexec {
            return before_call;
        }
        let mut regs = self.regs.clone();
        let mut inner: Vec<RecSnapFrame> = self
            .frames
            .iter()
            .map(|frame| {
                let mut regs = frame.regs.clone();
                // The closure of a frame that runs whichever closure of a
                // prototype the call had is a register after the chunk's.
                if let Some(closure) = frame.dynamic {
                    regs.push(Sym::Op(closure));
                }
                RecSnapFrame {
                    frame: frame.frame,
                    prev: frame.prev,
                    regs,
                }
            })
            .collect();
        let target = inner.last_mut().map_or(&mut regs, |frame| &mut frame.regs);
        for &(register, value) in overrides {
            target[register] = value;
        }
        let root_resume = self.frames.first().map_or(
            match phase {
                SnapshotPhase::Before { pc } => pc,
                SnapshotPhase::After { successor, .. } => successor,
            },
            // The root frame has returned when the caller runs: only what
            // its `Return` reads is live.
            |outermost| {
                outermost.call_pc + usize::from(!matches!(outermost.entry, Entry::Caller { .. }))
            },
        );
        let mut virtuals: Vec<(u32, VirtualState)> = Vec::new();
        let inner_regs = inner.iter().flat_map(|frame| frame.regs.iter());
        for sym in regs.iter().chain(inner_regs) {
            if let Sym::Virtual(id) = sym
                && !virtuals.iter().any(|(known, _)| known == id)
            {
                virtuals.push((*id, self.virtuals[*id as usize].clone()));
            }
        }
        // The tables those hold are listed too, as they stand.
        let mut position = 0;
        while position < virtuals.len() {
            let inner_ids: Vec<u32> = virtuals[position]
                .1
                .items()
                .filter_map(|item| match item {
                    VirtualItem::Virtual(inner) => Some(*inner),
                    _ => None,
                })
                .collect();
            for id in inner_ids {
                if !virtuals.iter().any(|(known, _)| *known == id) {
                    virtuals.push((id, self.virtuals[id as usize].clone()));
                }
            }
            position += 1;
        }
        let root_frame = self.root_frame_id();
        // An `After` snapshot has run the instruction being recorded.
        let counted = self.counted
            + match phase {
                SnapshotPhase::Before { .. } => 0,
                SnapshotPhase::After { .. } => self.counted_now,
            };
        let pending = self
            .frames
            .first()
            .filter(|outermost| outermost.results > 0)
            .map(|outermost| (outermost.call_pc, outermost.result_base, outermost.results));
        let held = self
            .frames
            .first()
            .and_then(|outermost| match outermost.entry {
                Entry::Concat { first, pending, .. } => Some((
                    usize::from(first),
                    usize::from(first) + usize::from(pending),
                )),
                _ => None,
            });
        let fresh = RecSnapshot {
            counted,
            phase,
            prev: self.prev,
            root_resume,
            root_frame,
            regs,
            pending,
            held,
            inner,
            virtuals,
        };
        // Guards with nothing between them share one snapshot.
        if let Some(last) = self.snapshots.last()
            && last.same_as(&fresh)
        {
            return u32::try_from(self.snapshots.len() - 1).expect("snapshot count fits u32");
        }
        self.snapshots.push(fresh);
        u32::try_from(self.snapshots.len() - 1).expect("snapshot count fits u32")
    }

    /// Builds and verifies the closed trace.
    // Each `Sym` case is its own documented shape; the not-live `Virtual`
    // and `Op` arms only happen to share the nil entry body.
    #[allow(clippy::match_same_arms)]
    // A trace site and a side trace are the domain's paired terms.
    #[allow(clippy::similar_names)]
    #[allow(
        clippy::too_many_lines,
        clippy::needless_range_loop,
        reason = "one linear pass over parallel per-register tables"
    )]
    fn finalize(&self, id: TraceId) -> Result<TraceIr, TraceAbortReason> {
        let _icount = crate::jit::icount::scope("finalize");
        // The window: guest registers, then the externals the iteration named.
        let count = self.regs.len();
        // A snapshot taken before an external was first used does not list
        // it. Nothing had touched it then, so it was still carried.
        let sym_of =
            |regs: &[Sym], register: usize| regs.get(register).copied().unwrap_or(Sym::Carried);
        let mut used_by_ops = vec![false; count];
        for op in &self.ops {
            op.kind.operands(|operand| {
                if let Operand::Carried(register) = operand {
                    used_by_ops[register] = true;
                }
            });
        }
        // An unguarded branch condition is a read: its type is fixed by the
        // entry contract, so the register needs an input and a type guard.
        for &register in &self.condition_reads {
            used_by_ops[register] = true;
        }
        // A cell a callee writes through its closure may be the variable of
        // a captured local of the root frame, which the trace would hold in
        // a register and write back at its exits. A cell a callee only reads
        // through its closure is that variable as the interpreter's register
        // holds it, which is the trace's while no member writes the
        // register: Mineclonia's ULL helpers are locals of random.lua's main
        // chunk, which its xoroshiro self-test (random.lua:1159) runs in, and
        // `getlong` reads them through each fork's closure.
        let holds_captured = self.root_captured.iter().any(|(register, _)| {
            let register = usize::from(*register);
            register < count && (self.written[register] || used_by_ops[register])
        });
        let writes_captured = self.root_captured.iter().any(|(register, _)| {
            let register = usize::from(*register);
            register < count && self.written[register]
        });
        let (tree_dynamic, tree_writes, tree_captured, tree_written) =
            self.side
                .as_ref()
                .map_or((false, false, false, false), |seed| {
                    (
                        seed.dynamic_cells,
                        seed.dynamic_cell_writes,
                        seed.captured_busy,
                        seed.captured_written,
                    )
                });
        if ((self.dynamic_cells || tree_dynamic) && (writes_captured || tree_written))
            || ((self.dynamic_cell_writes || tree_writes) && (holds_captured || tree_captured))
        {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A register that only snapshots read, and that nothing reads after
        // the interpreter resumes, carries no value across the exit.
        // Restoring it would force a typed entry input, and one more trace
        // version for every type a dead temporary takes at the header.
        let live_in = self
            .liveness
            .get_or_init(|| observable_registers(&self.chunk));
        // The liveness of each later activation's chunk, by IR frame. A
        // register past the activation's own is not the frame's: dead.
        let activation_liveness: Vec<(u32, Vec<RegisterSet>)> = self
            .activation_chunks
            .iter()
            .map(|(frame, chunk)| (*frame, observable_registers(chunk)))
            .collect();
        let guest_registers = self.register_count;
        let observed_in_frame = |frame: u32, pc: usize, register: usize| {
            // A cell or global is visible to the rest of the program.
            if register >= guest_registers {
                return true;
            }
            let sets = if frame == 0 {
                live_in.as_slice()
            } else {
                match activation_liveness
                    .iter()
                    .find(|(known, _)| *known == frame)
                {
                    Some((_, sets)) => sets.as_slice(),
                    None => return true,
                }
            };
            let (Some(set), Ok(register)) = (sets.get(pc), u8::try_from(register)) else {
                return true;
            };
            register_set_contains(set, register)
        };
        // The loop variables a boxed loop does not produce need no value of
        // their own at an exit: the lanes restore them (see
        // `Self::box_loop`).
        let live_in_frame = |frame: u32, pc: usize, register: usize| {
            !(frame == 0 && self.unread.contains(&register))
                && observed_in_frame(frame, pc, register)
        };
        let live_at = |pc: usize, register: usize| live_in_frame(0, pc, register);
        let mut used_by_snapshots = vec![false; count];
        // A register the outermost inlined call's return writes, which was
        // dead at the call, has no value to carry into a snapshot inside it.
        let pending_dead = |pending: Option<(usize, usize, usize)>, frame: u32, register: usize| {
            pending.is_some_and(|(call_pc, base, results)| {
                (base..base + results).contains(&register)
                    && !live_in_frame(frame, call_pc, register)
            })
        };
        for snapshot in &self.snapshots {
            let pc = snapshot.root_resume;
            for register in 0..count {
                if sym_of(&snapshot.regs, register) == Sym::Carried
                    && (used_by_ops[register]
                        || snapshot
                            .held
                            .is_some_and(|(from, to)| (from..to).contains(&register))
                        || (live_in_frame(snapshot.root_frame, pc, register)
                            && !pending_dead(snapshot.pending, snapshot.root_frame, register)))
                {
                    used_by_snapshots[register] = true;
                }
            }
        }
        // A virtual table a snapshot builds stores the entry values of some
        // registers: those arrive as inputs too.
        for snapshot in &self.snapshots {
            for register in 0..count {
                if let Sym::Virtual(id) = sym_of(&snapshot.regs, register)
                    && live_in_frame(snapshot.root_frame, snapshot.root_resume, register)
                {
                    // The table, and the ones it holds that the exit builds.
                    let mut pending = vec![id];
                    let mut seen: Vec<u32> = Vec::new();
                    while let Some(id) = pending.pop() {
                        if seen.contains(&id) {
                            continue;
                        }
                        seen.push(id);
                        let Some((_, state)) =
                            snapshot.virtuals.iter().find(|(known, _)| *known == id)
                        else {
                            continue;
                        };
                        // A table that is real is restored as its lane.
                        if state.real.is_some() {
                            continue;
                        }
                        for item in state.items() {
                            match item {
                                VirtualItem::Value(Operand::Carried(carried), _) => {
                                    used_by_snapshots[*carried] = true;
                                }
                                VirtualItem::Virtual(inner) => pending.push(*inner),
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        // Every exit restores the control register of a boxed loop from its
        // lane.
        if let Some(register) = self.boxed.filter(|register| *register < count) {
            used_by_snapshots[register] = true;
            if self.boxed_value {
                // The step stores into the lane, which makes it an input
                // that every iteration shares.
                used_by_snapshots[register + 2] = true;
                used_by_ops[register + 2] = true;
            }
        }
        let side = self.side.as_ref();
        // A side that rejoins an ancestor loop side hands the loop a value for
        // each of its parameters, so the ones it left alone arrive as values.
        if let Some(position) = self.link_loop {
            for &(register, _) in &self.loop_headers[position].params {
                let register = usize::from(register);
                if register < count && sym_of(&self.regs, register) == Sym::Carried {
                    used_by_snapshots[register] = true;
                }
            }
        }
        // A side that ends with an exit (in front of a loop, or at an
        // ancestor loop side's header) and a loop side, whose poll exit
        // resumes at its own header, restore their final state there. A
        // register they left alone that is live where they resume must
        // arrive as a value too: it holds the parent's value, which a native
        // link or a loop side before it may have changed while the
        // interpreter register kept the run's entry state. The loop side
        // that continues such an exit reads it from that state as well, and
        // a side compiled apart would otherwise take the run's entry input.
        if side.is_some() && (self.exit_at.is_some() || self.looping) {
            let resume = self.root_resume();
            for register in 0..count {
                if sym_of(&self.regs, register) == Sym::Carried && live_at(resume, register) {
                    used_by_snapshots[register] = true;
                }
            }
        }
        // Only a side that links back to the root's loop publishes a state
        // there. A register the root takes untyped (see `Self::with_lazy`)
        // that the side writes goes to the root's input slot at the link
        // (see `oximond::jit_ir::trace::plan_tree`), so the side may write
        // it like any other.
        if side.is_some() && self.exit_at.is_none() && !self.looping {
            // The link publishes the final state at the header. A register
            // the side left alone and the loop still needs must arrive as a
            // value: the interpreter register of one the parent changed is
            // stale. A loop parameter the header does not read needs none:
            // the link gives it a value of its type, and an input would pin
            // the stale type the register had at this exit.
            for register in 0..count {
                if sym_of(&self.regs, register) == Sym::Carried
                    && (used_by_ops[register] || live_at(self.header, register))
                {
                    used_by_snapshots[register] = true;
                }
            }
        }
        // Loop formation decides how each register crosses the loop-back. A
        // side trace has no loop of its own: every register it reads, or that
        // a snapshot must restore, is an entry input bound to the parent's
        // state, and none is a parameter.
        let usage = |register: usize| RegisterUse {
            written: self.written[register],
            read_by_operations: used_by_ops[register],
            read_by_snapshots: used_by_snapshots[register],
        };
        // A function trace has no loop. Its parameters are entry inputs: an
        // exit in a nested activation restores them into a frame it pushes.
        let function = self.function.as_ref();
        let param_count = function.map_or(0, |function| function.parameters.len());
        // A function trace's tail call count changes in every activation
        // that tail calls: every snapshot names it, like a parameter.
        let count_register = function.and_then(|_| {
            self.externals
                .iter()
                .position(|slot| *slot == ExternalSlot::TailCalls)
                .map(|position| self.register_count + position)
        });
        let counted = |register: usize| register < param_count || Some(register) == count_register;
        let is_param = |register: usize| {
            (side.is_none() || self.looping)
                && function.is_none()
                && usage(register).role() == LoopRole::Carried
        };
        // The tree binds the tables of these registers at entry, so every
        // member has an entry input for each, used or not.
        let is_bound = |register: usize| {
            self.bindings
                .iter()
                .any(|binding| usize::from(binding.register) == register)
        };
        let is_input = |register: usize| {
            if counted(register) {
                return self.entry[register].trace_type().is_some();
            }
            if side.is_some() {
                // A string or other untyped value has no input: the register
                // stays as the interpreter holds it.
                (used_by_ops[register] || used_by_snapshots[register] || is_bound(register))
                    && self.entry[register].trace_type().is_some()
            } else {
                usage(register).needs_entry_input()
            }
        };

        // Only a recording that ends in front of a loop in a call, or a loop
        // side of one, ends inside inlined frames.
        let in_frames = (self.enter_depth > 0 && self.terminal.is_some())
            || (self.loop_depth > 0 && self.frames.len() == self.loop_depth);
        if (!self.frames.is_empty() && !in_frames) || self.open.is_some() {
            return Err(TraceAbortReason::Divergence);
        }
        // The closing instruction: a backward branch, or (for a loop through
        // tail calls) the tail call back into the root closure, after which
        // the interpreter keeps no previous pc.
        let last_pc = match (self.prev, self.loop_tail) {
            // A function trace closes at its return: no loop header follows.
            (PrevPc::At(pc), None) if function.is_some() => Some(pc),
            (_, None) if function.is_some() => None,
            (PrevPc::Unset, Some(_)) => None,
            (PrevPc::At(pc), None) => Some(pc),
            _ => return Err(TraceAbortReason::Divergence),
        };
        // The `last_interpreted_pc` a frame has after a restore, given the pc
        // it executed last.
        let last_of = |prev: PrevPc| match prev {
            PrevPc::Latch => last_pc,
            PrevPc::Unset => None,
            PrevPc::At(pc) => Some(pc),
        };
        let window = u16::try_from(count).expect("register count fits u16");
        // Only a loop trace counts: a function trace and a loop through
        // tail calls end in a return or a tail call that nothing pays for.
        if self.counting
            && (self.function.is_some() || self.loop_tail.is_some() || self.activation.is_some())
        {
            return Err(TraceAbortReason::RootTailCall);
        }
        let mut builder =
            TraceBuilder::new(id, TraceFrameInfo::root(window, self.chunk.code.len()));
        builder.set_counted(self.counting);
        builder.set_seed_frames(
            side.map_or(0, |seed| u8::try_from(seed.frames.len()).unwrap_or(u8::MAX)),
        );
        builder.set_boxed_control(self.boxed.and_then(|register| u16::try_from(register).ok()));
        builder.set_boxed_variables(self.boxed_key, self.boxed_value);
        builder.set_boxed_steps(self.boxed_steps);
        builder.set_array_steps_plain(self.array_steps.0 == 0 && self.array_steps.1 > 0);
        if self.keep_guards {
            builder.set_keep_branch_guards(true);
        }
        for info in &self.inlined {
            let mut info = *info;
            // A later activation of the root frame has the root's window.
            if info.activation.is_some() {
                info.register_count = window;
            }
            builder.add_frame(info);
        }
        if let Some(site) = self.loop_tail {
            builder.set_loop_tail(site);
        }
        match side {
            // A side adds no closure of its own. It lists the root's aliases,
            // which every member of a tree shares, and the ones it found
            // between its callees' cells and the root's captured locals.
            Some(seed) => builder.set_cell_aliases(
                seed.aliases
                    .iter()
                    .copied()
                    .chain(
                        self.aliases[self.seeded_aliases..]
                            .iter()
                            .map(|(alias, _)| *alias),
                    )
                    .collect(),
            ),
            None => {
                builder.set_cell_aliases(self.aliases.iter().map(|(alias, _)| *alias).collect());
            }
        }
        builder.set_pinned_closures(self.pinned.clone());
        builder.set_external(self.externals.clone());
        builder.set_root_constants(self.root_constants.clone());
        builder.set_bindings(self.bindings.clone());
        builder.set_fields(self.fields.clone());
        for register in 0..count {
            if self.written[register] {
                builder.mark_written(FrameId(0), register_u16(register));
            }
        }
        // The registers of either type (see `Self::with_lazy`) that become
        // inputs of type `Any`: read, and no parameter, bound table or loop
        // control. A root's are never written. A side without a loop of its
        // own may write one too (the outer side of Mineclonia's `y` loop
        // computes `ly_dom` again): until it does, it reads the register as
        // the run's entry left it, and a link hands the value it wrote to the
        // root's slot (see `oximond::jit_ir::trace::untyped_link_writes`).
        let plain_side = side.is_some() && !self.looping;
        let (lazy_snapshot, lazy): (Option<u32>, Vec<usize>) = match &self.lazy_start {
            Some((snapshot, registers)) => (
                Some(*snapshot),
                registers
                    .iter()
                    .copied()
                    .filter(|&register| {
                        register < count
                            && is_input(register)
                            && !is_param(register)
                            && (!self.written[register] || plain_side)
                            && !counted(register)
                            && !is_bound(register)
                            && self.boxed != Some(register)
                    })
                    .collect(),
            ),
            None => (None, Vec::new()),
        };
        let is_lazy = |register: usize| lazy.contains(&register);
        // The ones a type guard narrows. A side whose operations never read
        // one only restores it, `Unchanged`, at its exits: a guard would
        // only leave for the other type, which the side does not care about.
        let guarded: Vec<usize> = lazy
            .iter()
            .copied()
            .filter(|&register| side.is_none() || used_by_ops[register])
            .collect();
        let mut inputs: Vec<Option<(ValueId, TraceType)>> = vec![None; count];
        for register in 0..count {
            if is_input(register) {
                let ty = self.entry[register]
                    .trace_type()
                    .ok_or(TraceAbortReason::UnsupportedOperand)?;
                let input_ty = if is_lazy(register) {
                    TraceType::Any
                } else {
                    ty
                };
                inputs[register] = Some((builder.input(register_u16(register), input_ty), ty));
            }
        }
        let mut carried: Vec<Option<ValueId>> = vec![None; count];
        let mut parameters = Vec::new();
        for register in 0..count {
            let Some((input, ty)) = inputs[register] else {
                continue;
            };
            if is_param(register) {
                carried[register] = Some(builder.loop_param(ty, LoopParamInit::Input(input)));
                parameters.push(register);
            } else {
                carried[register] = Some(input);
            }
        }
        // Every operation reads a register of either type as the value its
        // type guard narrows it to. The guards come first in the body, in
        // the order of `guarded`, so their values are the next ones.
        let first_narrowed = builder.next_value_id().index();
        for (position, &register) in guarded.iter().enumerate() {
            carried[register] = Some(ValueId(
                u32::try_from(first_narrowed + position).expect("value count fits u32"),
            ));
        }

        let mut values: Vec<Option<ValueId>> = Vec::with_capacity(self.ops.len());
        // A snapshot's register map: recorded values, the previous
        // iteration's value for a register written later, `Unchanged` for one
        // never written.
        // A global is read once at entry and never written, so the environment
        // holds it whatever the trace has done: no snapshot needs a value.
        let is_global = |register: usize| {
            register >= guest_registers
                && matches!(
                    self.externals.get(register - guest_registers),
                    Some(
                        ExternalSlot::Global(_)
                            | ExternalSlot::ClosureGlobal { .. }
                            | ExternalSlot::CallerGlobal(_)
                            | ExternalSlot::UpvalueField { .. }
                            | ExternalSlot::ClosureUpvalueField { .. }
                            | ExternalSlot::RootClosure
                            | ExternalSlot::VarargCount
                            | ExternalSlot::Vararg(_)
                            | ExternalSlot::Environment
                    )
                )
        };
        // How a snapshot names virtual table `id`: the lane of its real table
        // when it has one by then, otherwise its recipe, by its number among
        // the recipes the snapshot lists (`named`, in order of first use).
        let virtual_entry = |id: u32,
                             states: &[(u32, VirtualState)],
                             values: &[Option<ValueId>],
                             named: &mut Vec<u32>|
         -> Result<RegisterEntry, TraceAbortReason> {
            if let Some((_, state)) = states.iter().find(|(known, _)| *known == id)
                && let Some(real) = state.real
            {
                return values[real.table as usize]
                    .map(RegisterEntry::Value)
                    .ok_or(TraceAbortReason::Divergence);
            }
            let local = named
                .iter()
                .position(|known| *known == id)
                .unwrap_or_else(|| {
                    named.push(id);
                    named.len() - 1
                });
            Ok(RegisterEntry::Virtual(VirtualId(
                u32::try_from(local).expect("virtual count fits u32"),
            )))
        };
        // The recipe of a virtual table in a snapshot: its contents as the
        // snapshot's entries. The phantom charge covers the table's peak,
        // unless the table escapes: then nothing was charged for it.
        let recipe_of = |id: u32,
                         state: &VirtualState,
                         values: &[Option<ValueId>],
                         states: &[(u32, VirtualState)],
                         named: &mut Vec<u32>|
         -> Result<VirtualTable, TraceAbortReason> {
            let entry = |item: &VirtualItem, named: &mut Vec<u32>| match item {
                VirtualItem::Root(slot) => Ok(RegisterEntry::Root(*slot)),
                VirtualItem::Value(operand, _) => match operand {
                    Operand::Carried(register) => carried[*register],
                    Operand::Op(index) => values[*index as usize],
                }
                .map(RegisterEntry::Value)
                .ok_or(TraceAbortReason::Divergence),
                VirtualItem::Virtual(inner) => virtual_entry(*inner, states, values, named),
            };
            let mut array = Vec::with_capacity(state.array.len());
            for item in &state.array {
                array.push(entry(item, named)?);
            }
            let mut fields = Vec::with_capacity(state.fields.len());
            for (key, item) in &state.fields {
                fields.push((key.clone(), entry(item, named)?));
            }
            let metatable = match state.metatable {
                Some(meta) => Some(entry(
                    &VirtualItem::Value(meta.table, TraceType::Table),
                    named,
                )?),
                None => None,
            };
            Ok(VirtualTable {
                alloc: state.alloc.clone(),
                array,
                fields,
                phantom_entries: self.virtuals[id as usize]
                    .real
                    .is_none()
                    .then_some(self.virtuals[id as usize].peak),
                metatable,
            })
        };
        // The entry of one register of a snapshot's root frame. Besides the
        // recording's fixed facts it reads the register's `sym`, whether the
        // register is live at `resume` (in `frame`, given `pending` and
        // `held`), the value of the operation an `Op` names, and for a
        // virtual table the snapshot's states: `snapshot_of` relies on
        // that to take the entries of a run of registers from the snapshot
        // before.
        let map_register = |register: usize,
                            sym: Sym,
                            (frame, resume): (u32, usize),
                            pending: Option<(usize, usize, usize)>,
                            held: Option<(usize, usize)>,
                            values: &[Option<ValueId>],
                            (named, states): (&mut Vec<u32>, &[(u32, VirtualState)])|
         -> Result<RegisterEntry, TraceAbortReason> {
            let live_at = |pc: usize, register: usize| {
                held.is_some_and(|(from, to)| (from..to).contains(&register))
                    || live_in_frame(frame, pc, register) && !pending_dead(pending, frame, register)
            };
            // `named` holds the virtual tables the live registers name,
            // in order of first use: the snapshot's own ids.
            {
                match sym {
                    // A register of either type: no member writes it, so
                    // the interpreter holds what the run found at entry.
                    Sym::Carried if frame == self.root_frame_id() && is_lazy(register) => {
                        Ok(RegisterEntry::Unchanged)
                    }
                    // A parameter of a function trace: a nested
                    // activation's frame is a fresh one the exit pushes.
                    Sym::Carried if counted(register) => carried[register]
                        .map(RegisterEntry::Value)
                        .ok_or(TraceAbortReason::Divergence),
                    // A table nothing reads after the exit is not built.
                    Sym::Virtual(_) if !live_at(resume, register) => {
                        Ok(RegisterEntry::Const(TraceScalar::Nil))
                    }
                    Sym::Virtual(id) => virtual_entry(id, states, values, named),
                    Sym::Carried if is_global(register) => Ok(RegisterEntry::Unchanged),
                    // The control value of a `string.gmatch` loop is the step
                    // result of the last step. The trace does not carry it
                    // (nothing reads it but the interpreter's hidden slot),
                    // so an exit restores the result from the loop's own
                    // values: after the loop-back it is the step's first
                    // result. Before it, it is what the first variable held
                    // when the iteration began, whatever the body made of it
                    // since, or the slot as it is when the variable is no
                    // value the trace has. The first iteration of an entry
                    // can differ from the interpreter's slot when the body
                    // of the interpreted iteration before it rewrote the
                    // variable. A step the exit replays ignores the slot, and
                    // the loop-back writes it again.
                    Sym::Carried
                        if frame == self.root_frame_id()
                            && !self.written[register]
                            && self.gmatch_controls.iter().any(|(own, _)| *own == register) =>
                    {
                        let stored = self
                            .gmatch_controls
                            .iter()
                            .find(|(own, _)| *own == register)
                            .and_then(|(_, index)| *index)
                            .and_then(|index| values.get(index as usize).copied().flatten());
                        let first = carried.get(register + 1).copied().flatten();
                        Ok(stored
                            .or(first)
                            .map_or(RegisterEntry::Unchanged, RegisterEntry::Value))
                    }
                    // The first variable of a boxed loop that the body never
                    // reads holds the key, which the control lane holds; the
                    // second has a lane of its own. The step produces
                    // neither.
                    Sym::Carried
                        if frame == self.root_frame_id()
                            && observed_in_frame(frame, resume, register)
                            && self.boxed.is_some_and(|control| {
                                (self.boxed_key && register == control + 1)
                                    || (self.boxed_value && register == control + 2)
                            }) =>
                    {
                        let control = self.boxed.unwrap_or_default();
                        let lane = if register == control + 1 {
                            control
                        } else {
                            register
                        };
                        carried[lane]
                            .map(RegisterEntry::Value)
                            .ok_or(TraceAbortReason::Divergence)
                    }
                    // A temporary the interpreter writes before it reads
                    // it again. The compiler leaves no clear behind dead
                    // temporaries, so without this every temporary of the
                    // iteration would stay a live value up to the exit.
                    Sym::Op(_) if !live_at(resume, register) => {
                        Ok(RegisterEntry::Const(TraceScalar::Nil))
                    }
                    Sym::Op(index) => values[index as usize]
                        .map(RegisterEntry::Value)
                        .ok_or(TraceAbortReason::Divergence),
                    Sym::Root(slot) => Ok(RegisterEntry::Root(slot)),
                    // Dead after the exit: any value does. A register the
                    // trace writes cannot stay `Unchanged`, so it gets a
                    // constant.
                    Sym::Carried if !used_by_ops[register] && !live_at(resume, register) => {
                        Ok(if self.written[register] {
                            RegisterEntry::Const(TraceScalar::Nil)
                        } else {
                            RegisterEntry::Unchanged
                        })
                    }
                    Sym::Carried if self.written[register] => carried[register]
                        .map(RegisterEntry::Value)
                        .ok_or(TraceAbortReason::Divergence),
                    // The control register of a boxed `next` loop: the
                    // interpreter's register holds the key the entry found,
                    // and the lane holds the key now.
                    Sym::Carried
                        if self.boxed == Some(register) && frame == self.root_frame_id() =>
                    {
                        carried[register]
                            .map(RegisterEntry::Value)
                            .ok_or(TraceAbortReason::Divergence)
                    }
                    // A captured local of the root frame in a tree that writes
                    // cells through closures: no member writes the register,
                    // and the interpreter's register is the variable a write
                    // through a closure changed, so an exit leaves it be.
                    Sym::Carried
                        if side.is_some()
                            && (self.dynamic_cells || tree_dynamic)
                            && frame == self.root_frame_id()
                            && self
                                .root_captured
                                .iter()
                                .any(|(captured, _)| usize::from(*captured) == register) =>
                    {
                        Ok(RegisterEntry::Unchanged)
                    }
                    // A side's register is the parent's value, which the
                    // interpreter register may not hold yet.
                    Sym::Carried if side.is_some() => Ok(
                        carried[register].map_or(RegisterEntry::Unchanged, RegisterEntry::Value)
                    ),
                    Sym::Carried => Ok(RegisterEntry::Unchanged),
                }
            }
        };
        let map_registers = |regs: &[Sym],
                             at: (u32, usize),
                             pending: Option<(usize, usize, usize)>,
                             held: Option<(usize, usize)>,
                             values: &[Option<ValueId>],
                             (named, states): (&mut Vec<u32>, &[(u32, VirtualState)])|
         -> Result<Vec<RegisterEntry>, TraceAbortReason> {
            (0..count)
                .map(|register| {
                    map_register(
                        register,
                        sym_of(regs, register),
                        at,
                        pending,
                        held,
                        values,
                        (&mut *named, states),
                    )
                })
                .collect()
        };
        // Whether the registers of run `run` are as live at `pc` of `frame`
        // as at `other`: all that `map_register` reads of a resume point.
        // Registers past the chunk's and past the 256 a liveness set holds
        // count as live at every pc.
        let run_liveness = |frame: u32, pc: usize, run: usize| -> u64 {
            let sets = if frame == 0 {
                Some(live_in.as_slice())
            } else {
                activation_liveness
                    .iter()
                    .find(|(known, _)| *known == frame)
                    .map(|(_, sets)| sets.as_slice())
            };
            match sets.and_then(|sets| sets.get(pc)) {
                Some(set) if run < 32 => (set[run / 8] >> (8 * (run % 8))) & 0xff,
                _ => 0xff,
            }
        };
        // The recorded snapshot `snapshot_of` lowered last, and the root
        // frame's map it made of it.
        let lowered_last: RefCell<Option<(u32, RegisterMap)>> = RefCell::new(None);
        // The recipes of the tables `named` lists, and of the ones those hold.
        let named_recipes = |named: &mut Vec<u32>,
                             values: &[Option<ValueId>],
                             states: &[(u32, VirtualState)]|
         -> Result<Vec<VirtualTable>, TraceAbortReason> {
            let mut recipes = Vec::new();
            let mut position = 0;
            while position < named.len() {
                let id = named[position];
                // A table live where no snapshot named its contents
                // crosses the loop-back.
                let (_, state) = states
                    .iter()
                    .find(|(known, _)| *known == id)
                    .ok_or(TraceAbortReason::VirtualEscape)?;
                recipes.push(recipe_of(id, state, values, states, named)?);
                position += 1;
            }
            Ok(recipes)
        };
        // The registers of an inlined frame: nothing in one is carried.
        let map_inner = |regs: &[Sym],
                         values: &[Option<ValueId>],
                         (named, states): (&mut Vec<u32>, &[(u32, VirtualState)]),
                         caller: bool|
         -> Result<Vec<RegisterEntry>, TraceAbortReason> {
            regs.iter()
                .map(|sym| match sym {
                    Sym::Op(index) => values[*index as usize]
                        .map(RegisterEntry::Value)
                        .ok_or(TraceAbortReason::Divergence),
                    Sym::Root(slot) => Ok(RegisterEntry::Root(*slot)),
                    // The root frame's caller exists in the interpreter, and
                    // is told below which registers it holds.
                    Sym::Carried if caller => Ok(RegisterEntry::Unchanged),
                    Sym::Carried => Err(TraceAbortReason::Divergence),
                    Sym::Virtual(id) => virtual_entry(*id, states, values, named),
                })
                .collect()
        };
        let resolve = |operand: Operand, values: &[Option<ValueId>]| match operand {
            Operand::Carried(register) => carried[register],
            Operand::Op(index) => values[index as usize],
        };
        // A table operand is the entry input of the register the iteration
        // never writes. One it does write changes between iterations, and
        // the entry could not have resolved its storage.
        let table_input = |operand: Operand| match operand {
            Operand::Carried(register) if !is_param(register) => inputs[register]
                .map(|(value, _)| value)
                .ok_or(TraceAbortReason::UnsupportedOperand),
            // A table the iteration replaces: the next recording reaches it
            // through its lane (see [`Self::replaced_in_loop`]).
            Operand::Carried(_) if !self.rowless && self.function.is_none() => {
                self.refused_rows.set(true);
                Err(TraceAbortReason::TableMiss)
            }
            _ => Err(TraceAbortReason::UnsupportedOperand),
        };
        // The guard that leaves through the recorded snapshot `snapshot`.
        let snapshot_of = |builder: &mut TraceBuilder,
                           snapshot: u32,
                           values: &[Option<ValueId>]|
         -> Result<crate::jit_ir::trace::SnapshotId, TraceAbortReason> {
            let recorded = &self.snapshots[snapshot as usize];
            let mut named: Vec<u32> = Vec::new();
            #[cfg(debug_assertions)]
            let whole = map_registers(
                &recorded.regs,
                (recorded.root_frame, recorded.root_resume),
                recorded.pending,
                recorded.held,
                values,
                (&mut named.clone(), &recorded.virtuals),
            );
            // Snapshots in a row differ in a few registers. A run of
            // registers that holds what it held in the snapshot lowered
            // last, and is as live where this one resumes, has the entries
            // it had there (see `map_register`): only the others are
            // mapped. A run with a virtual table is always mapped, because
            // its entries number the snapshot's own recipes, and so is
            // every run of a `string.gmatch` loop, whose control entry
            // follows the values lowered so far.
            let at = (recorded.root_frame, recorded.root_resume);
            let last = lowered_last.borrow_mut().take();
            let before = last.as_ref().and_then(|(index, map)| {
                let before = &self.snapshots[*index as usize];
                (map.len() == count
                    && before.root_frame == recorded.root_frame
                    && before.pending == recorded.pending
                    && before.held == recorded.held
                    && self.gmatch_controls.is_empty())
                .then_some((before, map))
            });
            let root_registers = match before {
                Some((before, map)) => RegisterMap::made_like(map, |run, entries| {
                    let first = run * REGISTER_RUN;
                    let registers = first..first + entries.len();
                    let kept = registers.clone().all(|register| {
                        let sym = sym_of(&recorded.regs, register);
                        sym == sym_of(&before.regs, register) && !matches!(sym, Sym::Virtual(_))
                    }) && run_liveness(at.0, at.1, run)
                        == run_liveness(before.root_frame, before.root_resume, run);
                    if kept {
                        return Ok(true);
                    }
                    for (entry, register) in entries.iter_mut().zip(registers) {
                        *entry = map_register(
                            register,
                            sym_of(&recorded.regs, register),
                            at,
                            recorded.pending,
                            recorded.held,
                            values,
                            (&mut named, &recorded.virtuals),
                        )?;
                    }
                    Ok(false)
                })?,
                None => {
                    let entries = map_registers(
                        &recorded.regs,
                        at,
                        recorded.pending,
                        recorded.held,
                        values,
                        (&mut named, &recorded.virtuals),
                    )?;
                    builder.register_map(0, &entries)
                }
            };
            #[cfg(debug_assertions)]
            assert!(
                whole.is_ok_and(|whole| root_registers == whole),
                "the runs snapshot {snapshot} took from the one before differ from its own map"
            );
            *lowered_last.borrow_mut() = Some((snapshot, root_registers.clone()));
            // The registers that hold a number of run-time subtype restore
            // from the float and the tag.
            let mut numbers = Vec::new();
            // Most traces tag no number, and then no register has one.
            let tagged = if self.number_tags.is_empty() {
                &[][..]
            } else {
                &recorded.regs[..]
            };
            for (register, sym) in tagged.iter().enumerate() {
                if let Sym::Op(index) = sym
                    && let Some(tag) = self.number_tags.get(index)
                    && matches!(root_registers.get(register), Some(RegisterEntry::Value(_)))
                {
                    let tag = values
                        .get(*tag as usize)
                        .copied()
                        .flatten()
                        .ok_or(TraceAbortReason::Divergence)?;
                    numbers.push(NumberRegister {
                        register: u32::try_from(register).expect("register fits u32"),
                        is_integer: RegisterEntry::Value(tag),
                    });
                }
            }
            let mut frames = vec![SnapshotFrame {
                frame: FrameId(recorded.root_frame),
                last_interpreted_pc: last_of(recorded.prev),
                registers: root_registers.clone(),
            }];
            for inner in &recorded.inner {
                let caller = (inner.frame as usize)
                    .checked_sub(1)
                    .and_then(|position| self.inlined.get(position))
                    .and_then(|info| info.inlined)
                    .is_some_and(|call| matches!(call.entry, Entry::Caller { .. }));
                let mut registers = map_inner(
                    &inner.regs,
                    values,
                    (&mut named, &recorded.virtuals),
                    caller,
                )?;
                if caller {
                    // A register of the root frame's caller the region has
                    // not touched is the one the loop carries, or the
                    // interpreter's if no window slot stands for it.
                    for (register, entry) in registers.iter_mut().enumerate() {
                        if inner.regs[register] != Sym::Carried {
                            continue;
                        }
                        let slot = u16::try_from(register).ok().and_then(|register| {
                            self.externals
                                .iter()
                                .position(|known| *known == ExternalSlot::CallerRegister(register))
                        });
                        *entry = slot
                            .and_then(|position| carried[self.register_count + position])
                            .map_or(RegisterEntry::Unchanged, RegisterEntry::Value);
                    }
                }
                let registers = builder.register_map(frames.len(), &registers);
                frames.push(SnapshotFrame {
                    frame: FrameId(inner.frame),
                    last_interpreted_pc: last_of(inner.prev),
                    registers,
                });
            }
            let recipes = named_recipes(&mut named, values, &recorded.virtuals)?;
            let snapshot_id = builder.snapshot(
                Snapshot::new(recorded.phase, builder.effect_position(), frames)
                    .with_numbers(numbers)
                    .with_virtuals(recipes)
                    .with_counted(recorded.counted),
            );
            Ok(snapshot_id)
        };
        // The guard that leaves through the recorded snapshot `snapshot`.
        let guard_for = |builder: &mut TraceBuilder,
                         site: TraceSite,
                         snapshot: u32,
                         values: &[Option<ValueId>]|
         -> Result<crate::jit_ir::trace::GuardId, TraceAbortReason> {
            let snapshot_id = snapshot_of(builder, snapshot, values)?;
            Ok(builder.guard(site, snapshot_id))
        };
        if let Some(snapshot) = lazy_snapshot {
            let recorded = &self.snapshots[snapshot as usize];
            let pc = match recorded.phase {
                SnapshotPhase::Before { pc } | SnapshotPhase::After { pc, .. } => pc,
            };
            let site = TraceSite {
                frame: FrameId(recorded.root_frame),
                pc,
            };
            for &register in &guarded {
                let (Some((input, expected)), Some(narrowed)) =
                    (inputs[register], carried[register])
                else {
                    return Err(TraceAbortReason::Divergence);
                };
                let guard = guard_for(&mut builder, site, snapshot, &[])?;
                let value = builder.emit(
                    site,
                    TraceOp::GuardType {
                        source: input,
                        expected,
                        guard,
                    },
                    Some(expected),
                );
                if value != Some(narrowed) {
                    return Err(TraceAbortReason::Divergence);
                }
            }
        }
        let frame_params = self.loop_frame_params()?;
        let mut frame_param_values: Vec<ValueId> = Vec::new();
        for (position, op) in self.ops.iter().enumerate() {
            let site = TraceSite {
                frame: FrameId(op.frame),
                pc: op.pc,
            };
            let missing = || {
                if debug_env::trace() {
                    eprintln!("jit-trace: no value for an operand of {:?}", op.kind);
                }
                TraceAbortReason::Divergence
            };
            let value = match &op.kind {
                // A register of a seeded frame: a side input, no instruction.
                RecKind::FrameInput { frame, register } => {
                    let ty = op.ty.ok_or(TraceAbortReason::Divergence)?;
                    let input = builder.frame_input(FrameId(*frame), *register, ty);
                    // A register of a loop side's loop frame that the loop
                    // changes: its parameter stands for it from the header.
                    if frame_params
                        .iter()
                        .any(|(_, held)| *held as usize == position)
                    {
                        let param = builder.loop_param(ty, LoopParamInit::Input(input));
                        frame_param_values.push(param);
                        Some(param)
                    } else {
                        Some(input)
                    }
                }
                RecKind::Constant(scalar) => {
                    let mask = self
                        .mask_ops
                        .iter()
                        .find(|(mask_op, _)| *mask_op as usize == position)
                        .and_then(|(_, id)| self.virtuals[*id as usize].metatable);
                    let scalar = mask.map_or(*scalar, |meta| TraceScalar::Integer(meta.mask()));
                    builder.emit(site, TraceOp::Constant(scalar), op.ty)
                }
                RecKind::Move(source) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(site, TraceOp::Move { source }, op.ty)
                }
                RecKind::Negate(source) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(site, TraceOp::Negate { source }, op.ty)
                }
                RecKind::Not(source) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(site, TraceOp::Not { source }, op.ty)
                }
                RecKind::IntToFloat(source) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(site, TraceOp::IntToFloat { source }, op.ty)
                }
                RecKind::Numeric(op_kind, source) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::Numeric {
                            op: *op_kind,
                            source,
                        },
                        op.ty,
                    )
                }
                RecKind::ToWide(source, target, conversion) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::ToWide {
                            source,
                            target: *target,
                            conversion: *conversion,
                        },
                        op.ty,
                    )
                }
                RecKind::NativeUnary(function, source) => {
                    let source = resolve(*source, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::NativeUnary {
                            function: *function,
                            source,
                        },
                        op.ty,
                    )
                }
                RecKind::NativeBinary(function, left, right) => {
                    let left = resolve(*left, &values).ok_or_else(missing)?;
                    let right = resolve(*right, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::NativeBinary {
                            function: *function,
                            left,
                            right,
                        },
                        op.ty,
                    )
                }
                RecKind::Arithmetic(arithmetic, left, right) => {
                    let left = resolve(*left, &values).ok_or_else(missing)?;
                    let right = resolve(*right, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::Arithmetic {
                            op: *arithmetic,
                            left,
                            right,
                        },
                        op.ty,
                    )
                }
                RecKind::Compare(comparison, left, right) => {
                    let left = resolve(*left, &values).ok_or_else(missing)?;
                    let right = resolve(*right, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::Compare {
                            comparison: *comparison,
                            left,
                            right,
                        },
                        op.ty,
                    )
                }
                RecKind::GuardBranch {
                    condition,
                    expected,
                    snapshot,
                } => {
                    let condition = resolve(*condition, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::GuardBranch {
                            condition,
                            expected: *expected,
                            guard,
                        },
                        None,
                    )
                }
                RecKind::ArrayLoad {
                    binding,
                    table,
                    key,
                    expected,
                    past_end,
                    snapshot,
                } => {
                    let table = table_input(*table)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::ArrayLoad {
                            binding: *binding,
                            table,
                            key,
                            expected: *expected,
                            past_end: *past_end,
                            guard,
                        },
                        op.ty,
                    )
                }
                RecKind::LaneArrayLoad {
                    table,
                    field,
                    key,
                    expected,
                    snapshot,
                } => {
                    let table = resolve(*table, &values).ok_or_else(missing)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::LaneArrayLoad {
                            table,
                            field: *field,
                            key,
                            expected: *expected,
                            guard,
                        },
                        op.ty,
                    )
                }
                RecKind::NumberLoad {
                    binding,
                    table,
                    key,
                    snapshot,
                } => {
                    let table = table_input(*table)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::NumberLoad {
                            binding: *binding,
                            table,
                            key,
                            guard,
                        },
                        op.ty,
                    )
                }
                RecKind::NumberTag(load) => {
                    let load = resolve(*load, &values).ok_or_else(missing)?;
                    builder.emit(site, TraceOp::NumberTag { load }, op.ty)
                }
                RecKind::BoolAnd(left, right) => {
                    let left = resolve(*left, &values).ok_or_else(missing)?;
                    let right = resolve(*right, &values).ok_or_else(missing)?;
                    builder.emit(site, TraceOp::BoolAnd { left, right }, op.ty)
                }
                RecKind::Select(condition, if_true, if_false) => {
                    let condition = resolve(*condition, &values).ok_or_else(missing)?;
                    let if_true = resolve(*if_true, &values).ok_or_else(missing)?;
                    let if_false = resolve(*if_false, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::Select {
                            condition,
                            if_true,
                            if_false,
                        },
                        op.ty,
                    )
                }
                RecKind::NumberModulo(float, left, right, integer) => {
                    let left = resolve(*left, &values).ok_or_else(missing)?;
                    let right = resolve(*right, &values).ok_or_else(missing)?;
                    let integer = resolve(*integer, &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::NumberModulo {
                            float: *float,
                            left,
                            right,
                            integer,
                        },
                        op.ty,
                    )
                }
                RecKind::ArrayNext {
                    binding,
                    table,
                    key,
                    start,
                    ty,
                    snapshot,
                } => {
                    let table = table_input(*table)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::ArrayNext {
                            binding: *binding,
                            table,
                            key,
                            start: *start,
                            ty: *ty,
                            guard,
                        },
                        op.ty,
                    )
                }
                RecKind::HashLoad {
                    binding,
                    table,
                    key,
                    expected,
                    snapshot,
                } => {
                    let table = table_input(*table)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::HashLoad {
                            binding: *binding,
                            table,
                            key,
                            expected: *expected,
                            guard,
                        },
                        op.ty,
                    )
                }
                RecKind::ArrayStore {
                    binding,
                    table,
                    key,
                    value,
                    snapshot,
                    set,
                } => {
                    let table = table_input(*table)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let value = resolve(*value, &values).ok_or_else(missing)?;
                    // The guard's snapshot counts the effects before the
                    // store, so a failed guard leaves it uncommitted.
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    let effect = builder.next_effect();
                    let binding = *binding;
                    builder.emit(
                        site,
                        if *set {
                            TraceOp::ArraySet {
                                binding,
                                table,
                                key,
                                value,
                                effect,
                                guard,
                            }
                        } else {
                            TraceOp::ArrayStore {
                                binding,
                                table,
                                key,
                                value,
                                effect,
                                guard,
                            }
                        },
                        None,
                    )
                }
                RecKind::ArrayOverwrite {
                    binding,
                    table,
                    key,
                    value,
                } => {
                    let table = table_input(*table)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let value = resolve(*value, &values).ok_or_else(missing)?;
                    let effect = builder.next_effect();
                    builder.emit(
                        site,
                        TraceOp::ArrayOverwrite {
                            binding: *binding,
                            table,
                            key,
                            value,
                            effect,
                        },
                        None,
                    )
                }
                RecKind::FieldLoad { binding, table } => {
                    let table = table_input(*table)?;
                    builder.emit(
                        site,
                        TraceOp::FieldLoad {
                            binding: *binding,
                            table,
                        },
                        op.ty,
                    )
                }
                RecKind::FieldStore {
                    binding,
                    table,
                    value,
                } => {
                    let table = table_input(*table)?;
                    let value = resolve(*value, &values).ok_or_else(missing)?;
                    let effect = builder.next_effect();
                    builder.emit(
                        site,
                        TraceOp::FieldStore {
                            binding: *binding,
                            table,
                            value,
                            effect,
                        },
                        None,
                    )
                }
                RecKind::TableFieldLoad {
                    field,
                    table,
                    snapshot,
                } => {
                    let table = resolve(*table, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::TableFieldLoad {
                            field: *field,
                            table,
                            guard,
                        },
                        op.ty,
                    )
                }
                RecKind::TableFieldStore {
                    field,
                    table,
                    value,
                    snapshot,
                } => {
                    let table = resolve(*table, &values).ok_or_else(missing)?;
                    let value = resolve(*value, &values).ok_or_else(missing)?;
                    // The guard's snapshot counts the effects before the
                    // store, so a failed guard leaves it uncommitted.
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    let effect = builder.next_effect();
                    builder.emit(
                        site,
                        TraceOp::TableFieldStore {
                            field: *field,
                            table,
                            value,
                            effect,
                            guard,
                        },
                        None,
                    )
                }
                RecKind::LaneArrayStore {
                    table,
                    field,
                    key,
                    value,
                    snapshot,
                } => {
                    let table = resolve(*table, &values).ok_or_else(missing)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let value = resolve(*value, &values).ok_or_else(missing)?;
                    // As for a field store: a failed guard leaves the store
                    // uncommitted.
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    let effect = builder.next_effect();
                    builder.emit(
                        site,
                        TraceOp::LaneArrayStore {
                            table,
                            field: *field,
                            key,
                            value,
                            effect,
                            guard,
                        },
                        None,
                    )
                }
                RecKind::MetaGuard {
                    binding,
                    table,
                    snapshot,
                } => {
                    let table = resolve(*table, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::MetaGuard {
                            table,
                            binding: *binding,
                            guard,
                        },
                        None,
                    )
                }
                RecKind::TableLength { binding, table } => {
                    let table = table_input(*table)?;
                    builder.emit(
                        site,
                        TraceOp::TableLength {
                            binding: *binding,
                            table,
                        },
                        op.ty,
                    )
                }
                // A table that escaped is made for real, which counts by
                // itself: nothing stands for it.
                RecKind::PhantomTable { virt } if self.virtuals[*virt as usize].real.is_some() => {
                    None
                }
                RecKind::PhantomTable { virt } => builder.emit(
                    site,
                    TraceOp::PhantomTable {
                        entries: u32::try_from(self.virtuals[*virt as usize].peak)
                            .expect("a virtual table's size fits u32"),
                    },
                    None,
                ),
                RecKind::StringLength { binding, string } => {
                    let string = table_input(*string)?;
                    builder.emit(
                        site,
                        TraceOp::StringLength {
                            binding: *binding,
                            string,
                        },
                        op.ty,
                    )
                }
                RecKind::HostCall {
                    function,
                    args,
                    result,
                    snapshot,
                } => {
                    let args = args
                        .iter()
                        .map(|arg| resolve(*arg, &values).ok_or_else(missing))
                        .collect::<Result<Vec<_>, _>>()?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::HostCall {
                            function: *function,
                            args,
                            result: *result,
                            guard,
                        },
                        Some(*result),
                    )
                }
                RecKind::CallHelper {
                    helper,
                    args,
                    snapshot,
                } => {
                    let args = args
                        .iter()
                        .map(|arg| resolve(*arg, &values).ok_or_else(missing))
                        .collect::<Result<Vec<_>, _>>()?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    // The guard's snapshot counts the effects before the call,
                    // so a miss leaves the heap unwritten.
                    let effect = helper
                        .effects()
                        .commits_effect()
                        .then(|| builder.next_effect());
                    let result = helper.info().and_then(|info| info.results.first());
                    builder.emit(
                        site,
                        TraceOp::CallHelper {
                            helper: *helper,
                            args,
                            effect,
                            guard,
                        },
                        result.map(|result| result.trace_type()),
                    )
                }
                RecKind::HelperResult { call, index } => {
                    let call = resolve(Operand::Op(*call), &values).ok_or_else(missing)?;
                    builder.emit(
                        site,
                        TraceOp::HelperResult {
                            call,
                            index: *index,
                        },
                        op.ty,
                    )
                }
                RecKind::SelfCall { args, snapshot } => {
                    let args = args
                        .iter()
                        .map(|arg| resolve(*arg, &values).ok_or_else(missing))
                        .collect::<Result<Vec<_>, _>>()?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    let ty = op
                        .ty
                        .or_else(|| function.and_then(|function| function.result))
                        .ok_or(TraceAbortReason::Divergence)?;
                    builder.emit(site, TraceOp::SelfCall { args, guard }, Some(ty))
                }
                RecKind::StringByte {
                    binding,
                    string,
                    key,
                    snapshot,
                } => {
                    let string = table_input(*string)?;
                    let key = resolve(*key, &values).ok_or_else(missing)?;
                    let guard = guard_for(&mut builder, site, *snapshot, &values)?;
                    builder.emit(
                        site,
                        TraceOp::StringByte {
                            binding: *binding,
                            string,
                            key,
                            guard,
                        },
                        op.ty,
                    )
                }
            };
            values.push(value);
        }

        if let Some(function) = function {
            return self.finish_function(builder, function, &values, &carried, guest_registers);
        }
        // The loop-back drops the lanes of the tables the trace made, so no
        // register may carry one across it (a callee returned a table it
        // had stored somewhere, say). A register the header observes counts,
        // named locals in scope and debug.getlocal included: a snapshot of
        // the next iteration restores it before the loop rewrites it. A side
        // that exits resumes the interpreter where it stopped instead.
        let resume = self.root_resume();
        let header_live = self
            .liveness
            .get_or_init(|| observable_registers(&self.chunk))
            .get(resume);
        if self.regs.iter().enumerate().any(|(register, sym)| {
            matches!(sym, Sym::Op(index) if self.is_made_table(*index))
                && u8::try_from(register).map_or(true, |narrow| {
                    header_live.is_none_or(|set| register_set_contains(set, narrow))
                })
        }) {
            return Err(TraceAbortReason::VirtualEscape);
        }
        // The state a poll publishes: the iteration's final registers with
        // the closing instruction committed, resumed at the header. For a
        // side trace the same state is what it links the loop with.
        let mut loop_named = Vec::new();
        let mut final_registers = map_registers(
            &self.regs,
            (0, resume),
            None,
            None,
            &values,
            (&mut loop_named, &[]),
        )?;
        // A table live at the loop-back crosses the iteration.
        if !loop_named.is_empty() {
            return Err(TraceAbortReason::VirtualEscape);
        }
        if let Some(seed) = side.filter(|_| self.exit_at.is_none() && !self.looping) {
            let header_site = TraceSite {
                frame: FrameId(0),
                pc: self.header,
            };
            // A constant string of the roots has no type at the link, and the
            // tree may widen the root with the register it sits in: the
            // header gives it one, as its lane, in every register.
            for entry in &mut final_registers {
                if let RegisterEntry::Root(slot) = *entry
                    && let Some(value) = builder.emit(
                        header_site,
                        TraceOp::Constant(TraceScalar::String(
                            ROOT_STRING_LANE_BASE + u64::from(slot),
                        )),
                        Some(TraceType::String),
                    )
                {
                    *entry = RegisterEntry::Value(value);
                }
            }
            for &(register, ty) in &seed.params {
                let index = usize::from(register);
                // A register the loop writes before it reads again still
                // has a parameter, because snapshots inside inlined calls
                // restore it. Any value of its type will do.
                let dead = !live_at(self.header, index);
                let mut entry = match final_registers.get(index) {
                    Some(entry @ (RegisterEntry::Value(_) | RegisterEntry::Const(_))) => {
                        Some(*entry)
                    }
                    _ => carried
                        .get(index)
                        .copied()
                        .flatten()
                        .map(RegisterEntry::Value),
                };
                let held_by = |entry: Option<RegisterEntry>, builder: &TraceBuilder| match entry {
                    Some(RegisterEntry::Value(value)) => builder_value_type(builder, value),
                    Some(RegisterEntry::Const(scalar)) => Some(scalar.ty()),
                    _ => None,
                };
                if dead && held_by(entry, &builder) != Some(ty) {
                    // What the side holds there, if it is of the type.
                    let own = match self.regs.get(index) {
                        Some(Sym::Op(op)) => values[*op as usize].map(RegisterEntry::Value),
                        _ => None,
                    };
                    entry = match ty {
                        _ if held_by(own, &builder) == Some(ty) => own,
                        TraceType::Nil => Some(RegisterEntry::Const(TraceScalar::Nil)),
                        TraceType::Boolean => {
                            Some(RegisterEntry::Const(TraceScalar::Boolean(false)))
                        }
                        TraceType::Integer => Some(RegisterEntry::Const(TraceScalar::Integer(0))),
                        TraceType::Float => Some(RegisterEntry::Const(TraceScalar::Float(0.0))),
                        TraceType::String if !self.root_constants.is_empty() => builder
                            .emit(
                                header_site,
                                TraceOp::Constant(TraceScalar::String(ROOT_STRING_LANE_BASE)),
                                Some(TraceType::String),
                            )
                            .map(RegisterEntry::Value),
                        _ => entry,
                    };
                }
                let Some(entry) = entry else {
                    return Err(TraceAbortReason::TypeUnstable);
                };
                final_registers[index] = entry;
                if held_by(Some(entry), &builder) != Some(ty) {
                    return Err(TraceAbortReason::TypeUnstable);
                }
            }
        }
        // The values the poll publishes, for the strings a release keeps.
        let poll_values: Vec<ValueId> = final_registers
            .iter()
            .filter_map(|entry| match entry {
                RegisterEntry::Value(value) => Some(*value),
                _ => None,
            })
            .collect();
        // A side that rejoins an ancestor loop side, when the link binds.
        let link_target = self
            .link_loop
            .map(|position| &self.loop_headers[position])
            .filter(|held| self.loop_link_binds(held, &builder, &final_registers))
            .map(|held| held.side);
        // A loop through tail calls enters its header like a call: before
        // its first instruction, with no previous pc.
        // The exit in front of a loop in a call rebuilds the call's frames.
        let poll = if let Some(recorded) = self.terminal {
            snapshot_of(&mut builder, recorded, &values)?
        } else {
            builder.snapshot(
                Snapshot::new(
                    match last_pc {
                        Some(pc) => SnapshotPhase::After {
                            pc,
                            successor: resume,
                        },
                        None => SnapshotPhase::Before { pc: resume },
                    },
                    builder.effect_position(),
                    vec![SnapshotFrame {
                        frame: FrameId(0),
                        last_interpreted_pc: last_pc,
                        registers: builder.register_map(0, &final_registers),
                    }],
                )
                .with_counted(self.counted),
            )
        };
        if side.is_some() && !self.looping {
            if let Some(target) = link_target {
                return self.verified(builder.finish(TraceTerminator::LinkLoop {
                    side: target,
                    snapshot: poll,
                }));
            }
            if self.exit_at.is_some() {
                return self.verified(builder.finish(TraceTerminator::Exit { snapshot: poll }));
            }
            return self.finish_side(builder, poll);
        }
        // A root that ends in front of a loop inside its own: the terminal
        // exit resumes the interpreter at that loop's header, where a loop
        // side of the tree continues it. The root keeps a poll snapshot of
        // its own header's shape, which no run takes (the root never loops
        // back by itself, only the sides that link do), for the loop span
        // the exits and the sides are planned against.
        if side.is_none() && self.exit_at.is_some() && self.enter_loop.is_some() {
            let header_registers = map_registers(
                &self.regs,
                (0, self.header),
                None,
                None,
                &values,
                (&mut Vec::new(), &[]),
            )?;
            let own_poll = builder.snapshot(
                Snapshot::new(
                    SnapshotPhase::After {
                        pc: self.latch,
                        successor: self.header,
                    },
                    builder.effect_position(),
                    vec![SnapshotFrame {
                        frame: FrameId(0),
                        last_interpreted_pc: Some(self.latch),
                        registers: builder.register_map(0, &header_registers),
                    }],
                )
                .with_counted(self.counted),
            );
            builder.set_poll_snapshot(own_poll);
            return self.verified(builder.finish(TraceTerminator::Exit { snapshot: poll }));
        }
        builder.set_poll_snapshot(poll);

        let mut args = Vec::with_capacity(parameters.len());
        for &register in &parameters {
            let value = match self.regs[register] {
                Sym::Carried => carried[register],
                // A root string constant that went through an inlined call
                // is an operation, but carried it is the same root lane.
                Sym::Op(index)
                    if used_by_ops[register]
                        && self.root_constant_bytes(Operand::Op(index)).is_some() =>
                {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                // A table a host call gave has a lane the release at the
                // loop-back drops with the tables the run made.
                Sym::Op(index) if self.is_call_table(index) => {
                    self.refused_vmcall.set(true);
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                Sym::Op(index) => values[index as usize],
                // The constant rides as its root lane. An operation that read
                // the previous iteration's string would read a lane that is
                // not the entry's, so a register the operations read is not
                // taken.
                Sym::Root(slot) if !used_by_ops[register] => {
                    let constant = TraceScalar::String(ROOT_STRING_LANE_BASE + u64::from(slot));
                    builder.emit(
                        TraceSite {
                            frame: FrameId(0),
                            pc: self.header,
                        },
                        TraceOp::Constant(constant),
                        Some(TraceType::String),
                    )
                }
                Sym::Root(_) => return Err(TraceAbortReason::UnsupportedOperand),
                // The previous iteration's table is this one's input.
                // Only an exit reads the register, and the loop-back drops the
                // table (it is dead at the header, or `final_registers`
                // rejected it). The register keeps its entry value.
                Sym::Virtual(_) if !used_by_ops[register] => carried[register],
                Sym::Virtual(_) => return Err(TraceAbortReason::VirtualEscape),
            }
            .ok_or(TraceAbortReason::Divergence)?;
            args.push(value);
        }
        // The loop frame's parameters, in the order the frame inputs
        // declared them.
        if let Some(frame) = self.frames.last().filter(|_| !frame_params.is_empty()) {
            for &(register, _) in &frame_params {
                let Some(Sym::Op(index)) = frame.regs.get(register) else {
                    return Err(TraceAbortReason::UnsupportedOperand);
                };
                args.push(values[*index as usize].ok_or(TraceAbortReason::Divergence)?);
            }
        }
        if frame_param_values.len() != frame_params.len() {
            return Err(TraceAbortReason::Divergence);
        }
        // A loop that makes strings (or iterators) lets the VM drop the ones
        // the next iteration cannot name, so the run does not keep them all.
        if self.ops.iter().any(|op| match &op.kind {
            RecKind::HostCall { result, .. } => *result == TraceType::String,
            // An iterator takes a lane of the same bank as a string does.
            RecKind::CallHelper { helper, .. } => {
                *helper == HelperId::GMATCH_NEW
                    || *helper == HelperId::TABLE_NEW
                    || (helper.is_vm_call()
                        && helper
                            .info()
                            .is_some_and(|info| info.results.contains(&HelperType::Table)))
                    || helper
                        .info()
                        .is_some_and(|info| info.results.contains(&HelperType::String))
            }
            _ => false,
        }) {
            // A loop side would have to keep the lanes its iterations make
            // for the root's loop to release.
            if self.looping {
                self.refused_vmcall.set(true);
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            // A closure is an iterator lane only when this trace makes or
            // steps iterators; keeping a plain closure would only use up
            // the keep list.
            let makes_iterators = self.ops.iter().any(|op| {
                matches!(&op.kind, RecKind::CallHelper { helper, .. }
                    if helper.uses_iterator())
            });
            let mut keep: Vec<ValueId> = Vec::new();
            for value in args.iter().chain(&poll_values) {
                // A string, or an iterator an outer iteration made and the
                // next one steps.
                let ty = builder_value_type(&builder, *value);
                let kept = matches!(ty, Some(TraceType::String))
                    || (makes_iterators && matches!(ty, Some(TraceType::Closure)));
                if kept && !keep.contains(value) {
                    keep.push(*value);
                }
            }
            if keep.len() <= MAX_HOST_ARGS {
                builder.emit(
                    TraceSite {
                        frame: FrameId(0),
                        pc: self.header,
                    },
                    TraceOp::ReleaseStrings { keep },
                    None,
                );
            }
        }
        let ir = builder.finish(TraceTerminator::LoopBack { args });
        // A register that changes type cannot ride in a typed parameter.
        if !loop_types_are_stable(&ir) {
            if debug_env::trace() {
                for (pc, instruction) in self.chunk.code.iter().enumerate() {
                    eprintln!("  {pc}: {instruction:?}");
                }
                eprintln!(
                    "jit-trace: header {} type-unstable: params {:?} args {:?} (registers {:?})",
                    self.header,
                    ir.loop_params
                        .iter()
                        .map(|param| param.ty)
                        .collect::<Vec<_>>(),
                    match &ir.terminator {
                        TraceTerminator::LoopBack { args } => args
                            .iter()
                            .map(|arg| ir.value_type(*arg))
                            .collect::<Vec<_>>(),
                        TraceTerminator::Exit { .. }
                        | TraceTerminator::Link { .. }
                        | TraceTerminator::LinkLoop { .. }
                        | TraceTerminator::Return { .. }
                        | TraceTerminator::TailCall { .. } => Vec::new(),
                    },
                    parameters,
                );
            }
            return Err(TraceAbortReason::TypeUnstable);
        }
        self.verified(ir)
    }
}

/// The type of a value a builder has defined.
fn builder_value_type(builder: &TraceBuilder, value: ValueId) -> Option<TraceType> {
    builder.value_type(value)
}

impl Recorder {
    /// Closes a function trace at its return: the function's types, and for
    /// a root the entry state an activation polls through (its parameters at
    /// pc 0, nil, and the cells and globals as the interpreter holds them).
    /// A table the entry binds from a parameter must reach every nested
    /// activation unchanged, since they all use the entered one's bindings.
    fn finish_function(
        &self,
        mut builder: TraceBuilder,
        function: &FunctionSeed,
        values: &[Option<ValueId>],
        carried: &[Option<ValueId>],
        guest_registers: usize,
    ) -> Result<TraceIr, TraceAbortReason> {
        let parameters = function.parameters.len();
        for binding in &self.bindings {
            let register = usize::from(binding.register);
            if register >= parameters {
                continue;
            }
            let passes = |args: &[Operand]| {
                args.get(register)
                    .is_some_and(|arg| self.origin(*arg) == Operand::Carried(register))
            };
            let passed_on = self.ops.iter().all(|op| match &op.kind {
                RecKind::SelfCall { args, .. } => passes(args),
                _ => true,
            }) && self.tail.as_ref().is_none_or(|(args, _)| passes(args));
            if !passed_on || self.written[register] {
                return Err(TraceAbortReason::SelfCall);
            }
        }
        let resolve = |operand: Operand| match operand {
            Operand::Carried(register) => carried[register],
            Operand::Op(index) => values[index as usize],
        };
        let terminator = match (&self.tail, self.returned) {
            (Some((args, pc)), None) => TraceTerminator::TailCall {
                args: args
                    .iter()
                    .map(|arg| resolve(*arg))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(TraceAbortReason::Divergence)?,
                pc: *pc,
            },
            (None, Some(returned)) => TraceTerminator::Return {
                value: Some(resolve(returned).ok_or(TraceAbortReason::Divergence)?),
            },
            _ => return Err(TraceAbortReason::Divergence),
        };
        let count_register = self
            .externals
            .iter()
            .position(|slot| *slot == ExternalSlot::TailCalls)
            .map(|position| guest_registers + position);
        builder.set_function(FunctionTrace {
            parameters: function.parameters.clone(),
            result: function.result,
            callee_registers: self.callee_registers.clone(),
        });
        if self.side.is_none() {
            let registers = (0..self.regs.len())
                .map(|register| {
                    if register < parameters {
                        carried[register]
                            .map(RegisterEntry::Value)
                            .ok_or(TraceAbortReason::Divergence)
                    } else if register < guest_registers {
                        Ok(RegisterEntry::Const(TraceScalar::Nil))
                    } else if Some(register) == count_register {
                        carried[register]
                            .map(RegisterEntry::Value)
                            .ok_or(TraceAbortReason::Divergence)
                    } else {
                        Ok(RegisterEntry::Unchanged)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let entry = builder.snapshot(Snapshot::new(
                SnapshotPhase::Before { pc: 0 },
                0,
                vec![SnapshotFrame {
                    frame: FrameId(0),
                    last_interpreted_pc: None,
                    registers: builder.register_map(0, &registers),
                }],
            ));
            builder.set_poll_snapshot(entry);
        }
        self.verified(builder.finish(terminator))
    }

    /// The side of the root latch's guard when it resumes at the root's own
    /// header: the loop goes on where the parent's recording ended it, which
    /// happens when a side recorded the last trip of the root's loop and ran
    /// through the loops around it. There is nothing to record, so the side
    /// links back at once, as `LuaJIT`'s empty side trace does. Recording an
    /// iteration from the header instead copied the root's, and when that
    /// copy aborted (a call the root's own iteration did not make) every
    /// later failure of the guard left for the interpreter: the Mineclonia
    /// startup ran +37% one run in eight, depending on which match of
    /// `locate_biome_in_area` the side happened to record.
    ///
    /// Only the guard of the root's latch qualifies, failing after the latch
    /// ran: the iteration it left is complete, so the root starts the next
    /// one. Any other guard that resumes at the header failed at the top of
    /// the next iteration (the step of a generic for, whose snapshot is the
    /// state after the latch because nothing ran since), where the
    /// interpreter runs the instruction the root could not; an empty side
    /// there would hand the same state back to the same guard forever.
    /// The latch must be a numeric for or the conditional branch of a
    /// `repeat`, whose guard is the whole of the latch. A generic for's step
    /// sits next to its latch, so a guard there may have failed before the
    /// step ran; such a side records as before. So does a loop side (the
    /// seed of a terminal exit), which starts at a loop of its own.
    ///
    /// `None` when the side starts anywhere else, or when the state cannot
    /// link (a loop parameter of another type): the side then records an
    /// iteration as before.
    fn close_side_at_start(&self, at: &HookPoint<'_>, id: TraceId) -> Option<Step> {
        let seed = self.side.as_ref()?;
        let site = self.latch_guard?;
        if at.pc != self.header
            || seed.last != Some(self.latch)
            || seed.looping
            || seed.function.is_some()
            || seed.entered.is_some()
            || self.function.is_some()
            || !self.in_home()
            || self.loop_depth > 0
        {
            return None;
        }
        // The policy passes only the guard at the latch. A guard that came
        // to stand at the latch some other way (a step folded into it) would
        // hand the root the state it failed on, forever.
        debug_assert_eq!(
            site, self.latch,
            "only the latch's own guard links back at once"
        );
        let latch = at.chunk.code.get(self.latch)?.opcode().ok()?;
        if !matches!(latch, OpCode::NumericForLoop | OpCode::JumpIfFalse) {
            return None;
        }
        let ir = self.finalize(id).ok()?;
        Some(Step::Closed(Box::new(ClosedTrace {
            guest_instructions: 0,
            slot_reads_forwarded: 0,
            slot_overwrites: 0,
            header_pc: self.header,
            crossed: Vec::new(),
            remaining_trips: self.remaining_trips,
            run_trips: self.run_trips,
            remaining_work: self.remaining_work,
            nest_plan: self.nest_plan,
            entered_loop: None,
            ir,
        })))
    }

    /// Closes a side trace: the link snapshot ends it, and it verifies on its
    /// own. Whether it fits the tree is the tree's check.
    /// Whether a side that came round to the header of the ancestor loop
    /// side `held` can rejoin its loop with the final state `registers`: a
    /// value of the right type for each loop parameter, and no register the
    /// loop side reads from its entry state written on the way (by this side
    /// or by the sides between it and the loop side). Otherwise the side ends
    /// in front of the loop, which a copy of the loop continues.
    fn loop_link_binds(
        &self,
        held: &SeedLoop,
        builder: &TraceBuilder,
        registers: &[RegisterEntry],
    ) -> bool {
        let typed = held.params.iter().all(|&(register, ty)| {
            let entry = registers.get(usize::from(register));
            match entry {
                Some(RegisterEntry::Value(value)) => {
                    builder_value_type(builder, *value) == Some(ty) && !ty.is_borrowed()
                }
                Some(RegisterEntry::Const(scalar)) => scalar.ty() == ty,
                _ => false,
            }
        });
        let untouched = held.invariant.iter().all(|&register| {
            let by_sides = held.written.contains(&register);
            let here = self.written.get(usize::from(register)).copied();
            !by_sides && !here.unwrap_or(true)
        });
        typed && untouched
    }

    fn finish_side(
        &self,
        builder: TraceBuilder,
        link: crate::jit_ir::trace::SnapshotId,
    ) -> Result<TraceIr, TraceAbortReason> {
        let ir = builder.finish(TraceTerminator::Link { snapshot: link });
        self.verified(ir)
    }

    /// `ir` if it verifies. A recorder that builds a trace the verifier
    /// refuses has a bug, which `OXIMOND_JIT_TRACE` reports.
    fn verified(&self, ir: TraceIr) -> Result<TraceIr, TraceAbortReason> {
        // A table set beside storage a resize would move is a recording the
        // iteration made, not a recorder bug: another may not touch both. A
        // set on a table that may still become a list (a ring filling up) is
        // one a later recording may keep.
        if (ir.moves_hash_storage_outside_vm_calls() || ir.deletes_keys())
            && let Some(detail) = ir.storage_a_set_would_move()
        {
            if debug_env::trace() {
                eprintln!("jit-trace: recording refused: {detail}");
            }
            // Another recording that reads fields through lookups can hold
            // both.
            if !self.rowless && !self.set_waiting && self.function.is_none() {
                self.refused_rows.set(true);
            }
            return Err(if self.set_waiting || self.refused_rows.get() {
                TraceAbortReason::TableMiss
            } else {
                TraceAbortReason::UnsupportedOperand
            });
        }
        // A table a host call gave has a lane the next release drops with
        // the tables the run made, which a table that holds it would outlive.
        // The recording that stores it into a table is refused, and the next
        // one aborts at the call.
        if self.ops.iter().any(|op| {
            let (RecKind::ArrayStore { value, .. }
            | RecKind::ArrayOverwrite { value, .. }
            | RecKind::FieldStore { value, .. }
            | RecKind::TableFieldStore { value, .. }
            | RecKind::LaneArrayStore { value, .. }) = &op.kind
            else {
                return false;
            };
            matches!(value, Operand::Op(index) if self.is_call_table(*index))
        }) {
            self.refused_vmcall.set(true);
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // Likewise a metatable change beside a fact about a metatable.
        if ir.stale_metatable_fact().is_some() {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A coroutine the trace resumes runs code that may write anything the
        // trace holds across the call. The same goes for a local of the root
        // frame that a closure captured and the trace writes: the register is
        // the variable.
        let resumes = ir.resumes();
        // A host call the trace makes is such a resume, and the next
        // recording of the loop aborts at the call.
        let calls_host = ir.makes_vm_calls();
        // A field binding keeps the address of a slot in native code, which
        // a host call may move. The loop is recorded again with the fields
        // read through the lanes of their tables.
        if calls_host && !self.lane_fields && ir.vm_call_beside_field_bindings() {
            self.wants_lane_fields.set(true);
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if let Some(detail) = ir.resume_conflict() {
            if debug_env::trace() {
                eprintln!("jit-trace: recording refused: {detail}");
            }
            if calls_host {
                self.refused_vmcall.set(true);
            }
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // A host function call checks a captured local the trace writes
        // again after the call, as it does a cell.
        if resumes
            && !ir.resumes_only_in_vm_calls()
            && self.root_captured.iter().any(|(register, _)| {
                ir.written_registers
                    .contains(&(FrameId(0), u16::from(*register)))
            })
        {
            if debug_env::trace() {
                eprintln!("jit-trace: recording refused: a resume beside a captured local");
            }
            if calls_host {
                self.refused_vmcall.set(true);
            }
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        // The calls cost more than the interpreter's; a loop with little
        // else in it stays with the interpreter.
        if calls_host && self.guest < self.limits.vmcall_work {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: recording refused: {} guest instructions beside a host call, {} wanted",
                    self.guest, self.limits.vmcall_work
                );
            }
            self.refused_vmcall.set(true);
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        if debug_env::trace() {
            eprintln!(
                "jit-trace: recorded {} guest instructions with {} snapshots, {} guarded ({} array reads forwarded, {} stores proven)",
                self.guest,
                self.snapshots.len(),
                self.guarded_snapshots,
                self.slot_reads_forwarded,
                self.slot_overwrites
            );
        }
        match ir.verify() {
            Ok(()) => Ok(ir),
            Err(error) => {
                if debug_env::trace() {
                    eprintln!("jit-trace: recorded trace failed verification: {error:?}");
                    if debug_env::trace_ir() {
                        eprintln!("jit-trace: rejected {ir:#?}");
                    }
                }
                // A trace that calls host functions itself may fail on what
                // that implies (a global slot beside effects); the next
                // recording aborts at the call.
                if calls_host {
                    self.refused_vmcall.set(true);
                }
                Err(TraceAbortReason::CompileRejected)
            }
        }
    }
}

/// Whether `pattern` has no special character, so that `string.find` looks
/// for it as plain text.
fn pattern_is_plain(pattern: &[u8]) -> bool {
    !pattern.iter().any(|byte| b"^$*+?.([%-".contains(byte))
}

/// Whether `string.find (haystack, pattern, init, plain)` finds the pattern,
/// when it looks for plain text (`plain` set, or no special character in the
/// pattern). `None` for a call the recorder does not predict: patterns, an
/// `init` that is not an integral number, a `plain` that is no boolean.
fn predict_find(live: &[Value]) -> Option<bool> {
    let (Value::String(haystack), Value::String(pattern)) = (live.first()?, live.get(1)?) else {
        return None;
    };
    let init = match live.get(2) {
        None | Some(Value::Nil) => 1_i64,
        Some(Value::Integer(init)) => *init,
        #[allow(clippy::cast_possible_truncation)]
        Some(Value::Number(init)) if init.fract() == 0.0 && init.abs() < 9.0e15 => *init as i64,
        _ => return None,
    };
    let plain = match live.get(3) {
        None | Some(Value::Nil | Value::Boolean(false)) => false,
        Some(Value::Boolean(true)) => true,
        _ => return None,
    };
    if !plain && !pattern_is_plain(pattern.as_bytes()) {
        return None;
    }
    let length = i64::try_from(haystack.len()).ok()?;
    let start = if init > 0 {
        init
    } else if init == 0 || init < -length {
        1
    } else {
        length + init + 1
    };
    if start > length + 1 {
        return Some(false);
    }
    let tail = &haystack.as_bytes()[usize::try_from(start - 1).ok()?..];
    let needle = pattern.as_bytes();
    Some(needle.is_empty() || tail.windows(needle.len()).any(|window| window == needle))
}

/// How many bytes `string.byte (s [, i [, j]])` gives for the live
/// arguments, when there are at most four and the indices are integral
/// numbers. `None` for any other call.
fn predict_byte(live: &[Value]) -> Option<usize> {
    let Some(Value::String(source)) = live.first() else {
        return None;
    };
    let index = |value: Option<&Value>, default: i64| -> Option<i64> {
        match value {
            None => Some(default),
            Some(Value::Integer(index)) => Some(*index),
            #[allow(clippy::cast_possible_truncation)]
            Some(Value::Number(index)) if index.fract() == 0.0 && index.abs() < 9.0e15 => {
                Some(*index as i64)
            }
            _ => None,
        }
    };
    let length = isize::try_from(source.len()).ok()?;
    let first = index(live.get(1), 1)?;
    let last = index(live.get(2), first)?;
    let (first, last) = (isize::try_from(first).ok()?, isize::try_from(last).ok()?);
    // The interpreter's own clamp, so the prediction cannot disagree with it.
    let count = substring_range(length, first, last).map_or(0, |range| range.len());
    Some(count).filter(|count| *count <= 4)
}

impl Recorder {
    /// Whether the callee in register `callee` is a function a trace may call
    /// through the VM, so that its string constants are readable arguments.
    fn is_host_callee(at: &HookPoint<'_>, callee: usize) -> bool {
        match &at.registers[callee] {
            Value::BuiltinFunction(
                BuiltinFunction::ToString
                | BuiltinFunction::StringFormat
                | BuiltinFunction::StringGsub,
            ) => true,
            Value::NativeFunction(function) => at
                .surroundings
                .and_then(|around| around.runtime.direct_native_kind(*function))
                .is_some_and(|kind| {
                    matches!(
                        kind,
                        DirectNativeKind::Find
                            | DirectNativeKind::Match
                            | DirectNativeKind::Gmatch
                            | DirectNativeKind::TableConcat
                            | DirectNativeKind::Format
                            | DirectNativeKind::StringFormatLuaJit
                            | DirectNativeKind::StringChar
                            | DirectNativeKind::StringLen
                            | DirectNativeKind::MathMax
                            | DirectNativeKind::MathMin
                            | DirectNativeKind::Type
                            | DirectNativeKind::MathType
                            | DirectNativeKind::RawLen
                            | DirectNativeKind::GetMetatable
                            | DirectNativeKind::SetMetatable
                            | DirectNativeKind::TableInsert
                            | DirectNativeKind::TableInsertRaw
                            | DirectNativeKind::StringRep
                            | DirectNativeKind::StringUpper
                            | DirectNativeKind::StringLower
                            | DirectNativeKind::StringReverse
                            | DirectNativeKind::Tonumber
                            | DirectNativeKind::StringSlice
                            | DirectNativeKind::StringByte
                    ) || Self::is_text_kind(kind)
                }),
            _ => false,
        }
    }

    /// Like [`Self::read`], and a string constant the interpreter loaded is
    /// a value too: a host call reads it from the trace's root strings, so
    /// the operand is a constant naming its slot.
    fn host_operand(
        &mut self,
        pc: usize,
        registers: &[Value],
        register: usize,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        self.check_register(register)?;
        if let Sym::Root(slot) = self.sym(register) {
            if !matches!(registers[register], Value::String(_)) {
                return Err(TraceAbortReason::Divergence);
            }
            return Ok(self.root_operand(pc, slot));
        }
        self.read(registers, register)
    }

    /// Like [`Self::read_for_comparison`], and a string constant the
    /// interpreter loaded is its root string, which an equality compares by
    /// bytes.
    fn read_compared(
        &mut self,
        pc: usize,
        registers: &[Value],
        register: usize,
    ) -> Result<(Operand, TraceType), TraceAbortReason> {
        self.check_register(register)?;
        if let Sym::Root(_) = self.sym(register) {
            return self.host_operand(pc, registers, register);
        }
        self.read_for_comparison(registers, register)
    }

    /// The operation holding the constant root string `slot`.
    fn root_constant(&mut self, pc: usize, slot: u32) -> u32 {
        self.emit(
            pc,
            RecKind::Constant(TraceScalar::String(ROOT_STRING_LANE_BASE + u64::from(slot))),
            Some(TraceType::String),
        )
    }

    /// The constant operation naming root string `slot`.
    fn root_operand(&mut self, pc: usize, slot: u32) -> (Operand, TraceType) {
        let constant = self.emit(
            pc,
            RecKind::Constant(TraceScalar::String(ROOT_STRING_LANE_BASE + u64::from(slot))),
            Some(TraceType::String),
        );
        (Operand::Op(constant), TraceType::String)
    }

    /// Emits the call of `function` on `args`, with the `Before` snapshot of
    /// the instruction at `pc` for the guard that leaves when the VM cannot
    /// answer. Returns the operation holding the answer.
    fn emit_host_call(
        &mut self,
        pc: usize,
        function: HostFunction,
        args: &[(Operand, TraceType)],
        result: TraceType,
    ) -> u32 {
        // A call into the library costs the interpreter several guest
        // instructions' worth of host instructions: the policy weighs the
        // loop by it.
        self.host_work = self.host_work.saturating_add(HOST_CALL_GUEST_WORK);
        let snapshot = self.snapshot(SnapshotPhase::Before { pc }, &[]);
        self.emit(
            pc,
            RecKind::HostCall {
                function,
                args: args.iter().map(|(operand, _)| *operand).collect(),
                result,
                snapshot,
            },
            Some(result),
        )
    }

    /// Emits the call of `helper` on `args` ([`TraceOp::CallHelper`]), with
    /// the `Before` snapshot of the instruction at `pc` for the guard that
    /// leaves when the helper misses. Returns the operation holding the
    /// helper's first result.
    fn emit_helper_call(
        &mut self,
        pc: usize,
        helper: HelperId,
        args: &[(Operand, TraceType)],
    ) -> u32 {
        // A helper is cheaper than the interpreter's call, but still more
        // than a guest instruction.
        self.host_work = self.host_work.saturating_add(HOST_CALL_GUEST_WORK);
        // A check of which metamethod runs, inside an inlined call that has
        // stored nothing, starts the call over: the access that went another
        // way gets a side trace in the root frame.
        let snapshot = if matches!(
            helper,
            HelperId::INDEX_HANDLER
                | HelperId::NEWINDEX_HANDLER
                | HelperId::LEN_HANDLER
                | HelperId::OPERATOR_HANDLER
                | HelperId::TABLE_ABSENT_INDEX
        ) {
            self.read_snapshot(pc)
        } else {
            self.snapshot(SnapshotPhase::Before { pc }, &[])
        };
        let result = helper
            .info()
            .and_then(|info| info.results.first())
            .map(|result| result.trace_type());
        self.emit(
            pc,
            RecKind::CallHelper {
                helper,
                args: args.iter().map(|(operand, _)| *operand).collect(),
                snapshot,
            },
            result,
        )
    }

    /// Emits the call of the library helper `helper` on `args` and returns
    /// the operation holding its answer, which has type `result`. A float
    /// answer is the helper's integer converted, as the profile has numbers;
    /// a nil answer is a helper that gives none (it misses when the call
    /// would have given something).
    fn emit_library_helper(
        &mut self,
        pc: usize,
        helper: HelperId,
        args: &[(Operand, TraceType)],
        result: TraceType,
    ) -> u32 {
        let value = self.emit_helper_call(pc, helper, args);
        match result {
            TraceType::Nil => self.emit(
                pc,
                RecKind::Constant(TraceScalar::Nil),
                Some(TraceType::Nil),
            ),
            TraceType::Float => {
                self.emit(pc, RecKind::IntToFloat(Operand::Op(value)), Some(result))
            }
            _ => value,
        }
    }

    /// The byte count of the string `operand` of type `ty` as a number of
    /// the profile. A string the trace holds by lane has its length in a
    /// table native code reads inline (a [`HostFunction::StringLength`]
    /// the backend does not call). The strings of the trace's roots and the
    /// ones borrowed from a table slot are not in that table: they take a
    /// helper.
    fn emit_string_length(&mut self, pc: usize, operand: Operand, ty: TraceType) -> u32 {
        let number = self.number_type();
        let root = match self.origin(operand) {
            Operand::Op(index) => matches!(
                self.ops[index as usize].kind,
                RecKind::Constant(TraceScalar::String(lane)) if lane >= ROOT_STRING_LANE_BASE
            ),
            Operand::Carried(_) => false,
        };
        if ty == TraceType::String && !root {
            return self.emit_host_call(pc, HostFunction::StringLength, &[(operand, ty)], number);
        }
        let helper = if ty == TraceType::StringRef {
            HelperId::STRING_LEN_REF
        } else {
            HelperId::STRING_LEN
        };
        let length = self.emit_helper_call(pc, helper, &[(operand, ty)]);
        if number == TraceType::Float {
            self.emit(pc, RecKind::IntToFloat(Operand::Op(length)), Some(number))
        } else {
            length
        }
    }

    /// A call of a native function the VM answers for a trace (see
    /// [`HostFunction`]), or `None` when the call has no such form and the
    /// other native forms may take it.
    #[allow(clippy::too_many_lines)]
    fn record_host_native(
        &mut self,
        at: &HookPoint<'_>,
        (kind, function): (DirectNativeKind, NativeFunctionHandle),
        callee: Operand,
        (arguments, first_argument): (&[(Operand, TraceType)], usize),
        (base, results): (usize, usize),
    ) -> Result<Option<Flow>, TraceAbortReason> {
        let pc = at.pc;
        let text = |ty: TraceType| matches!(ty, TraceType::String | TraceType::StringRef);
        let numeric = |ty: TraceType| matches!(ty, TraceType::Integer | TraceType::Float);
        let number = self.number_type();
        if Self::is_pattern_kind(kind)
            && let Some(flow) = self.record_pattern_native(
                at,
                kind,
                (callee, function),
                (arguments, first_argument),
                (base, results),
            )?
        {
            return Ok(Some(flow));
        }
        if Self::is_text_kind(kind) {
            return self
                .record_text_native(
                    at,
                    (kind, function),
                    callee,
                    (arguments, first_argument),
                    (base, results),
                )
                .map(Some);
        }
        let (plan, result) = match (kind, arguments) {
            // `tonumber` of an integer is the integer.
            (DirectNativeKind::Tonumber, [(argument, TraceType::Integer)]) => {
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                let value = self.emit(pc, RecKind::Move(*argument), Some(TraceType::Integer));
                return Ok(Some(self.native_result(pc, value, (base, results))));
            }
            // More than two integers fold the compare from the left, as
            // the native's loop does.
            (DirectNativeKind::MathMax | DirectNativeKind::MathMin, [first, rest @ ..])
                if rest.len() > 1 && arguments.iter().all(|(_, ty)| *ty == TraceType::Integer) =>
            {
                let host = if kind == DirectNativeKind::MathMax {
                    HostFunction::MathMax
                } else {
                    HostFunction::MathMin
                };
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                let mut value = first.0;
                for next in rest {
                    value = Operand::Op(self.emit_host_call(
                        pc,
                        host,
                        &[(value, TraceType::Integer), *next],
                        TraceType::Integer,
                    ));
                }
                let Operand::Op(value) = value else {
                    unreachable!("a fold of two or more arguments makes a value")
                };
                return Ok(Some(self.native_result(pc, value, (base, results))));
            }
            (
                DirectNativeKind::MathMax | DirectNativeKind::MathMin,
                [(_, TraceType::Integer), (_, TraceType::Integer)],
            ) => {
                let host = if kind == DirectNativeKind::MathMax {
                    HostFunction::MathMax
                } else {
                    HostFunction::MathMin
                };
                (Plan::Host(host), TraceType::Integer)
            }
            (DirectNativeKind::StringChar, args)
                if !args.is_empty() && args.len() <= MAX_HOST_ARGS =>
            {
                if !args.iter().all(|(_, ty)| numeric(*ty)) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                (Plan::Helper(HelperId::STRING_CHAR), TraceType::String)
            }
            (DirectNativeKind::StringLen, [(operand, ty)]) if text(*ty) => {
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                let value = self.emit_string_length(pc, *operand, *ty);
                return Ok(Some(self.native_result(pc, value, (base, results))));
            }
            (DirectNativeKind::TableConcat, [(_, TraceType::Table), rest @ ..])
                if rest.len() <= 3 =>
            {
                let shaped = rest.iter().enumerate().all(|(position, (_, ty))| {
                    if position == 0 {
                        text(*ty)
                    } else {
                        numeric(*ty)
                    }
                });
                if !shaped {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                (Plan::Helper(HelperId::TABLE_CONCAT), TraceType::String)
            }
            (
                DirectNativeKind::Format | DirectNativeKind::StringFormatLuaJit,
                [(_, format), rest @ ..],
            ) if text(*format) && rest.len() < MAX_HOST_ARGS => {
                if !rest.iter().all(|(_, ty)| numeric(*ty) || text(*ty)) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let helper = if kind == DirectNativeKind::Format {
                    HelperId::STRING_FORMAT_NATIVE
                } else {
                    HelperId::STRING_FORMAT_LUAJIT
                };
                (Plan::Helper(helper), TraceType::String)
            }
            (DirectNativeKind::StringSlice, args)
                if (2..=3).contains(&args.len())
                    && text(args[0].1)
                    && args[1..].iter().all(|(_, ty)| numeric(*ty))
                    && !self.is_one_byte_slice(args, results) =>
            {
                if let Some(flow) = self.record_slice_sub(
                    at,
                    (callee, function),
                    (arguments, first_argument),
                    (base, results),
                )? {
                    return Ok(Some(flow));
                }
                (Plan::Helper(HelperId::STRING_SUB), TraceType::String)
            }
            (
                DirectNativeKind::StringUpper
                | DirectNativeKind::StringLower
                | DirectNativeKind::StringReverse,
                [(_, ty)],
            ) if text(*ty) => {
                let helper = match kind {
                    DirectNativeKind::StringUpper => HelperId::STRING_UPPER,
                    DirectNativeKind::StringLower => HelperId::STRING_LOWER,
                    _ => HelperId::STRING_REVERSE,
                };
                (Plan::Helper(helper), TraceType::String)
            }
            (DirectNativeKind::StringRep, args)
                if (2..=3).contains(&args.len())
                    && text(args[0].1)
                    && numeric(args[1].1)
                    && args
                        .get(2)
                        .is_none_or(|(_, ty)| text(*ty) || *ty == TraceType::Nil) =>
            {
                (Plan::Helper(HelperId::STRING_REP), TraceType::String)
            }
            (DirectNativeKind::StringByte, args)
                if (1..=3).contains(&args.len())
                    && text(args[0].1)
                    && args[1..].iter().all(|(_, ty)| numeric(*ty))
                    && !(args[0].1 == TraceType::String
                        && self.bound_string(args[0].0).is_ok()) =>
            {
                // The substring a noted `string.sub` made: the bytes come
                // from its source.
                if let Some(flow) =
                    self.record_slice_byte(at, (callee, function), arguments, (base, results))?
                {
                    return Ok(Some(flow));
                }
                // A string the trace made, or one held in a table: the bytes
                // come from a helper, as many as the recorded call gave.
                let live = at
                    .registers
                    .get(first_argument..first_argument + args.len())
                    .ok_or(TraceAbortReason::Divergence)?;
                let count = predict_byte(live).ok_or(TraceAbortReason::UnsupportedOperand)?;
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                let wanted = self.emit(
                    pc,
                    RecKind::Constant(TraceScalar::Integer(
                        i64::try_from(count).map_err(|_| TraceAbortReason::Divergence)?,
                    )),
                    Some(TraceType::Integer),
                );
                let mut given = vec![(Operand::Op(wanted), TraceType::Integer)];
                given.extend_from_slice(arguments);
                let call = self.emit_helper_call(pc, HelperId::STRING_BYTE, &given);
                let mut bytes = Vec::with_capacity(count);
                for index in 0..count {
                    let byte = if index == 0 {
                        call
                    } else {
                        self.emit(
                            pc,
                            RecKind::HelperResult {
                                call,
                                index: u8::try_from(index)
                                    .map_err(|_| TraceAbortReason::Divergence)?,
                            },
                            Some(TraceType::Integer),
                        )
                    };
                    bytes.push(if self.integers {
                        byte
                    } else {
                        self.emit(pc, RecKind::IntToFloat(Operand::Op(byte)), Some(number))
                    });
                }
                return Ok(Some(self.native_results(pc, &bytes, (base, results))));
            }
            (DirectNativeKind::Tonumber, args)
                if (1..=2).contains(&args.len())
                    && (text(args[0].1) || args.len() == 2)
                    && args
                        .iter()
                        .all(|(_, ty)| text(*ty) || numeric(*ty) || *ty == TraceType::Nil) =>
            {
                // The type of the answer is part of the trace: the profile's
                // own conversion says it, and the helper leaves when a later
                // iteration's string converts to another one.
                let runtime = at
                    .surroundings
                    .ok_or(TraceAbortReason::UnsupportedOperand)?
                    .runtime;
                let live = at
                    .registers
                    .get(first_argument..first_argument + args.len())
                    .ok_or(TraceAbortReason::Divergence)?;
                let helper = match runtime.probe_native(function, live) {
                    Some(Value::Integer(_)) => HelperId::TONUMBER_INTEGER,
                    Some(Value::Number(_)) => HelperId::TONUMBER_FLOAT,
                    Some(Value::Nil) => HelperId::TONUMBER_NIL,
                    _ => return Err(TraceAbortReason::UnsupportedOperand),
                };
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                let call = self.emit_helper_call(pc, helper, arguments);
                let value = if helper == HelperId::TONUMBER_NIL {
                    self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Nil),
                        Some(TraceType::Nil),
                    )
                } else {
                    call
                };
                return Ok(Some(self.native_result(pc, value, (base, results))));
            }
            (DirectNativeKind::MathModf, [(operand, ty)]) if numeric(*ty) => {
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                if *ty == TraceType::Integer {
                    // An integer is its own integral part.
                    let whole = self.emit(pc, RecKind::Move(*operand), Some(TraceType::Integer));
                    let zero = self.emit(
                        pc,
                        RecKind::Constant(TraceScalar::Float(0.0)),
                        Some(TraceType::Float),
                    );
                    return Ok(Some(self.native_results(
                        pc,
                        &[whole, zero],
                        (base, results),
                    )));
                }
                let runtime = at
                    .surroundings
                    .ok_or(TraceAbortReason::UnsupportedOperand)?
                    .runtime;
                let Some(Value::Number(live)) = at.registers.get(first_argument) else {
                    return Err(TraceAbortReason::Divergence);
                };
                let integral = live.is_finite()
                    && matches!(
                        runtime.integral_float_value(live.trunc()),
                        Value::Integer(_)
                    );
                let helper = if integral {
                    HelperId::MATH_MODF_INTEGER
                } else {
                    HelperId::MATH_MODF_FLOAT
                };
                let call = self.emit_helper_call(pc, helper, arguments);
                let fraction = self.emit(
                    pc,
                    RecKind::HelperResult { call, index: 1 },
                    Some(TraceType::Float),
                );
                return Ok(Some(self.native_results(
                    pc,
                    &[call, fraction],
                    (base, results),
                )));
            }
            (DirectNativeKind::MathRandom, args)
                if args.len() <= 2 && args.iter().all(|(_, ty)| numeric(*ty)) =>
            {
                let runtime = at
                    .surroundings
                    .ok_or(TraceAbortReason::UnsupportedOperand)?
                    .runtime;
                // Lua 5.3 and later draw integers for a bound, floats for
                // none; LuaJIT and the profiles before always draw floats.
                let integer =
                    !runtime.random_is_luajit() && runtime.tagged_integers() && !args.is_empty();
                if integer && args.iter().any(|(_, ty)| *ty != TraceType::Integer) {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let lane = self.pin(pinned_native_identity(function.index()))?;
                self.guard_callee(pc, callee, lane);
                let helper = if integer {
                    HelperId::MATH_RANDOM_INTEGER
                } else {
                    HelperId::MATH_RANDOM_FLOAT
                };
                let value = self.emit_helper_call(pc, helper, arguments);
                return Ok(Some(self.native_result(pc, value, (base, results))));
            }
            (DirectNativeKind::Find, args) => {
                // The second result, the end of the match, is not an answer
                // the VM hands back.
                if results != 1
                    || !(2..=4).contains(&args.len())
                    || !text(args[0].1)
                    || !text(args[1].1)
                    || !args[2..].iter().all(|(_, ty)| {
                        numeric(*ty) || matches!(ty, TraceType::Boolean | TraceType::Nil)
                    })
                {
                    return Err(TraceAbortReason::UnsupportedOperand);
                }
                let live = at
                    .registers
                    .get(first_argument..first_argument + args.len())
                    .ok_or(TraceAbortReason::Divergence)?;
                match predict_find(live) {
                    Some(true) => (Plan::Helper(HelperId::STRING_FIND), number),
                    Some(false) => (Plan::Helper(HelperId::STRING_FIND_ABSENT), TraceType::Nil),
                    None => return Err(TraceAbortReason::UnsupportedOperand),
                }
            }
            _ => return Ok(None),
        };
        let lane = self.pin(pinned_native_identity(function.index()))?;
        self.guard_callee(pc, callee, lane);
        let value = match plan {
            Plan::Host(host) => self.emit_host_call(pc, host, arguments, result),
            Plan::Helper(helper) => self.emit_library_helper(pc, helper, arguments, result),
        };
        Ok(Some(self.native_result(pc, value, (base, results))))
    }

    /// A call of the builtin `tostring` or `string.format` the VM answers
    /// for a trace.
    fn record_host_builtin(
        &mut self,
        at: &HookPoint<'_>,
        builtin: BuiltinFunction,
        callee: Operand,
        arguments: &[(Operand, TraceType)],
        (base, results): (usize, usize),
    ) -> Result<Flow, TraceAbortReason> {
        let pc = at.pc;
        let text = |ty: TraceType| matches!(ty, TraceType::String | TraceType::StringRef);
        let numeric = |ty: TraceType| matches!(ty, TraceType::Integer | TraceType::Float);
        if let (BuiltinFunction::ToString, &[argument @ (_, TraceType::Table)]) =
            (builtin, arguments)
        {
            if let Some(flow) =
                self.record_tostring_handler(at, callee, argument, (base, results))?
            {
                return Ok(flow);
            }
            if let Some(flow) = self.record_tostring_plain(at, callee, argument, (base, results))? {
                return Ok(flow);
            }
        }
        let (helper, identity) = match (builtin, arguments) {
            (BuiltinFunction::ToString, [(_, ty)])
                if numeric(*ty)
                    || text(*ty)
                    || matches!(ty, TraceType::Boolean | TraceType::Nil) =>
            {
                (
                    HelperId::TOSTRING,
                    pinned_builtin_identity(PINNED_BUILTIN_TOSTRING),
                )
            }
            (BuiltinFunction::StringFormat, [(_, format), rest @ ..])
                if text(*format)
                    && rest.len() < MAX_HOST_ARGS
                    && rest.iter().all(|(_, ty)| numeric(*ty) || text(*ty)) =>
            {
                (
                    HelperId::STRING_FORMAT,
                    pinned_builtin_identity(PINNED_BUILTIN_STRING_FORMAT),
                )
            }
            _ => return Err(TraceAbortReason::UnsupportedOperand),
        };
        let lane = self.pin(identity)?;
        self.guard_callee(pc, callee, lane);
        let value = self.emit_library_helper(pc, helper, arguments, TraceType::String);
        Ok(self.native_result(pc, value, (base, results)))
    }

    /// `a = b .. ... .. c` of strings and numbers: the VM concatenates.
    fn record_concat(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, first, last) = (
            usize::from(instruction.a()),
            usize::from(instruction.b()),
            usize::from(instruction.c()),
        );
        self.check_register(a)?;
        if last < first || last - first + 1 > MAX_HOST_ARGS {
            return Err(TraceAbortReason::UnsupportedOperand);
        }
        let mut operands = Vec::with_capacity(last - first + 1);
        for register in first..=last {
            let (operand, ty) = self.host_operand(pc, at.registers, register)?;
            if !matches!(
                ty,
                TraceType::String | TraceType::StringRef | TraceType::Integer | TraceType::Float
            ) {
                return Err(TraceAbortReason::UnsupportedOperand);
            }
            operands.push((operand, ty));
        }
        let value = self.emit_library_helper(pc, HelperId::CONCAT, &operands, TraceType::String);
        self.write(a, value);
        Ok(pc + 1)
    }

    /// `a = #b` of a string the trace made or holds as a constant: the VM
    /// counts its bytes. (A string the entry holds has its length bound.)
    fn record_host_length(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
    ) -> Result<usize, TraceAbortReason> {
        let pc = at.pc;
        let (a, b) = (usize::from(instruction.a()), usize::from(instruction.b()));
        self.check_register(a)?;
        let number = self.number_type();
        if let Some(text) = self.constant_text(at.registers, b) {
            // A constant's length is a constant.
            let length = i64::try_from(text.len()).map_err(|_| TraceAbortReason::Divergence)?;
            let scalar = if self.integers {
                TraceScalar::Integer(length)
            } else {
                #[allow(clippy::cast_precision_loss, reason = "a string length fits f64")]
                TraceScalar::Float(length as f64)
            };
            let value = self.emit(pc, RecKind::Constant(scalar), Some(number));
            self.write(a, value);
            return Ok(pc + 1);
        }
        let (operand, ty) = self.host_operand(pc, at.registers, b)?;
        if !matches!(ty, TraceType::String | TraceType::StringRef) {
            return Err(TraceAbortReason::Divergence);
        }
        let value = self.emit_string_length(pc, operand, ty);
        self.write(a, value);
        Ok(pc + 1)
    }

    /// `GetTabUp` whose field holds a table: `_ENV.math`. The table is
    /// another window slot, read once at entry like a global, so a trace can
    /// call `math.max`. `None` when the field holds anything else.
    fn record_tab_up_table(
        &mut self,
        at: &HookPoint<'_>,
        instruction: Instruction,
        (handle, runtime): (TableHandle, &Runtime),
    ) -> Result<Option<usize>, TraceAbortReason> {
        let pc = at.pc;
        let a = usize::from(instruction.a());
        let TableKey::Field(text) = Self::field_key(at, instruction.c())? else {
            return Ok(None);
        };
        if !matches!(
            runtime.table_get_string_ref(handle, &text),
            Some(Value::Table(_))
        ) {
            return Ok(None);
        }
        self.check_register(a)?;
        let constant = u32::from(instruction.c());
        let upvalue = u16::from(instruction.b());
        let slot = match self.upvalue_slot(upvalue) {
            ExternalSlot::Upvalue(upvalue) => ExternalSlot::UpvalueField { upvalue, constant },
            ExternalSlot::ClosureUpvalue { closure, index } => ExternalSlot::ClosureUpvalueField {
                closure,
                upvalue: index,
                constant,
            },
            _ => return Err(TraceAbortReason::Divergence),
        };
        let cell = Self::upvalue_cell(at, upvalue)?.0;
        let external = self.external(at, slot)?;
        let (source, ty) = self.read_external(at, external)?;
        if ty != TraceType::Table {
            return Err(TraceAbortReason::Divergence);
        }
        if !self.field_cells.contains(&Some(cell)) {
            self.field_cells.push(Some(cell));
        }
        let value = self.emit(pc, RecKind::Move(source), Some(ty));
        self.write(a, value);
        Ok(Some(pc + 1))
    }
}

impl From<u32> for Sym {
    fn from(index: u32) -> Self {
        Self::Op(index)
    }
}

/// The iterations a numeric for loop runs after a latch at `control` that
/// continues: the one it starts now and the ones after it.
#[allow(clippy::cast_precision_loss)]
fn remaining_trips(control: Live, limit: Live, step: Live) -> f64 {
    let number = |live: Live| match live {
        Live::Integer(value) => value as f64,
        Live::Float(value) => value,
    };
    let (control, limit, step) = (number(control), number(limit), number(step));
    if step == 0.0 {
        return f64::INFINITY;
    }
    (((limit - control) / step).floor()).max(0.0)
}

/// An upper estimate of the iterations one run of a numeric for loop has,
/// from its counters mid-run: `limit / step` for a loop that counts up from
/// a non-negative number. `None` for any other loop, whose start the counters
/// do not bound.
#[allow(clippy::cast_precision_loss)]
fn run_trips((control, limit, step): (Live, Live, Live)) -> Option<f64> {
    let number = |live: Live| match live {
        Live::Integer(value) => value as f64,
        Live::Float(value) => value,
    };
    let (control, limit, step) = (number(control), number(limit), number(step));
    (step > 0.0 && control >= 0.0 && limit >= 0.0 && limit.is_finite())
        .then(|| (limit / step).floor() + 1.0)
}

/// The interpreter's own arithmetic for a for loop latch, to learn the
/// direction it takes: whether the step is positive and whether the loop
/// continues. An integer advance that overflows ends the loop without a
/// trace to close.
fn for_direction(control: Live, limit: Live, step: Live) -> Result<(bool, bool), TraceAbortReason> {
    Ok(match (control, limit, step) {
        (Live::Float(control), Live::Float(limit), Live::Float(step)) => {
            let advanced = control + step;
            let positive = step > 0.0;
            (
                positive,
                if positive {
                    advanced <= limit
                } else {
                    advanced >= limit
                },
            )
        }
        (Live::Integer(control), Live::Integer(limit), Live::Integer(step)) => {
            let Some(advanced) = control.checked_add(step) else {
                return Err(TraceAbortReason::LoopNotClosed);
            };
            let positive = step > 0;
            (
                positive,
                if positive {
                    advanced <= limit
                } else {
                    advanced >= limit
                },
            )
        }
        _ => return Err(TraceAbortReason::Divergence),
    })
}

fn register_u16(register: usize) -> u16 {
    u16::try_from(register).expect("register index fits u16")
}

fn short_offset(instruction: Instruction) -> i32 {
    i32::from(i16::from_ne_bytes(instruction.d().to_ne_bytes()))
}

#[cfg(test)]
mod reloaded_tests {
    use super::Recorder;
    use crate::bytecode::{Chunk, Constant, Instruction};
    use crate::compat::CompatibilityProfile;
    use std::sync::Arc;

    /// A recorder over `code` for the loop `0..=code.len() - 1`.
    fn over(code: Vec<Instruction>) -> Recorder {
        let latch = code.len() - 1;
        let chunk = Chunk::new(8, vec![Constant::Number(0.0)], code);
        Recorder::new(
            Arc::new(chunk),
            1,
            0,
            latch,
            None,
            CompatibilityProfile::Lua51,
        )
    }

    fn filler() -> Instruction {
        Instruction::load_constant(7, 0)
    }

    #[test]
    fn a_get_upvalue_no_instruction_of_the_loop_sets_is_a_reload() {
        let recorder = over(vec![Instruction::get_upvalue(1, 0), filler(), filler()]);
        assert!(recorder.reloaded_invariant(1, 1, 0));
    }

    #[test]
    fn a_set_upvalue_of_the_cell_in_the_loop_makes_it_vary() {
        let recorder = over(vec![
            Instruction::get_upvalue(1, 0),
            filler(),
            Instruction::set_upvalue(2, 0),
        ]);
        assert!(!recorder.reloaded_invariant(1, 1, 0));
        // Another cell's store does not.
        let recorder = over(vec![
            Instruction::get_upvalue(1, 0),
            filler(),
            Instruction::set_upvalue(2, 1),
        ]);
        assert!(recorder.reloaded_invariant(1, 1, 0));
    }

    #[test]
    fn a_move_of_a_register_the_loop_never_writes_is_a_reload() {
        let recorder = over(vec![Instruction::move_(2, 1), filler(), filler()]);
        assert!(recorder.reloaded_invariant(2, 1, 0));
        // The source written in the loop and not reloaded: it varies.
        let recorder = over(vec![
            Instruction::move_(2, 1),
            Instruction::load_constant(1, 0),
            filler(),
        ]);
        assert!(!recorder.reloaded_invariant(2, 1, 0));
    }

    #[test]
    fn a_chain_of_moves_ends_in_the_reload() {
        let recorder = over(vec![
            Instruction::get_upvalue(1, 0),
            Instruction::move_(2, 1),
            Instruction::move_(3, 2),
            filler(),
        ]);
        assert!(recorder.reloaded_invariant(3, 3, 0));
        // The same chain over a cell that the loop sets.
        let recorder = over(vec![
            Instruction::get_upvalue(1, 0),
            Instruction::move_(2, 1),
            Instruction::move_(3, 2),
            Instruction::set_upvalue(5, 0),
        ]);
        assert!(!recorder.reloaded_invariant(3, 3, 0));
    }

    #[test]
    fn a_register_with_no_write_in_front_of_the_call_is_not_a_reload() {
        let recorder = over(vec![filler(), filler(), filler()]);
        assert!(!recorder.reloaded_invariant(1, 2, 0));
    }

    #[test]
    fn a_reload_on_only_one_branch_arm_is_not_a_reload() {
        // pc 0: if r6 jumps to 3; pc 1: r1 = upvalue; pc 2: (the call is at
        // 3, so the jump at 0 lands past the reload).
        let skip = Instruction::jump_if_false(6, 2);
        let recorder = over(vec![
            skip,
            Instruction::get_upvalue(1, 0),
            filler(),
            filler(),
        ]);
        assert!(!recorder.reloaded_invariant(1, 3, 0));
        // A branch that lands on the reload or before it does not matter.
        let recorder = over(vec![
            Instruction::jump_if_false(6, 0),
            Instruction::get_upvalue(1, 0),
            filler(),
            filler(),
        ]);
        assert!(recorder.reloaded_invariant(1, 3, 0));
    }
}

#[cfg(test)]
mod scout_tests {
    use super::{Scout, WakeCandidate};

    /// The loop of the root closes at pc 69 and begins at pc 20. The queued
    /// side's guards: pc 69 failing to 70 (the next trips of the root's loop
    /// are none) leaves the nest, and pc 30 failing to 31 does not.
    fn scout(header: usize) -> Scout {
        Scout {
            latch: 69,
            header,
            candidates: vec![
                WakeCandidate {
                    guard_pc: 69,
                    successor: 70,
                    origin: 4,
                    beyond: true,
                    root: false,
                },
                WakeCandidate {
                    guard_pc: 30,
                    successor: 31,
                    origin: 2,
                    beyond: false,
                    root: false,
                },
            ],
            following: header == usize::MAX,
            previous: None,
            budget: 100,
        }
    }

    fn watch(scout: &mut Scout, path: &[usize]) -> Vec<Option<u32>> {
        path.iter()
            .map(|pc| scout.observe(*pc).map(|candidate| candidate.origin))
            .collect()
    }

    #[test]
    fn the_scout_follows_nothing_until_the_roots_loop_exits() {
        // The failure edge of a guard of the side means nothing while the
        // interpreter is still in the root's loop.
        let mut scout = scout(20);
        assert_eq!(
            watch(&mut scout, &[20, 21, 30, 31, 69, 20, 21, 69]),
            [None; 8]
        );
        assert!(!scout.following);
    }

    #[test]
    fn the_scout_wakes_at_the_guard_that_leaves_the_nest_after_the_side_path() {
        // The root's loop ends (69 to 70), the side's path runs to its own
        // last trip (the loop test at 69 again) and fails there to 70.
        let mut scout = scout(20);
        let woken = watch(&mut scout, &[20, 69, 70, 9, 10, 19, 69, 70]);
        assert_eq!(woken[..7], [None; 7]);
        assert_eq!(woken[7], Some(4), "the guard of pc 69 that leaves the nest");
    }

    #[test]
    fn a_side_path_that_links_back_to_the_header_ends_the_following() {
        let mut scout = scout(20);
        // 69 to 70 starts the path, which comes round to the header: the
        // next exit of the loop starts it again, and only a failure after it
        // wakes the scout.
        let woken = watch(&mut scout, &[20, 69, 70, 9, 20, 69, 70, 9, 69, 70]);
        assert_eq!(woken[..9], [None; 9]);
        assert_eq!(woken[9], Some(4));
    }

    #[test]
    fn a_guard_that_stays_in_the_nest_takes_the_scout_off_the_path() {
        let mut scout = scout(20);
        // The path of the side fails the guard at pc 30 (to 31), which does
        // not leave the nest: no wake, and the scout waits for the next exit.
        assert_eq!(watch(&mut scout, &[69, 70, 30, 31, 69, 70]), [None; 6]);
        assert!(scout.following, "the exit at 69 to 70 began another path");
    }

    #[test]
    fn a_scout_for_the_roots_own_exit_follows_from_the_start() {
        let mut scout = scout(usize::MAX);
        scout.candidates = vec![WakeCandidate {
            guard_pc: 69,
            successor: 70,
            origin: 1,
            beyond: true,
            root: true,
        }];
        // The header is visited on every iteration of the root's loop and
        // ends nothing here.
        let woken = watch(&mut scout, &[20, 21, 69, 20, 21, 69, 70]);
        assert_eq!(woken[..6], [None; 6]);
        assert_eq!(woken[6], Some(1));
    }
}

#[cfg(test)]
mod unroll_while_tests {
    use super::{MAX_UNROLLED_WHILE_BACKEDGES, Recorder};
    use crate::bytecode::{Chunk, Constant, Instruction};
    use crate::compat::CompatibilityProfile;
    use std::sync::Arc;

    fn recorder() -> Recorder {
        let code = vec![Instruction::load_constant(7, 0); 8];
        let chunk = Chunk::new(8, vec![Constant::Number(0.0)], code);
        Recorder::new(Arc::new(chunk), 1, 0, 7, None, CompatibilityProfile::Lua51)
    }

    #[test]
    fn a_recording_that_does_not_unroll_follows_no_while_jump() {
        assert!(!recorder().may_unroll_while(5, 2));
    }

    #[test]
    fn a_while_loop_in_the_root_frame_is_not_unrolled_until_its_root_retired() {
        let unrolling = recorder().with_loop_unrolling(true);
        assert!(!unrolling.may_unroll_while(5, 2));
        let short = unrolling.with_short_loops(vec![2]);
        assert!(short.may_unroll_while(5, 2));
        // Another loop of the frame is still left to its own root.
        assert!(!short.may_unroll_while(6, 3));
    }

    #[test]
    fn a_forward_jump_is_no_while_loop() {
        let short = recorder()
            .with_loop_unrolling(true)
            .with_short_loops(vec![6]);
        assert!(!short.may_unroll_while(2, 6));
    }

    #[test]
    fn the_trips_of_one_recording_are_limited() {
        let mut short = recorder()
            .with_loop_unrolling(true)
            .with_short_loops(vec![2]);
        short.unrolled_whiles = MAX_UNROLLED_WHILE_BACKEDGES - 1;
        assert!(short.may_unroll_while(5, 2));
        short.unrolled_whiles = MAX_UNROLLED_WHILE_BACKEDGES;
        assert!(!short.may_unroll_while(5, 2));
    }

    #[test]
    fn a_numeric_loop_given_up_as_short_unrolls_whatever_its_range() {
        // `for dy = ytop, ybot, -1 do ... break end` with 80 trips in its
        // range and a break on the first: its own root was given up for
        // `LoopNotClosed`, and the loop around follows its latch.
        let unrolling = recorder().with_loop_unrolling(true);
        assert!(!unrolling.may_unroll_short(5, 2));
        let mut short = unrolling.with_short_loops(vec![2]);
        assert!(short.may_unroll_short(5, 2));
        // Another loop of the frame, and a forward branch, are not.
        assert!(!short.may_unroll_short(6, 3));
        assert!(!short.may_unroll_short(2, 6));
        // The trips share the budget of the `while` loops.
        short.unrolled_whiles = MAX_UNROLLED_WHILE_BACKEDGES;
        assert!(!short.may_unroll_short(5, 2));
    }

    #[test]
    fn a_recording_that_does_not_unroll_follows_no_short_numeric_loop() {
        let short = recorder().with_short_loops(vec![2]);
        assert!(!short.may_unroll_short(5, 2));
    }
}
