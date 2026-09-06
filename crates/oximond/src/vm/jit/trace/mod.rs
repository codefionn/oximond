//! Trace engine ownership.
//!
//! [`TracePolicy`] decides, per loop header, whether a compiled trace or the
//! interpreter runs hot code. Opcode handlers and the native entry path call
//! its small interface.
//!
//! # Lifecycle
//!
//! The interpreter consults [`TracePolicy::at_loop_header`] on every backward
//! step in tracing mode, and after every Lua tail call that replaced the
//! running frame, for the callee's pc 0 (a chain of tail calls that comes
//! back to a function is a loop, see [`recorder`], "Loops through tail
//! calls"), and after every Lua call that pushed a frame of the calling
//! frame's own chunk, for that function's entry (see "Function traces"
//! below). The policy counts backedges per header (see
//! [`policy`]). A hot header starts a [`recorder::Recorder`]. While one
//! records, the interpreter calls [`TracePolicy::record`] before each guest
//! instruction, from the fast loop and from the general loop. A loop that
//! closes becomes a verified [`TraceIr`]. The policy queues it as a compile
//! request (see [`TracePolicy::take_outbox`]). The VM hands requests to the
//! shared [`oximond::jit::JitBroker`], and finished code comes back at a VM
//! safepoint ([`TracePolicy::complete_compile`]) and installs in the
//! [`cache`] at its header (see [`native`]). The next consult of that header
//! returns [`TraceDispatch::Enter`], and the interpreter runs the native code
//! with [`TracePolicy::enter`]. Any abort applies the site's retry or
//! rejection policy.
//!
//! A compile that panics (a codegen bug: release builds run no Cranelift
//! verifier, so malformed code can reach its register allocator) does not
//! take the compile thread down with it. The broker catches the panic around
//! that one compile and returns [`TraceCompileError::Panicked`], which settles
//! like any other compile error: the header is rejected as
//! [`TraceAbortReason::CompileRejected`], a side or retree fails as one that
//! did not compile, and `compile_panics` counts it. A compile builds all its
//! state itself, so the worker goes on with the next request.
//! `OXIMOND_JIT_VERIFY` turns Cranelift's verifier on in release builds, so
//! such a bug shows as a rejection with the verifier's message instead.
//!
//! # Ownership
//!
//! The trace engine is the only native engine for loop headers and for the
//! entries of directly self-recursive functions. Other function entries
//! interpret, and so does every header the tracer rejects. Calls the
//! interpreter hands to native code on its own (call plans, recursive
//! numeric calls) never reach the policy.
//!
//! - An installed trace owns its header until it proves unproductive (fewer
//!   than a few loop-backs per entry, see [`native`]). Then it is retired
//!   and the site backs off with growing cooldowns (hysteresis), and is
//!   rejected for good after the retry budget.
//! - A branch that alternates is not unproductive: its guard gets a side
//!   trace (see "Side traces" below), and the productivity check forgives the
//!   trace while one is on its way. Only a branch whose other arm no trace can
//!   hold (an unsupported opcode, an inner loop, a call of a native function)
//!   keeps failing its guard for good, and that is what retires a trace.
//!
//! # Profiles
//!
//! Every compatibility profile creates sites and records. Lua 5.1, 5.2 and
//! `LuaJIT` 2.1 have floats only, so their traces are float traces. Lua 5.3 to
//! 5.5 have integers and floats as distinct subtypes: entry guards, loop
//! parameters and exit areas keep them apart, integer arithmetic wraps, an
//! integer division or modulo guards its divisor first, an integer
//! comparison with a float is exact, and an integer numeric for loop stops
//! exactly where the interpreter's `checked_add` does. See [`recorder`] for
//! the per-operation rules. A string is an opaque value, and its bytes are
//! readable natively (see "Strings" below).
//!
//! # Tables
//!
//! A trace reads and overwrites table slots natively, under a contract the
//! entry establishes (see [`native`], "bindings"). A table is an opaque value
//! only known by identity, so an access needs a table that is the same one in
//! every iteration: an entry input, reached through moves, whose register the
//! loop never writes. For each such table the entry looks up its storage once
//! and hands the code the addresses:
//!
//! - `t[k]` with a number `k` that names a slot of the array part. The code
//!   leaves *before* the instruction unless `k` is an integral number inside
//!   the array and the slot holds a non-nil value of the type recorded. The
//!   interpreter then does the access itself: a nil slot, a hash-part or
//!   fractional key, `__index`, another type, an error. A read recorded at a
//!   nil hole of a table with no metatable expects nil instead, and the entry
//!   refuses a table that has gained a metatable
//!   ([`oximond::jit_ir::trace::BindingKind::NoMetatable`]): only then is nil
//!   the value of the access.
//! - `t[k] = v` for a number or boolean `v` into a slot that holds a value
//!   of the same type. That store overwrites the payload only. It keeps the
//!   slot's tag, the table's border and every GC edge, so neither
//!   `__newindex` nor the table's bookkeeping could have seen it. The store
//!   is one committed effect: a guard that fails after it resumes after it.
//! - `t[k] = v` for a number `k` and a number, boolean or nil `v` that the
//!   recording saw add a key (a nil hole, the slot past the end, a hash key,
//!   a value of another type): a raw set
//!   ([`oximond::jit_ir::trace::TraceOp::ArraySet`]). A slot of the stored
//!   type is still overwritten in place. Any other store calls the VM, which
//!   stores as the interpreter does into a table without `__newindex` for
//!   the key, growing the array or the hash part, and hands the code the
//!   array words of the table again. The guard leaves before the
//!   instruction when the VM would not store raw: an absent key of a table
//!   whose metatable has `__newindex`, a NaN key, a read-only table. Since
//!   such a store may move storage, a tree that holds one reads array words
//!   again after each one and binds no field or lane storage (a length
//!   binding is fine: the VM rewrites the border's word with the array
//!   words). A table that is a dense list takes the stores at its two ends
//!   without the VM: the code binds its header
//!   ([`oximond::jit_ir::trace::BindingKind::Header`]) and writes the slot
//!   after the last, or takes the last one off, in the array part itself,
//!   updating the length and the border. A pop leaves its key as a deleted
//!   key the table has not recorded yet; the table records it when anything
//!   else asks (see `oximond::runtime::NativeTableLayout`). Storing a string
//!   or a table calls the VM's store entry, which owns the reference count
//!   and the write barrier. The stores at the ends of a list the recording
//!   saw take the store in place leave through their guard, not through the
//!   VM, so they bind no storage a rehash would move.
//! - `t.name` and `t.name = v` for a constant string key, with the slot
//!   resolved at entry (the key is looked up by content, so a key string
//!   allocated elsewhere matches) and its type checked there. The type of a
//!   slot cannot change during a run, so these cannot fail. A read may find
//!   the field behind table-valued `__index` hops (a class table, a defaults
//!   object): the entry follows exactly the hops the recording saw and
//!   refuses a table that has gained the key or changed its chain. A store
//!   always names the table's own slot.
//! - `t.name` holding a function (a method, `math.sin`): the entry checks that
//!   the field holds the very function the recording pinned, and the trace
//!   reads it as that constant.
//! - `#t` for a table without a `__len`, read once at entry: nothing a
//!   trace stores changes a border.
//! - `t[k]` with a string `k` known only at run time (a borrow out of an
//!   array, or a string an entry input holds) and `t` a bound table that has
//!   no metatable at all
//!   ([`oximond::jit_ir::trace::TraceOp::HashLoad`]). One call into the VM
//!   looks the string up by content in whatever representation the table has
//!   (a small ordered scan, a shaped record, a hashed index). The call
//!   reads only: it allocates nothing, collects nothing, runs no Lua and never
//!   consults `__index`, so it ends no borrow. A missing key is nil. The
//!   result is a number, a boolean or nil, in the type the recording saw
//!   (profile number kinds included); another type leaves *before* the
//!   instruction, and the entry refuses a table that has gained a metatable.
//!   A side trace gets the borrow its parent holds as an entry input of its
//!   own (the slot's address, not a lane), and can use it until it ends
//!   borrows itself.
//! - `t[k] = v` with a string `k` known only at run time, and `t.name = v`
//!   for a name the table lacks (or a value of a type the in-place store
//!   does not take), are one call into the VM that stores as the interpreter
//!   does ([`oximond::jit_ir::trace::HelperId::TABLE_SET_STRING`]): it walks
//!   `__newindex` through tables, stores in the table the walk ends at, and
//!   leaves before the instruction if the walk ends at a function. It moves
//!   hash storage only, so the array parts and the strings borrowed from them
//!   stay valid, and a tree that holds one binds no field or lane storage.
//! - `t[k]`, `t[k] = v` and `#t` that run a Lua closure as `__index`,
//!   `__newindex` or `__len` are an inlined call of the closure behind a
//!   check of which handler runs (see the recorder's `inline_meta`).
//!
//! A metatable matters for none of these: Lua consults `__index` and
//! `__newindex` only for an absent key, and a present non-nil slot is what
//! every one of these accesses insists on (a raw set and a nil read check
//! the metatable themselves). Nothing in a run can collect garbage or run
//! Lua code, and only a raw set can resize a table, after which the code
//! reads the array words again, so the addresses it uses stay valid. An
//! entry whose table lost its storage (the field is gone or has another
//! type, a `__len` appeared) is an entry mismatch, and so is an array whose
//! first slot has another type than the recording saw. Each array access
//! still guards its own slot, but a loop is usually run over arrays of one
//! element type, so the first slot picks the version (integers, then floats).
//!
//! A recording that meets a table read it cannot repeat on the iteration it
//! happened to start on (`t[0]` in a loop from 0, the slot past the end, a
//! hole of a table with a metatable) aborts with
//! [`TraceAbortReason::TableMiss`]. It is retried after an odd wait, because
//! another iteration usually records, but three aborts in a row at the same
//! instruction for the same reason reject the header
//! ([`policy::MAX_SAME_ABORTS`]).
//!
//! A table the loop does not hold the same in every iteration (one read out of
//! an array, or, when a tree could run the loops around this one, one held in
//! a register those loops write) has no storage to bind at entry for its
//! fields. Two operations make it usable:
//!
//! - `t[k]` on a bound array whose slot holds a table gives the table a
//!   *lane*, an index in the run's root bank (see "Tables and closures"
//!   below). The load leaves before the instruction like any array read when
//!   the slot holds something else, and also when the run has no lane left
//!   ([`oximond::jit::MAX_TRACE_TABLE_LANES`] tables).
//! - `t.name` and `t.name = v` on a table known only by lane. The VM keeps a
//!   *row* per lane with the address of each field slot the trace names, found
//!   when the lane is given (or at entry, for a table in an entry input) and
//!   valid for the run for the same reason a bound slot is. Native code
//!   indexes the row by lane. A zero slot (the key is absent, the value has
//!   another type, the table is read-only) leaves before the instruction, so
//!   the interpreter runs `__index`, `__newindex` or the error. Nothing about
//!   the table is assumed from another access: a field a bound table
//!   resolved at entry and a field of the same table met by lane are the same
//!   slot, because the heap operations of a trace never reorder.
//!
//! A table met by lane has no address bound at entry, so the rest of its
//! accesses are one call each into the VM, which looks the key up on every
//! call (see [`oximond::jit_ir::trace::HelperId::TABLE_GET_INTEGER`] and its
//! siblings by result type, and `TABLE_LENGTH`):
//!
//! - `t[k]` with a number or string `k`, and `t.name` for a field that holds
//!   a string, a function, or is found behind `__index` hops (the class
//!   pattern: `obj:method()` on an object whose metatable's `__index` is the
//!   class table). The call reads as `rawget` does and, with its walk
//!   argument, follows `__index` tables up to the interpreter's hop limit. It
//!   leaves before the instruction unless the key is present with a non-nil
//!   value of the type recorded, so a nil, a function `__index`, a type that
//!   changed and an error are the interpreter's. A table, function or string
//!   result is a lane: a table or Lua closure joins the run's bank, a string
//!   is a new string lane that holds the very string. The call reads only:
//!   it allocates nothing the collector sees, collects nothing and runs no
//!   Lua. A lane table is looked into on every call, so a result is never
//!   assumed from an earlier access.
//! - `#t` is one such call too. It leaves for a `__len` in the profiles
//!   that honour it.
//! - A table in a field of any table (bound or by lane) is read from the
//!   lane's row (`TableFieldLoad` of type `Table`) and gets a lane, so
//!   `region.solids` and `recs[i].inner` work in loops, not only in function
//!   traces.
//! - A table a lookup call gives (`TABLE_GET_TABLE`) has its row filled
//!   right after the call in a loop tree with row fields, so the fields read
//!   after it (`starts[hash].starts`, `cache[shape].n`) are read through a
//!   row that is this run's. Rows are not cleared between runs: before this,
//!   the row of such a lane was only filled in a tree with lane array views,
//!   so a field read found either a zero row and left at every entry (one
//!   iteration per entry) or the words another run left for another table.
//!   Both backends make the fill (`call_helper_call` in `emit.rs`, and
//!   `fills_table` in `mir/lower/calls.rs`), so these trees compile under
//!   MIR too. The rule is the lookup's alone: a table a constructor or a
//!   cell read gives keeps the row it had. Filling the row of every table-giving helper broke the trees that
//!   store a fresh table into a cell (every entry left at once).
//! - The rule for which helper results get their row filled after the call
//!   covers every result, not only the first. The value of a `next` step is
//!   the second result, and in a tree with lane array views
//!   (`for _, box in pairs (fixed) do ... box[5]`) the loop reads the value's
//!   array through its row. Before, only the first result (the key) was
//!   looked at, so the row of a value table that was not an entry input held
//!   what an earlier run of any trace left in that lane. The loop then read
//!   another table's array, and faulted once that table was freed: the
//!   intermittent SIGSEGV at Mineclonia startup in
//!   `gwp_compute_fixed_ground_height` (pathfinding.lua:1151). Each startup
//!   read 17 to 41 such rows, most of them holding another table's address,
//!   and a crash needed that table to be freed. MIR refuses `LaneArrayLoad`,
//!   so that tree was Cranelift's under either backend setting. MIR follows
//!   the same rule.
//! - A userdata (`TABLE_GET_USERDATA`, type `Userdata`) read out of a field
//!   or an array slot, of a bound table too, is a lane that can be moved,
//!   tested for truthiness and restored into a register, nothing else. It is
//!   never an entry input, so a loop that carries one across iterations in a
//!   register aborts.
//! - A string read out of an array slot of a bound table is a borrow of the
//!   slot (`ArrayLoad` of type `StringRef`), as before. A string compared
//!   with `==` is still refused: a trace never compares strings.
//! - A table register the loop itself replaces (`if t == a then t = b else
//!   t = a end`) is another table at every iteration, which entry cannot bind.
//!   A recording that closes with such a table is refused once (`TableMiss`)
//!   and marks the site rowless; the retry reaches the table through its lane
//!   in the register, by the lookup calls. A loop whose swap sits in a rare
//!   arm the recording did not take keeps its entry bindings, and the arm
//!   leaves through a guard.
//! - `t[k] = v` with a number `k` into a table met by lane, other than an
//!   in-place overwrite of an array slot (see the lane array store below),
//!   is one call (`TABLE_SET_INDEX`): the walk of `__newindex` through tables, as for
//!   `TABLE_SET_STRING`, ends in a table that holds the key or lacks a
//!   handler, and the VM's raw store (`Runtime::table_set`, with its write
//!   barrier and accounting) puts the value there. A function `__newindex`,
//!   a read-only table and a NaN key miss, having changed nothing, so the
//!   interpreter makes the store. The store may grow the array part or move
//!   the hash part of any table, so it is a helper that *moves storage*: the
//!   words of array and length bindings are read again after it, borrowed
//!   strings end, and the rule below applies to the field bindings and rows
//!   beside it.
//!
//! The lookup calls bind no storage, so they sit beside any store. The row
//! reads (`TableFieldLoad`/`TableFieldStore`) and field bindings
//! address storage a hash-moving store (`TABLE_SET_STRING`, `SET_STRING`,
//! `TABLE_SET_INDEX`) can move, and a trace holds both only in a *rowless*
//! recording. When the verifier refuses a recording for that pair, the loop's
//! site is marked rowless and the next recording reads every field through
//! the lookup calls above and stores every field through the store helpers.
//! Storing a string, a table or a function (a Lua closure, a native or a
//! builtin; a closure lane of the bank, passed as a `Value` helper argument)
//! into a field or a number slot is then the VM's store entry (reference
//! count and write barrier included). No store of a lane table runs in a window where the collector can run: the helpers
//! allocate nothing and collect nothing, and the lane bank roots every value
//! the run took out of a table until the run ends.
//!
//! A table met by lane has no binding, but its array part is still read
//! inline: the row of the lane holds the table's address
//! ([`oximond::jit_ir::trace::TABLE_ADDRESS_KEY`], written fresh by the VM
//! each time it fills or rebases the row, zero for a read-only table), and a
//! [`oximond::jit_ir::trace::TraceOp::LaneArrayLoad`] reads the array pointer
//! and length from the table at the offsets
//! [`oximond::runtime::NativeTableLayout`] names, then makes the bound-array
//! checks: it leaves before the instruction unless the key is inside the
//! array and the slot holds a present value of the recorded type. The words
//! are cached per lane within a straight-line stretch and dropped wherever
//! array binding words are read again (after a raw set, a helper that moves
//! storage or rewrites arrays, and at loop and side boundaries), so an array
//! that `table.insert` moved is read from where it went. A table address
//! stays put for the run, and the lane bank roots the table.
//!
//! `t[k] = v` into such a table is a
//! [`oximond::jit_ir::trace::TraceOp::LaneArrayStore`] instead of the
//! `TABLE_SET_INDEX` call when the recording finds `k` inside the array part
//! and slot `k` holding a number or a boolean of `v`'s type in a writable
//! table. Native code finds the slot as the read does and leaves before the
//! instruction unless the slot still holds a non-nil value of that type
//! (the row's address word is zero for a read-only table, which leaves
//! too). The store then overwrites the payload only: the tag, the border,
//! the hash part and every GC edge stay (a number or a boolean needs no
//! write barrier), and `__newindex` could not have run because the key is
//! present. It moves no storage, so it sits beside row reads and field
//! bindings, and the slot's fact answers a later read of the same table and
//! key in the recording and in the optimizer. Any other store (a nil slot, a
//! key past the array, another type, a string or a table value) stays the
//! call. Mineclonia's `mark_occupied` (`region.solids[idx] = bor
//! (region.solids[idx], mask)`, `mcl_util/shape.lua`) made that call beside
//! the row read of `region.solids`, so every recording of the `region_op`
//! `z` loop (`shape.lua:359`) was refused ("a table set beside a table met
//! by lane"), retried rowless, and its sides aborted with `TableMiss`. With
//! the store in place no recording is refused there; on a repro of that loop
//! (`OXIMOND_JIT_SYNC=1`) the run went from 110.6 G to 14.5 G instructions
//! (the interpreter takes 14.3 G; the run before also ended in the wrong
//! result noted under "Linking through a return").
//!
//! A tree may also lose a binding to a side: a binding resolves the table its
//! register held at the tree's entry, and a side that writes the register
//! (an outer side that calls the function again, which rewrites its
//! parameters) makes that wrong. `plan_tree` then demotes an array binding
//! whose uses are all reads of present numbers, booleans or tables to lane
//! array views in every member. A field, length, header or no-metatable
//! binding (and an array binding with stores) stays bound when only sides
//! write the register, each ends in a link or an exit, and none has sides of
//! its own: each linking writer then leaves through its link snapshot unless
//! the register holds the entry table again, so every path through the tree
//! sees the table the binding resolved, and the root does not carry the
//! register. A callee, string or metatable-class binding on a written
//! register, or one the root itself writes, still refuses the side
//! ([`oximond::jit_ir::trace::TreeError::TableWritten`]).
//!
//! What a trace does not do: take `#t` of a table met by lane inline (it is
//! a call), store a userdata, compare strings, and carry a userdata in a
//! register across iterations. Such a loop aborts its recording or exits at
//! the access.
//!
//! `==` on two tables or two userdata records only the case where both are
//! the same object, behind a guard that the result is true: two distinct
//! ones may be equal through `__eq`, which only the interpreter runs, so
//! that pair leaves before the compare (or aborts the recording).
//!
//! A tree shares one list of table bindings. A side trace starts from the
//! root's list, reuses the bindings it needs, and may add its own: a table
//! only the side's arm touches. Entry resolves them all, from the table in
//! the entry input of each binding's register, so every member of the tree
//! has an input for each of those registers. A side whose start finds no
//! table in such a register (a temporary of the root's iteration, which some
//! member then writes) may list the binding as long as none of its own
//! operations names it; the plan demotes it or refuses the tree as below.
//! That is only sound while no
//! member writes such a register, since the binding resolved the table the
//! register held at entry. [`oximond::jit_ir::trace::plan_tree`] refuses a
//! side that does (the tree keeps its exit), and a side recorded against an
//! older list of bindings is stale and records again.
//!
//! # Strings
//!
//! A string is known by identity, a lane like a table's. A trace moves it
//! between registers and cells, and an exit restores it. It can also read the
//! string's bytes: a string in an entry input that no iteration writes is
//! bound at entry like a table (see [`oximond::jit_ir::trace::BindingKind::Bytes`]),
//! and the VM hands the code the address and length of its bytes. `#s` is the
//! length. `string.byte (s, i)` is a byte read that leaves before the call
//! unless `i` is an integral number inside the string, so the interpreter
//! makes every other call (an empty result, a negative index, an error).
//! `string.sub (s, i, i)` followed by `string.byte` is one such read.
//!
//! Equal bytes may sit in two allocations, so no lane comparison tells two
//! strings apart. `==` and `~=` of two strings (`Equal`, `TestEqual`, and
//! `TestEqualK` with a string constant, which takes the constant as a root
//! string) are recorded as a call of
//! [`oximond::jit_ir::trace::HelperId::STRING_EQUAL`], which compares the
//! bytes and misses for a slot that holds no string. Both backends make that
//! call only when they must. A borrowed string against a short root constant
//! compares three words of the string's storage with the constant's. Any
//! other pair reads both strings (a lane through the run's bank or the
//! trace's roots) and compares their words: two inline strings are equal
//! exactly when their words are, and an inline and a heap string never are.
//! Two heap strings in two allocations, and a lane the code cannot read, go
//! to the helper (`string_equal_constant` and `string_equal_by_words` in
//! `jit/trace/emit.rs`). One value is equal to itself, and two root strings fold by their bytes at
//! recording time. A string and a value of another type are never equal and
//! `__eq` never runs for a string, so the entry types (or the read that gave
//! the value) decide that comparison, as for any two types. `<` and `<=` of
//! strings stay with the interpreter.
//!
//! A call that wants `c` results writes its result registers when it
//! returns, so the chunk's liveness (`Chunk::live_registers`) counts them
//! dead before the call. A temporary that only takes a call result (the
//! third value `pairs (t)` gives a generic `for`) is then no entry input of
//! the loop around it, and an arm that skips the call does not leave a
//! value of another type there for the next entry check to refuse.
//!
//! # Library calls
//!
//! A call of a library function the VM can answer without leaving the trace
//! is a [`oximond::jit_ir::trace::TraceOp::CallHelper`]: `tostring`,
//! `string.find`, `string.format`, `string.char`, `string.len`,
//! `table.concat`, and the operators `..` and `#` on the strings those make.
//! (`math.max` and `math.min` of integers lower to a compare, `tonumber` of
//! an integer to a move. Of more than two arguments, all integers or all
//! floats, they fold the two-argument form from the left, as the native's
//! loop does: `mathmax (0, a, b)`. A mix of integers and floats in Lua 5.3
//! and later still refuses, since the result keeps the chosen argument's
//! subtype.)
//!
//! The scalar library has helpers too. `string.sub` in general, `string.rep`,
//! `string.upper`, `string.lower` and `string.reverse` give a string lane,
//! and `string.byte` of a string the trace did not bind gives one to four
//! bytes as results. `tonumber` of a string has a helper for each class of
//! answer (integer, float, nil): the profile's own conversion tells the
//! recorder the class, and the helper leaves the trace when a later string
//! converts to another one. `math.modf` has one for an integral part that is
//! an integer and one for a float. `math.random` draws from the generator
//! the interpreter's call draws from, by the same code. It is flagged as
//! advancing private state
//! ([`oximond::jit_ir::trace::HelperEffects::PRIVATE_STATE`]): nothing hoists,
//! merges, drops or reorders a draw, but a draw writes no table, so cached
//! table reads stay valid across it. It leaves the trace before the generator
//! moves for an argument the interpreter raises an error for.
//! `math.log` and `math.pow` are float operations, not helpers (the
//! logarithm in a base is the profile's own function of the two numbers, as
//! the native computes it), and `tostring` also takes booleans and nil.
//!
//! The pattern functions have helpers too (see
//! [`recorder`]'s `inline_pattern` and [`host::pattern`]). `string.find`,
//! `string.match` and the steps of a `string.gmatch` iterator give their
//! captures as results. A pattern has captures of a fixed number and kind, so
//! the recorder runs the pattern on the live arguments and calls the helper of
//! the shape the call gave (how many results, which are strings and which
//! positions); a later call of another shape misses, and the interpreter
//! makes it. A call that finds nothing has a helper that misses when there is
//! a match. `string.gmatch` makes its iterator in a helper (a lane of the
//! bank), and a step reads and commits the iterator's cursor, flagged as
//! private state. `string.gsub` with a string or a table replacement gives the
//! new string and the count; a replacement the profile's rules reject, a
//! table with `__index` and a function stay in the interpreter.
//!
//! A method call on a string (`s:upper ()`) looks the function up through the
//! string metatable. A helper checks on every run that the metatable's
//! `__index` still holds the recorded native under the method's name, and
//! the call then records as the `string.upper (s)` form. A call that takes
//! the open results of another (`string.upper (tostring (i))`,
//! `table.insert (t, f (x))`, `tonumber (f ())`) records too, if nothing was
//! stored between the two calls: a miss leaves through the snapshot from
//! before the first call, and the interpreter makes both.
//!
//! `table.insert (t, value)` and `table.remove (t)` on a table the trace
//! binds, that is a dense list now, are the stores they stand for: an insert
//! sets `#t + 1`, and a remove reads the last slot and sets it to nil. They
//! run in place (see the stores above) and call no helper. Any other call of
//! the two is a helper that writes: for a table met by lane, a position, a
//! value that is not a number, boolean, string or table, or an element that
//! is not a number or boolean. The helpers act on a dense list (no hole, no deleted hash key, a border at the end of the
//! array part) and move the elements after the place in one step, leaving the
//! table as the interpreter's loop of raw gets and sets leaves it, down to
//! the records `next` resumes from. A remove has a helper per kind of element
//! it takes out (integer, float, boolean, string, table), and one that finds
//! nothing to take. From Lua 5.3 on the helpers also decline a table whose
//! metatable has `__index`, `__newindex` or `__len`. Since the call moves
//! array parts, the trace reads the array and length words again after it
//! ([`oximond::jit_ir::trace::HelperEffects::REWRITES_ARRAYS`]), and every
//! cached read of a table is dropped. Native code calls an `extern "C"` function of
//! [`helpers`] with the arguments in registers, and the helper runs the code
//! the interpreter runs for the call ([`host`]) and hands back one value. A
//! string it makes is a lane of the run's bank, published by the helper under
//! the run's room for strings,
//! and a loop that makes strings drops the ones the next iteration cannot
//! name ([`oximond::jit_ir::trace::TraceOp::ReleaseStrings`]). A table a
//! virtual one becomes when it escapes ([`oximond::jit_ir::trace::HelperId::TABLE_NEW`])
//! takes a lane the same way and goes at the same release. The entry first
//! reserves room in the arena of tables, since a table made with the arena
//! full would move every table native code holds the address of; a call that
//! finds no room, or no lane, is a miss. The guard of a
//! call leaves before the instruction when the VM cannot answer (an error, a
//! metamethod, a pattern, a result of another type than recorded, no lane
//! left), so the interpreter makes the call itself and raises whatever it
//! raises. The optimizer hoists a call whose arguments the loop never
//! changes into the preheader, where the call runs once per entry.
//!
//! # Side traces
//!
//! A guard that fails often is the loop taking another path than the one the
//! trace recorded. Every branch guard counts its native exits
//! ([`native`]). Each time a guard has failed another
//! [`policy::SIDE_EXIT_THRESHOLD`] times, and the tree has room
//! ([`policy::MAX_SIDE_TRACES`] sides by default, [`policy::MAX_SIDE_DEPTH`] deep, a few
//! attempts per guard), the exit that just restored the frame starts a
//! [`recorder::Recorder`] in side mode. It begins at the pc the guard
//! resumes at, with the interpreter state the exit restored, and records the
//! rest of the iteration along the path the interpreter takes. It closes
//! where the loop header comes round again, as a `Link`: a side trace that
//! rejoins the loop. Anything the recorder cannot follow (a call it cannot
//! inline, an inner loop, a path that leaves the loop) aborts it, and an arm
//! that will always fail spends all the guard's attempts at once. The one
//! inner loop a recording follows is a `string.gmatch` loop with a few trips:
//! it is unrolled, each trip recorded after the one before, and the step that
//! ends the loop is a guard that a side trace continues for a line with
//! another number of matches (see `Recorder::check_branch_of`).
//!
//! A branch guard that fails after stores of the iteration committed (the
//! body writes `gn[i] = v`, then the branch of the iterator step goes the
//! other way) is an exit of class `CommittedEffect`: the snapshot resumes at
//! the guard with those stores done and never replays them. That is the same
//! state a side starts from, so such a branch guard takes a side like one
//! that fails before any store (`native::side_exits_of`). Without it, a loop
//! whose body stores and then runs an inlined stateful iterator (`for x, y, z
//! in ipos(...)` over an upvalue-writing closure) left to the interpreter at
//! every row, with no side trace ever recorded. A table read (`Array`), a
//! callee or class guard after a committed store takes a side as well, from
//! the same state. A guard whose exit rebuilds inlined frames
//! (`restore.inner` is not empty) takes a side that starts inside those
//! frames when the calls are plain ones (see "Sides from exits inside
//! inlined frames" below). A guard inside an inlined call whose exit resumes
//! in the root frame before the call takes a side that starts at the call.
//!
//! A side that ends with `LoopNotClosed` because the loop itself ended on its
//! path (the last row of a stateful iterator, back in the root frame, at a
//! guard that continues the root's own back branch) spends one attempt and
//! retries soon. The same guard fails on rows that go on, and the next
//! failure may close. Before, such an abort spent all of the guard's
//! attempts, as for an arm that always leaves the loop. Mineclonia's
//! `ipos3` loop at mineshaft.lua:299 then left to the interpreter at each row
//! advance: about 2 iterations per entry, against 22 once the retry is
//! allowed (in the repro `mineshaft-299.lua` of "Mineshaft corridors"
//! below).
//!
//! # Calling host functions
//!
//! A loop that calls a host C function (a C API callback, such as Luanti's
//! `core.get_biome_data`) used to abort its recording at the call and stay in
//! the interpreter. A trace now makes the call itself, without leaving, when
//! the loop is of the shape the call can leave alone. The call is a helper
//! ([`oximond::jit_ir::trace::HelperId::vm_call`], one per set of result
//! kinds, the way `coroutine.resume` has them) with the effects of a helper
//! that runs a coroutine ([`oximond::jit_ir::trace::HelperEffects::RESUMES`]):
//! the function may call back into Lua and change anything, so the trace
//! holds nothing across it that the code could invalidate
//! (`TraceIr::resume_conflict`), the helper reads the words of every array,
//! length and header binding again afterwards, and checks that the cells,
//! globals and captured locals the trace read still hold what it read
//! (`RunTables::refresh_bindings`, `externals_hold`). A recording that ends
//! up holding something the function could change is refused
//! (`Recorder::refused_vmcall`), the site is marked (`Site::no_vmcall`) and
//! the next recording aborts at the call, as it did before traces made host
//! calls.
//!
//! The recording sleeps through the call, as it does through a resume
//! (`recorder/inline_vmcall.rs`): the interpreter makes the call, the Lua
//! code the function calls back is skipped (the VM does not show it to the
//! recorder, `TracePolicy::skips_nested_host`), and the registers after the
//! call say what it gave. A call records outside a coroutine, from `Call`,
//! `CallOne` or `TailCall`, in the root frame or a frame the trace inlined,
//! with no count hook and no table the trace never made in a register, and
//! up to three results of integers, floats, booleans, nil, strings and
//! tables (the kinds are the base-6 digits of the helper id). A `TailCall`
//! gives its results as open results to the `ReturnOpen` after it, and the
//! helper takes it only when the function gave exactly the recorded number.
//! Every other call aborts the recording.
//!
//! At run time native code spills the values of the call's `Before` snapshot
//! to the exit area, exactly as an exit does, and calls the helper
//! (`helpers/vmcall.rs`). The helper writes the snapshot back to the frame
//! (`RunTables::restore_exit`, the restore plan of that exit), so the frame
//! is what the interpreter's would be in front of the call, and makes the
//! call through the interpreter's own host boundary with the arguments in
//! their registers copied out (`Vm::host_call_for_trace`, the sequence of the
//! dispatch loop's host call). The collector sees the run in flight (whole
//! windows and the bank as roots), native execution stays off for what the
//! function runs, and a flush waits. Then:
//!
//! - results of the recorded kinds go to the trace and it carries on. A nil
//!   result needs no word; a table or a string gets a lane of the run's bank
//!   (`give_lanes`), which roots it until the run ends, and a table gets its
//!   row filled (`RunTables::fill_row`), so the trace reads its fields as
//!   those of any table met by lane. A run out of lanes leaves like a result
//!   of another kind;
//! - the helper answers before the call, with the frame written back, when
//!   the interpreter would not make the call itself: a hook, a coroutine,
//!   a callee that is another function or no host function. The trace
//!   leaves through the call's guard, and the interpreter makes the call;
//! - an error is mapped as the dispatch loop maps it and kept in
//!   `Vm::vmcall_error`; the trace leaves with the frame after the call and
//!   the entry raises the error from there, so the message, the traceback
//!   and a `pcall`'s result are the interpreter's and no hook sees the call
//!   twice;
//! - a call that completed with other results than the recording saw, or
//!   that installed a hook, wrote a local (`debug.setlocal`, `lua_setlocal`)
//!   or asked for a flush or a collection, leaves
//!   with the frame as the call left it (`TracePolicy::vmcall_landed`): the
//!   exit restores nothing and the interpreter resumes after the call. A
//!   call hook the function installed sees the call once it returned, as the
//!   dispatch loop reports it.
//!
//! What holds across the call. The recorder already forgets its facts about
//! array slots at any helper that writes the heap, and the optimizer takes
//! the helper as a barrier: no read is forwarded, merged or hoisted across it,
//! and the metatable guards before it answer for none after it. The run does
//! the rest when the call returns (`helpers/vmcall.rs`): it reads the words of
//! array, length and header bindings again, finds the slot of every field
//! binding again (and misses if the field now holds another type), checks
//! what the metatable of a class holds and that a table without a metatable
//! has none, sends the lookups of a string index to the VM from then on, and
//! makes every row of a table met by lane again
//! (`RunTables::refill_rows`). So a loop may read fields, guard metatables and
//! look up hash keys on both sides of the call (`HashLoad`,
//! `TableFieldLoad`, `MetaGuard`), read and write array words of tables met by
//! lane, and set table keys. A cell, global or captured local the trace
//! writes lives in a native value until an exit stores it: the call's restore
//! stores it, as an exit does, and the run compares it with memory again
//! after the call (`RunTables::written_externals`), leaving if the function
//! changed it. `TraceIr::resume_conflict` still refuses a trace that holds
//! a header binding, a call that writes besides the host calls or a table of
//! the trace's own beside the call, and every one of these beside a
//! coroutine.
//!
//! A string a register holds as a borrow of an array slot is given a lane of
//! its own in front of the call (`HelperId::STRING_OWN`), since the function
//! may move the array. A host call is not an operation that ends the borrows
//! of strings without this.
//!
//! A call inside a frame the trace inlined (an `__index` function too) is made
//! with the frames rebuilt: the restore pushes them on top of the root as an
//! exit does (`RunTables::restore_exit`), so `debug.traceback`,
//! `debug.getinfo` and an error at level 2 see the interpreter's stack, and
//! the helper takes them off again when the call returned and the trace goes
//! on. A call that landed leaves them where the call left them. A call used
//! as a statement is made from the trace (`a_call_as_a_statement`).
//!
//! The MIR backend declines a trace with such a call (the Cranelift backend
//! spills the exit), and the compile falls back. `OXIMOND_JIT_NO_VMCALL`
//! turns the recording off, for comparisons, and
//! `Vm::set_trace_vmcall (false)` does it for a test. Tests:
//! `tests/trace_vmcall.rs`.
//!
//! Measured on 2026-11-05 on the `perf/integ` branch with a microbench host
//! (a C program that registers `cadd` through the C API and runs a loop of
//! `k` arithmetic statements around one call; release, one pinned CPU,
//! `perf stat -e instructions:u` under `env -i`, the slope between 100000 and
//! 200000 iterations), instructions per iteration with the calls from the
//! trace and in the interpreter: k=1 1545 and 1721, k=2 1570 and 2297, k=4
//! 1621 and 3449, k=8 1721 and 5754, k=24 2121 and 15064. `LuaJIT` runs k=8 in
//! 489 and k=24 in 938. Single-stepped, one k=8 iteration is about 1860
//! instructions: the compiled code 250 (the 16 statements, the spill of the
//! state in front of the call, the call), the restore of that state to the
//! frame (`RunTables::restore_exit`, `ExitRestore::apply` and `write_action`)
//! about 410, the VM side of the call (`helper_vm_call`,
//! `host_call_for_trace`, the copy of the arguments, the results and their
//! vector) about 440, the host boundary about 240, and the C API bridge and
//! the C function about 470, which the interpreter's own call pays too.
//! Binding and cell checks after the call cost 55.
//!
//! The calls cost about 500 instructions more than the interpreter's (the
//! restore, and the checks afterwards), and the interpreter spends about 23
//! instructions on a guest instruction of this body, so a loop pays from
//! about 22 guest instructions. `VMCALL_MIN_GUEST` is 20, and a recording with
//! fewer aborts at the call: a loop of `s = s + biomes[cadd (i, 1) % 64 + 1]`
//! (11 guest instructions) costs 1796 with the calls and 1442 without, and
//! stays with the interpreter. `OXIMOND_JIT_VMCALL_MIN_GUEST` sets the bar,
//! for measuring it.
//!
//! What it does not do: more than three results, results that are userdata or
//! functions, a call with open arguments or through `pcall`, and a loop
//! whose recording holds a table of its own (a virtual table) across the
//! call. Those recordings abort at the call. A table or string result takes a
//! lane of the run's bank, which holds 256 tables: a loop that gets a new
//! table from every call leaves the trace once per 250 or so iterations to
//! start a run with an empty bank.
//!
//! # Value selects
//!
//! Not every branch that flips from one iteration to the next is another
//! path. `(walkable_p (cid) and 1.0 or 0.0)`, `if x < 0 then v = -x else
//! v = x end`: both arms only load constants, move registers or do number
//! arithmetic, and meet a few instructions later. A branch guard there fails
//! on every other iteration, and in Mineclonia's density loop each of the 27
//! inlined calls has one, after an upvalue write: every failure is an exit
//! through the call's frames or a side that holds the rest of the iteration.
//! The recorder records both arms of such a branch instead, decoding the arm
//! the interpreter skips from the chunk's code, and gives each register they
//! write a [`oximond::jit_ir::trace::TraceOp::Select`] on the branch's
//! condition (see the recorder's `select` module for the shapes). The
//! branch has no guard and no snapshot, so no exit, no side and no side key
//! comes from it.
//!
//! The rule that keeps this exact: every operation of an arm is pure and
//! total on the types the recording guards (constants, moves, `not`, and
//! `+`, `-`, `*`, `/`, unary `-` on integers and floats; never `%` or `//`,
//! whose integer forms raise on a zero divisor, nor a table access or a
//! call, which may run a metamethod), both arms give each register one
//! integer, float or boolean type (or the same value), and the arms join
//! inside the loop without a backward jump. Running the arm the interpreter
//! skipped is then unobservable, and the select gives every register the
//! value the interpreter's arm gives it. A recording that counts
//! instructions for a count hook keeps its branch guards: the arms may have
//! different lengths, and the count the trace charges must be the
//! interpreter's. The region in the root frame's caller keeps them too. A
//! select holds numbers and booleans only, so it adds no lane, no root and no
//! write barrier. An arm runs at most six instructions, its jumps included
//! (`MAX_ARM_INSTRUCTIONS` in the recorder's `select` module), and both
//! successors of the branch lie after it. `OXIMOND_JIT_SELECTS=0` keeps every
//! branch guard, to measure what the selects change. A number of run-time
//! subtype (`number.rs` in the recorder) declines the select, since a select
//! has one type. Both arms end on one source line with the same locals live:
//! a snapshot after the join names the recorded arm's last pc as the
//! previous one, which only debug information reads before the interpreter
//! replaces it.
//!
//! An arm decides some things from types alone: a `JumpIfFalse` on a nil
//! or a number has a fixed direction, and two different values of one type
//! join (two nils without a select). When that value is a carried register,
//! the type is the recording's entry type, and nothing guards it unless an
//! op reads the register. A select that succeeds therefore adds every such
//! register to the recorder's `condition_reads`, which makes it an entry
//! input with a type guard, the same rule as a `JumpIfFalse` on a carried
//! non-boolean outside an arm. A declined select adds none.
//!
//! A select changes which guards a loop keeps: `scale = flag and 2 or 3`
//! in front of a loop takes away the guard on `flag` there and leaves the
//! loop's own `if flag` as its first branch on an invariant, which then
//! moves into the preheader. A select itself never leaves, so it adds no
//! exit; the exits it changes are those of the hoisted guards, whose entry
//! snapshot must name the state the loop started from (see "Loop-invariant
//! code motion" in `jit_ir::trace::optimize`). That snapshot was wrong for
//! loop sides, and the shape-359 miscompile below showed it through a
//! select. `trace_value_selects.rs` pins it with the select's value stored
//! through `TABLE_SET_INDEX` into a nil hole.
//!
//! # Sides from exits inside inlined frames
//!
//! A guard that fails inside an inlined call leaves through the call's
//! frames once the call wrote an upvalue cell or ran more than
//! `BEFORE_CALL_MAX_PREFIX` (64) guest instructions; before that it leaves
//! through the `Before` snapshot of the call, and the interpreter makes the
//! call again (see `Recorder::starts_call_over`). The exit pushes the
//! callee's frames for the interpreter (`push_inner`, counted in
//! `inner_exits`). Such an exit takes a side when `ExitRestore::seedable`
//! holds: at most `policy::MAX_SEED_FRAMES` (4) frames, each made by a plain
//! call that keeps results (no `pcall`, metamethod, varargs, tail call or
//! closure that varies by run), no virtual table in the state and no tail
//! call of the root.
//!
//! The side starts where the guard resumes, in the innermost callee
//! (`SeedPlace::inner`, `recorder::SideSeed::frames`). The recorder rebuilds
//! each call as an inlined frame of its own (`Recorder::start_seed_frames`):
//! the frame runs the pinned closure the parent's snapshot names, and each of
//! its registers is either a string constant of the root bank or a
//! `FrameInput` value (`TraceIr::frame_inputs`, `TraceIr::seed_frames`)
//! that reads what the parent's state holds there. A frame input binds to a
//! typed value of the parent's snapshot frame (`plan_tree_with`), or, for a
//! side compiled apart, to what the exit would have written
//! (`emit::detached_inputs`). A register the snapshot leaves `Unchanged`, a
//! borrowed string or a virtual table has no such value, and the side is
//! refused. The root frame's window slots (cells, globals, the environment)
//! are read from the root frame at the start (`RootView`), not from the
//! callee the first hook shows. The recording then returns out of the calls
//! like any inlined call and links back at the loop header. A guard inside
//! the side's own rebuilt frames leaves through them too, so a side of such
//! a side starts in the same frames.
//!
//! A seeded side has no caller link, enters no enclosing loop and is not a
//! function trace's side. Each tree may record seeded sides for at most
//! `policy::MAX_SEEDED_GUEST_INSTRUCTIONS` guest instructions in all
//! (`SideBook::seeded_guest`); a guard whose calls cannot be rebuilt spends
//! all its attempts at once. The cells and environment the side reads at its
//! start belong to the root frame, which is live below the callee frames
//! during the whole recording.
//!
//! A store helper that reaches its table by lane (`TABLE_SET_INDEX`, the
//! `rawset` of an inlined `__newindex`) may grow the array part of a table
//! that the tree also binds, because the handler walk hands the call the
//! same table the loop reads `#t` from. The helper therefore rewrites the
//! array and length words of the table it stored into
//! (`HelperEffects::REWRITES_WORDS`). Before, the border word stayed one
//! behind after a growth, the next key named a slot that held a value, and a
//! loop that gained a `__newindex` between runs lost one iteration per
//! growth (`trace_table_writes`, `trace_hot_exits`). That loss came from the
//! helper, not from sides: it happened with no side installed.
//!
//! `TraceStatistics::interpreter_exits` counts the exits that resume the
//! interpreter by exit class and guard kind, so the exits no side took show as
//! large cells (`OXIMOND_JIT_STATS=1`).
//!
//! A branch guard that a side continues stays in the loop body of its trace.
//! The optimizer hoists a guard on an invariant to the preheader, where it
//! fails before the loop and no side can start; but a link brings the loop
//! new values for what the guard tests, so an invariant of the root's loop
//! (the round of the loop around it) is none of the tree's.
//!
//! A closed side is added to the trace's tree
//! ([`oximond::jit_ir::trace::TraceIr::sides`]), in one of two ways.
//!
//! The usual way compiles the whole tree. It is checked
//! ([`oximond::jit_ir::trace::plan_tree`]) and compiled as one native function:
//! the failing guard jumps into the side in place, passing the values its
//! snapshot holds, and the side's end jumps back to the root's loop header
//! with one value per loop parameter. Nothing returns to Rust, no frame is
//! rebuilt, the fuel carries across (each link spends one unit like a
//! loop-back, and a poll exit inside a side publishes the state the header
//! would see), and the native roots stay what they were. The compiled tree
//! replaces the trace's code at a safepoint ([`CompileTarget::Retree`]). A
//! side inline costs the loop nothing at all, but it costs the compile of the
//! whole tree, which takes about 13 000 instructions per line of CLIF
//! ([`TraceStatistics::clif_instructions_tree`]): millions for a tree of a
//! few hundred.
//!
//! The other way compiles the side alone, as a function of its own that the
//! exit of the guard calls ([`CompileTarget::Side`], `jit::compile_side`), and
//! leaves the tree's code as it is. Every exit of a tree has a cell, which is
//! empty or holds the entry of a side (`CompiledTrace::attach`). When a run
//! leaves the code through an exit, it reads the exit's cell and calls the
//! side it finds there with the run's own buffers. The exit area holds the
//! state the exit restores, which is the side's entry state, so the call is a
//! step inside the run (its tables, its lanes and the fuel stay as they are)
//! and not an entry of the trace. The side ends in an exit of its own, or in a
//! link back to the loop. An exit of a side returns a code that names the side
//! (`unit << 24 | index`), its descriptor joins the tree's, and it has a cell
//! too, so sides hang from sides. A link hands the run the loop parameters in
//! the exit area. The run puts them in the entry input slots they start from
//! and enters the tree again, which runs the loop on.
//!
//! A call costs the loop a hundred instructions or so over a side inline
//! (`CHILD_CALL_COST`), and the compile of the tree with the side inline costs
//! millions. A side is therefore compiled apart only for a guard that failed
//! in fewer than one in `LINKED_RARITY` of the iterations since the tree
//! installed (`RARITY_EVIDENCE_ITERATIONS` of them at least), and a tree whose
//! sides have cost more than its compile would is compiled again with them
//! inline (`InstalledTrace::wants_relink`). A guard that fails often gets its
//! side in a compile of the tree, as before. So does a side of a root that
//! other loops enclose, until the tree has run as long as an outer side waits
//! for (its sides come in bursts, which one compile of the tree takes
//! together), unless the side returns into the caller's frame.
//!
//! A side must add nothing to the code it joins. Planned with the tree
//! ([`oximond::jit_ir::trace::plan_tree_with`]) the root must come out as it
//! is, so the side widens no register and adds no entry input. A side that
//! links back may write only what the optimizer of the root already assumed
//! some member writes ([`oximond::jit_ir::trace::TreeWrites::covers`]), and
//! the run must be set up for everything it does (`TreeFacts`). A register
//! the side reads that the root does not is read from an extra input slot
//! after the entry inputs, and the side checks its type itself: it declines to
//! run if it finds another (`DECLINE_CODE`), and the exit is taken as it would
//! be with no side.
//!
//! A register the side writes that the root does not carry as a loop
//! parameter goes through the run too (below). Where it cannot, the side
//! leaves at the loop header instead: it ends in its poll exit, which
//! restores what the link names and resumes the interpreter, which enters the
//! tree again. That costs an entry of the tree per run of the side
//! (`LINK_EXIT_COST`), so only a guard that fails in fewer than one in
//! `EXIT_STYLE_RARITY` iterations takes such a side. Its exits do not count
//! as entries in the productivity window, and the same accounting as for the
//! calls compiles the tree again with the side inline when they add up.
//!
//! The cell is the only thing patched, and it holds the entry of code its tree
//! owns. A result from the worker installs at a safepoint on the VM's thread,
//! never while native code runs, and is checked against the trace's serial and
//! version and the flush epoch like a tree's, so a flush, an eviction, a
//! retirement or a retree while the side compiles drops it. The side is added
//! to the compiled tree's tables first, the restore plans of its exits and the
//! cache's charges are made, and the cell is stored last (release): one
//! aligned word, which a run reads once (acquire), so a run sees no side or a
//! whole one. If the plans cannot be made, the side is taken out again before
//! any run could see it and goes back to the queue for the tree's compile. A
//! flush, an eviction, a retirement or a retree drops the tree with its sides.
//!
//! A side whose next queued side is a loop side recorded behind it
//! (`PREVIOUS_SIDE`) is not compiled apart, so the pair waits for the tree's
//! compile (`plan_detached_inner`). The loop side of an enclosed root waits
//! for that compile anyway (the rule on enclosing loops above), and there it
//! found no exit of the side compiled apart to hang from ("no exit continues
//! its guard"): it never landed, and the side left at its terminal exit to
//! the interpreter on every run. A side that returns into the caller, and a
//! side of such a chain, still compile apart: their loop sides may follow
//! them apart, each hanging from the exit of the one before.
//!
//! A side refused when the tree grows is stale, not bad, when another side
//! landed since it was recorded and the refusal is one that landing explains
//! (`native::refused_as_stale`): a mismatch of window, binding, pin, type or
//! link value, or the verifier's "a side trace has another window than its
//! root". Its guard records it again. The verifier case is two sides of one
//! parent recorded while a compile was in flight, each extending the root's
//! window from the same tree: the second no longer fits once the first has
//! joined. In the emerge, Mineclonia's density loop (`mg_ersatz.lua:408`) lost
//! the side of a hot guard that way, and that guard's exits retired the tree
//! (1 run of 7). The other exits of 408, and those of the traces at lines
//! 174 and 197, are not a defect: a temporary exit log showed warm-up exits
//! at two guards until the async compile of their sides lands, then per call
//! one poll exit and one exit at the loop end (the iterator's `x1 < x2`
//! guard, classed `GuardFail`, which ends the call's loop). They run native
//! between those exits.
//!
//! # Sides that run through the loops around the root, and sides that return
//!
//! An outer side and a side that returns from the root frame into the frame
//! that called it write registers the root does not carry (the outer loops'
//! counters, a caller's loop and its state), and a side that returns reads the
//! caller's registers. Planned with its tree, such a side *widens* the root:
//! nbody's side that returns from `advance` into the loop that calls it gives
//! the root five registers of the caller frame as entry inputs (the window
//! grows from 2 to 7 slots, the entry from 12 inputs to 17, the pinned
//! closures from 1 to 2), and the compile of the grown tree costs 25M
//! instructions, which repays after about 2900 calls. A side compiled apart
//! changes nothing the root's code is made of, so it takes these registers
//! some other way:
//!
//! - *Reading.* The registers a side reads of the caller are slots of its own
//!   window ([`oximond::jit_ir::trace::ExternalSlot::CallerRegister`]) beyond
//!   the tree's. `plan_tree_with` keeps those slots, the pinned closures
//!   beyond the root's and the side's own pins out of the root. The side
//!   reads them from *extra input slots* after the entry inputs and the
//!   bindings (`CompiledTrace::extras`), which the entry fills from the
//!   caller's frame, like any entry input, from the registers `trace.caller`
//!   checks (a tree with a side that returns is entered only by the frame
//!   that the same caller called the same way, see "Linking through a
//!   return"). A global of the caller, or an upvalue of the caller's closure,
//!   that the side only reads is an extra input too. The side checks the type
//!   of each extra input itself and declines to run if it finds another
//!   (`DECLINE_CODE`), and it pins what it needs behind the root's lanes, which
//!   every run pins with the tree's.
//! - *Writing.* A side that links back hands the run the root's loop
//!   parameters in the exit area, as before. The other live registers it
//!   wrote (the outer loops' counters, the caller's counters, the arguments of
//!   the next call) it stores into the run's input slots itself, in the
//!   entry input the root reads the register from, or in an extra input
//!   slot (`DetachedSpec::link_writes`, `apart_link_writes`). The root reads
//!   its entry inputs again when the run enters it, so the new values are its
//!   invariants. Tables and closures go as lanes of the run's bank, which keeps
//!   them for the run; a string, a coroutine or a table the root binds does
//!   not (the side then compiles with its tree).
//! - *Leaving.* The tree's exits know nothing of these writes: they leave a
//!   register as the interpreter's frame holds it. The same goes for the
//!   root's loop parameters at an exit before the iteration writes them,
//!   which after a link hold what the link handed the root, not what the
//!   frame held at the run's first entry. A run that linked copies the slots
//!   its sides rewrote, and the link slots of the loop parameters, to the end
//!   of the exit area (`CompiledTrace::writebacks`), and the VM writes them
//!   into the frames (the root frame's, or the caller's below it), or into
//!   the upvalue cell of a window slot, before it restores the exit
//!   (`native::write_back_links`). An exit of the side itself names the
//!   caller's registers it changed through its own snapshots, whose window is
//!   the side's: `ExitRestore::callers` writes them, the exit pops the root
//!   frame and resumes the caller (`InnerPlan` with `Entry::Caller`), and the
//!   frames above are pushed as for any exit, so the full frame stack of the
//!   interpreter is rebuilt wherever the run leaves, after the first call or
//!   the thousandth. The restore plan of each side's exits is made when it
//!   joins, and only for its own exits.
//! - *Patching.* Nothing changes for the worker: the side compiles from a
//!   snapshot of the tree (`SideContext`: the root's entry contract, the
//!   exit it continues, the extra inputs so far), and its result installs at a
//!   safepoint on the VM's thread only if the trace's serial, tree version
//!   and the flush epoch are the ones it was made for. `prepare_attach` adds
//!   the side's extras and the slots it rewrites to the tree (a slot another
//!   side rewrites has the same place in both), the restore plans and the
//!   cache's charges follow, and the cell is stored last, so a run sees no
//!   side or a whole one. A run whose exit leaves through a side that was
//!   attached while no run was in flight finds its extras and its pins in
//!   place.
//!
//! A side that returns into the caller hangs off the exit of the tree like
//! the others, and the loop sides recorded right behind it (see "Loop sides")
//! hang from its terminal exit, one after the other, each compiled when the
//! one before has landed (`InstalledTrace::chain_parent`): nbody's link into
//! `advance` is a side that ends in front of the first pair loop, the loop
//! side of that loop, and the side that links back.
//!
//! Which sides compile apart is a matter of cost, tuned with the whole suite
//! (see the constants in `native`). A side that returns saves the entry and the
//! exit of the tree for every call, thousands of instructions, so it is never
//! held for the rarity of its guard and is recorded after
//! `RETURN_THROUGH_ENTRIES` entries whatever the size of the tree. An outer
//! side that runs through loops is called once per outer iteration, 260
//! instructions or so over the same side in the tree, so it compiles apart only
//! when its guard is rare in a tree that has run `SIDE_BEYOND_ITERATIONS`
//! iterations, which is long enough to have seen the rate (mandelbrot's guard
//! looked rare for 18 000 iterations and failed every 25th afterwards). A tree
//! whose calls of sides have cost more than its compile would is compiled
//! again with them in (`InstalledTrace::wants_relink`).
//!
//! The root has to hand an in-tree side the registers it needs and take back
//! the ones the side wrote, so such a tree is *widened*: see
//! [`oximond::jit_ir::trace::plan_tree`]. A side reads and writes the same
//! window as the root and shares its cell aliases, adding none. A side may
//! pin callees behind the root's lanes, so it can inline a call of its own.
//! A
//! guard inside an inlined callee whose snapshot rebuilds the callee's
//! frames takes a side that starts inside them when the calls are plain
//! (see "Sides from exits inside inlined frames"); the callee identity
//! guards and the guards of other calls take none. A branch guard inside a
//! call that has written no cell and run few instructions leaves before the
//! call instead, so its side starts at the call and inlines it along the
//! other arm.
//!
//! # Nested loops
//!
//! A trace records one loop, so the root trace of a nest belongs to its
//! innermost loop, and its exit is hit once per iteration of the loop around
//! it. Entering and leaving native code costs about 3000 instructions, which
//! is more than a short inner loop runs. The tree takes care of that: when
//! other loops of the chunk enclose the root's loop, its exits can take side
//! traces that run *on through those loops*. Such an outer side starts where
//! the exit resumes, records the rest of the outer iteration (the code after
//! the inner loop, the outer loop's own latch with its continue guard, and
//! the code before the inner loop, including its numeric for prep) and links
//! back to the root's header. It is an ordinary side, with the same checks,
//! the same widening and the same compile (see "Side traces"). Only three
//! things differ:
//!
//! - The recorder may leave the root's span, and may take the backward
//!   branch of an enclosing loop. It still aborts at the backward branch of
//!   any other loop (a second inner loop, say, which would need a root of its
//!   own) and wherever a side aborts (a call it cannot inline, strings with no
//!   root slot).
//! - The outer loop's own exit (its latch falling through, a `break`) is a
//!   guard of the side. It is an exit like any other, and takes a side of its
//!   own, so a nest of any depth up to [`policy::MAX_SIDE_DEPTH`] becomes a
//!   chain of sides, one per level, all in one function.
//! - A side recorded from the interpreter (a scout's, below) or in another
//!   call may have seen a local it only passes on, one in scope where it runs
//!   that it neither reads nor writes, under another type than its parent
//!   hands it: a nil-or-number bound like `ly_dom` of `region_evaluate` is
//!   nil on one call and a number on the next. Planning gives such an input
//!   the parent's type when one of the two is a number
//!   (`retype_passed_inputs` in [`oximond::jit_ir::trace`]'s tree planner),
//!   so the side binds instead of retiring the tree.
//!
//! The header of an outer loop is `Covered` ([`policy::SiteState`]) as soon
//! as the tree holds a side that took the loop's backward branch: the
//! interpreter never sees that backedge again while the tree runs. A tree
//! that is retired, evicted or flushed hands its covered headers back, and
//! an uncovered header is rejected.
//!
//! Coverage is also revocable. Every interpreted backedge of a covered header
//! is an outer iteration the tree did not run. After
//! [`policy::COVERED_WINDOW`] of them the policy checks that the tree ran at
//! least [`policy::COVERED_MIN_ITERATIONS`] native iterations per one
//! ([`policy::COVERED_MIN_ITERATIONS_FLOAT`] for float trees). If it did not
//! (the arm that carried the outer side stopped being taken, or its guard
//! fails every time and no side can land), the header is revoked and
//! rejected for good (`covered_revoked`).
//!
//! The root of a nest whose inner loop runs fewer iterations per entry than a
//! trace needs to pay for one is installed only to give its exit the outer
//! side: the tree compiled over it replaces it within a few dozen entries
//! (`TracePolicy::is_stepping_stone`). With one loop around the root, that
//! root is not compiled at all. It is installed as a placeholder that holds no
//! code ([`oximond::jit::CompiledTrace::staged`]), and the sides of its tree
//! are recorded in the interpreter, one after the other, before the tree
//! compiles once at the optimized tier (`TracePolicy::stage_root`):
//!
//! 1. A *scout* (a recorder that records nothing, see `Recorder::scouting`)
//!    watches the interpreter run the frame and wakes when it leaves the root's
//!    loop. The outer side is recorded from there, as one recorded from the
//!    exit of compiled code would be.
//! 2. The scout of the next side follows the path of the one just recorded.
//!    It wakes at the guard of that side the interpreter fails to leave the
//!    outermost loop around the root, and the code beyond that loop is
//!    recorded (see "Loop sides"). A failed guard that leaves the nest is the
//!    one the compiled code would take at that point; a recording that does
//!    not end in front of a loop is dropped, and the scout watches for the
//!    next place, a few times (`native::EAGER_BEYOND_TRIES`).
//! 3. The queued sides compile together with the root.
//!
//! The trips of a `while` or generic-for loop are not known when its root
//! closes, so such a root is compiled at the policy's tier and judged at its
//! first productivity window instead (`InstalledTrace::keep_stone`): when
//! loops enclose it (`nest::can_cover`) and the window closes below the bar,
//! it becomes a stepping stone and is not retired while a side of its tree
//! is on its way, or while most of its runs leave by guards that can still
//! take a side and a side landed within [`policy::STONE_PATIENCE_ENTRIES`]
//! entries. Each side that lands starts that count again, so a three-level
//! `while` nest grows its sides one level after the other. Once the guards
//! have spent their attempts (the outer part has something no side can
//! record), or the runs leave where no side starts, it retires as any root
//! does, and a second short retirement marks the loop `retired_short`, which
//! lets the loops around it unroll it (see "Loop sides"). A root of a
//! loop whose stones gave up twice (`Site::stone_gave_up`: their sides never
//! landed), or once with most entries finding types no version takes, is
//! not made a stone again.
//!
//! Once a root is a stone it keeps its branch guards in the loop body
//! (`TraceIr::keep_branch_guards`), and code that hoisted one compiles again
//! (`TracePolicy::consider_kept_guards`, a tree compile of the root and its
//! sides): its recorded trip may hold still what the next run of the loop
//! changes (a `while lz ~= lzm or rz ~= rzm` recorded on a trip that steps
//! only `rz`), and since every link of an outer side runs the preheader
//! again, a hoisted guard would leave there on nearly every pass, where no
//! side can start. A long `while` loop never becomes a stone and keeps
//! hoisting its guards.
//! The end of an `ipairs` loop is such a root's outer exit too: the step's
//! read past the array leaves, and the outer side records the step reading
//! nil (raw, on a table with no metatable), leaves the loop and runs on.
//!
//! An exit that resumes at the outermost loop's latch normally waits until
//! the tree has run `SIDE_BEYOND_ITERATIONS`, since its side may run beyond
//! the nest. A latch that loops back now (a `while` loop's jump, a numeric
//! for latch with trips left) keeps the side in the nest, so that exit does
//! not wait: the exit of a `while` loop that ends a `for` body resumes at the
//! `for` latch on every outer iteration.
//!
//! A deeper nest, and a root the scout cannot follow, keep the stepping stone:
//! the root compiles at the baseline tier (no optimization, the single pass
//! register allocator), which takes about a third of the optimized tier's
//! compile time, and the tree compiles at the optimized one. If the outer side
//! never lands (every recording of it aborts, so the guard spends its
//! attempts), the root is the code that runs, and
//! `TracePolicy::recompile_stone` compiles it again at the policy's tier. The
//! placeholder gives way to the root alone the same way.
//!
//! # Loop sides
//!
//! An outer side that goes beyond the nest meets the loops of the code after
//! it. A numeric for loop with a few trips was unrolled into the side, one copy
//! of the body per trip, which made the compile of the tree several times
//! dearer. Now the side records the loop's first trip (a `while` loop's, or a
//! generic for loop's, as well as a numeric one's) and ends in front of the
//! second with a terminal exit ([`oximond::jit_ir::trace::TraceTerminator::Exit`]),
//! and the loop is a side of its own, a *loop side*, that continues that exit
//! ([`oximond::jit_ir::trace::TERMINAL_GUARD`]). It is recorded right behind
//! the side that ends in front of it (`native::PREVIOUS_SIDE`), from the
//! state the exit restores, so that both join the tree in one compile. A loop
//! side has loop parameters, a poll snapshot and the fuel of the root's loop,
//! and its guards take sides in turn; the loop ends when its closing guard
//! fails. It makes the loops beside a nest native whatever their trips:
//! none, one, a number that changes with every call, or a million, and each is
//! one copy of the body in the compiled code.
//!
//! A tree that counts instructions for a count hook has no loop sides: the
//! count hook's budget runs through the root's loop only. Neither does a root
//! recorded while a count hook is installed end in front of a loop (see
//! "Outer roots").
//!
//! # Outer roots
//!
//! A nest is carried by the root of its inner loop and the outer sides of
//! that root (see "Nested loops"). Two nests cannot be: two loops side by side
//! inside the outer one, where a side of one would abort at the other's
//! backward branch (`nest::can_cover` says no for them), and an inner loop
//! whose own root was given up (a call no recording takes on the paths its
//! recordings saw, say). For those the outer loop's root carries the nest:
//!
//! - The policy names the loops inside the root's span that no tree of their
//!   own carries (`TracePolicy::free_loops_of`): the header is `Rejected`, or
//!   its root is installed, queued or compiling but cannot cover the loop
//!   around it. A loop with loops of its own is not free (its loop side
//!   would abort at their backward branches), nor is one given up for its
//!   few trips (`ShortLoop` or `LoopNotClosed`). Loop sides that leave after
//!   a trip or two cost the three-level nest of Mineclonia's
//!   `mcl_util/shape.lua` 9% more instructions, most of it in retired trees
//!   recorded and compiled again. Any other state (counting, recording, cooling down, a stone,
//!   `Covered`) leaves the loop to its own root, and the recording of the
//!   outer header aborts at it with `InnerLoop` as before.
//! - A root recording that takes the backward branch of a free loop (a
//!   numeric for latch with more trips than unrolling takes or counters that
//!   follow the root's, a `while` loop's jump unless the loop is one the
//!   policy found short, a generic for latch) commits the branch and ends at
//!   the loop's header with a terminal exit
//!   ([`oximond::jit_ir::trace::TraceTerminator::Exit`]), as a side that ends
//!   in front of a loop does. The root has no loop-back of its own: it keeps
//!   its loop parameters and a poll snapshot of its own header's shape, which
//!   only define the loop's span and what the sides link with, and no run
//!   takes that poll. Its terminal exit takes the loop as a loop side
//!   (`native::terminal_enters_loop` accepts member 0 and every loop kind),
//!   recorded from the exit when it fires as any side is.
//! - The sides of such a tree end in front of any loop inside the root's span
//!   they meet ([`recorder::SideSeed::inner_loops`]): the side of the first
//!   loop side's closing guard runs the rest of the outer iteration, ends in
//!   front of the next inner loop, whose loop side follows it, and the side
//!   of the last closing guard links back to the root's header. The whole
//!   outer iteration runs in the tree, and only the outer loop's end leaves
//!   it.
//! - A side of a loop side (a branch in the inner loop's body) that comes
//!   round to that loop's header rejoins the loop
//!   ([`oximond::jit_ir::trace::TraceTerminator::LinkLoop`]): native code
//!   jumps to the loop side's header with one value per loop parameter of the
//!   loop side, found by register in the link's snapshot, and spends a unit
//!   of fuel. The loop side's header dominates the jump (the side is lowered
//!   inside the loop side's body), so the loop side's invariants are in scope.
//!   A link may write no register the loop side reads from its entry state
//!   without carrying it (`SideSeed::loop_headers` names them, and
//!   `plan_tree` checks it again): the loop side would go on with the value
//!   it entered with. Such a side ends in front of the loop instead, which a
//!   second loop side, a copy of the loop, continues. Carrying is not
//!   extended to loop sides. A side that links into a loop side is never
//!   compiled apart: its jump target is code of the tree's own compile.
//! - The tree covers nothing new: its loop sides are members of the tree
//!   whose own header is the root. It may cover the loops around it as any
//!   tree does, and for that its inner loops are no siblings
//!   (`nest::can_cover_from_outer_root`).
//! - A rejected inner loop no longer hides the loop around it: the skip a
//!   rejected loop arms swallows the backedges of every loop while no trace
//!   is installed, so the rejected loop keeps its lookups while a loop around
//!   it still counts, for up to [`policy::UNSKIPPED_LOOKUPS`] of them, whatever
//!   turned it away.
//! - Each loop side and each copy counts against
//!   [`policy::MAX_SIDE_TRACES`] and [`policy::MAX_SIDE_DEPTH`] like any side,
//!   so a tree cannot grow copies without bound.
//! - A side that ends with an exit (in front of a loop, or at an ancestor
//!   loop side's header) and a loop side (whose poll resumes at its header)
//!   take every register live where they resume as an entry input, also the
//!   ones they never read, so the snapshot of that exit names each as a
//!   value. The interpreter register of such a register is the run's entry
//!   state, which the tree's links and loop sides have moved on since: the
//!   outer loop's variable after a few outer iterations ran natively. The
//!   loop side that continues the exit reads its inputs from that snapshot,
//!   and one compiled apart would take an `Unchanged` register from the
//!   run's entry inputs. Without the inputs, a loop side behind a side that
//!   did not read the outer variable resumed the interpreter at an older
//!   outer iteration, which ran its trips again (a local that turned float
//!   on every hundredth outer iteration: 795574.0 for 796000.0 in Lua 5.4,
//!   and 3.1x the native iterations).
//! - A loop side with a parameter its loop-back passes on unchanged (an
//!   accumulator `acc = acc + 0` folds into, in the integer profiles) runs
//!   its preheader at its header, where the parameter is defined, as the
//!   root does at its outer header: the preheader may read it, the entry
//!   snapshot of its hoisted guards names it, and a side that rejoins the
//!   loop there (`LinkLoop`) may bring it a new value.
//!
//! Not done: a loop inside a loop side (a three-level nest whose middle loop
//! is free) still aborts the loop side's recording, and a generic for loop
//! whose step makes strings (`pairs` over string keys) cannot be a loop
//! side, since a loop side has no release of the strings its iterations make.
//! The end of an `ipairs` loop side over a table the tree reads by lane
//! takes no side either: the loop side reads the nil past the array, which
//! needs a proof of no metatable. An unrolled `ipairs` loop has that proof
//! from [`oximond::jit_ir::trace::HelperId::TABLE_ABSENT_INDEX`] (see
//! "Short generic for loops in calls"); a loop side does not use it yet.
//!
//! The other way to run a nest natively is to record the outer loop as the
//! root and *unroll* the short loops inside it, in its own frame or in a call
//! it inlines (`for _ = 1, n do step() end` where `step` loops over a handful
//! of bodies). The recorder follows the backward branch of a numeric for loop
//! that has at most [`recorder::MAX_UNROLLED_LOOP_TRIPS`] iterations left, up
//! to [`recorder::MAX_UNROLLED_LOOP_BACKEDGES`] times in a recording, and
//! records each trip after the one before. The loop's latch is a guard like
//! any other, so a run of another length leaves the trace there and a side
//! continues it. A longer loop is still an inner loop that ends the recording
//! (it keeps a trace of its own), as is a loop whose counters are computed
//! from the counter of the recorded loop (`for j = i + 1, n` inside `for i`):
//! its length changes with every iteration, so the trace would leave on all
//! but one of them, and every length takes a side and a compile of the tree
//! (a loop of `outer % 12` trips next to one of six: nine compiles and 545M
//! instructions where the interpreter takes 258M). A side of a root that
//! unrolls unrolls too, by the same rules: its counters are known to be
//! independent of the root's only when they are not registers that the loop
//! wrote before the side starts (`Recorder::depends_on_root_counter`).
//! A recording keeps the ordinary limits on instructions, IR
//! operations and snapshots. One that runs out of them while it unrolls
//! loops (a nest of several heavy bodies, as in `nbody`) is not tried again
//! with more room: such a trace takes milliseconds to compile, which a loop
//! that stops sooner would not repay. The loops inside keep their own traces
//! and the header is turned away as before.
//!
//! A short `while` loop is unrolled the same way. The recording that unrolls
//! short loops (the retry after an `InnerLoop` abort, `Site::inner_loop_met`)
//! follows the backward `Jump` of a `while` loop in a call it inlined, up to [`recorder::MAX_UNROLLED_WHILE_BACKEDGES`] trips
//! in a recording (`Recorder::may_unroll_while`). A `while` loop in the root
//! frame is left to its own root while that root may pay (an escape-time
//! kernel with 24 trips per pixel is that case: unrolled to the limit it
//! replaces the outer loop's trace with one that leaves at the loop's test on
//! most pixels). It is unrolled once the loop is known to be short: its root
//! was retired for running too few trips per entry (`Site::retired_short`),
//! or its header was given up. Both reopen the loops around it that were
//! rejected for meeting it (`TracePolicy::reopen_outer_loops`, once per
//! site), and the recordings of those, root and sides alike, follow its jump
//! (`TracePolicy::short_loops_of`). The loop's
//! length is not known ahead, so it unrolls the trips this iteration took:
//! the loop's test is a guard on every trip, and a call that runs another
//! number of trips leaves at that guard and takes a side trace, one per
//! length. A `break` or a `return` from inside the loop is a guard like any
//! other. A loop that runs past the limit, or any loop the unrolling
//! recording still cannot follow, sets `met_long_loop`, which rejects the
//! header at the second abort as before. A generic for loop is unrolled the
//! same way (see "Short generic for loops in calls" below). Before this, the first
//! abort at a `while` loop in an inlined call rejected the header, and the
//! loop became a root of its own that ran two or three trips per entry and
//! was retired as unproductive (`for i = 1, 300000 do t = t + run(...) end`
//! with a three-trip `while` in `run`: 525M instructions, 79M now). A loop
//! with more trips than a root's entry costs (eight) still keeps its own
//! root, and takes the loop around it with an outer side as before.
//!
//! A numeric for loop given up as short is unrolled the same way, whatever
//! its range has left (`Recorder::may_unroll_short`, `is_short_loop`). The
//! case is a scan with a long range that a `break` leaves after a trip or
//! two: the column scan of Mineclonia's `build_heightmap`
//! (`mg_ersatz.lua:123`, `for dy = ytop, ybot, -1`) looks down from the top
//! of an underground chunk and finds stone on the first trip. Its recordings
//! start on loop entry, on a trip that breaks, so every one ends with
//! `LoopNotClosed` and the loop is retired short. The column loop around it
//! then aborted with `InnerLoop` at the scan's latch, since a numeric loop
//! was unrolled only when its range had at most
//! [`recorder::MAX_UNROLLED_LOOP_TRIPS`] trips left, and the scan had 80. Now
//! the unrolling recording follows its latch like a `while` jump, sharing
//! the budget of [`recorder::MAX_UNROLLED_WHILE_BACKEDGES`] trips, and the
//! `break` is a guard. A run past that budget sets `met_long_loop` as before.
//! A side of the tree that meets the loop (a column of the surface layer,
//! with 30 to 60 trips of air) does not unroll it: it ends in front of it
//! and the loop runs as a loop side (`wants_loop_side_at`), as a side does
//! for any long inner loop. On the Emerge-0 attribution `mg_ersatz.lua:100`
//! (3.6% own, 8.2% inclusive) and its callee `nodeprops.lua:921` (4.5%) left
//! the interpreted list; the column loop at line 119 compiles with sides.
//! The repro `target/flow-luanti-native/repros/ersatz-123.lua` went from
//! 22.86 G instructions to 6.92 G (21.46 G interpreted). I tried unrolling
//! six trips in a side before the loop side takes over: the repro came to
//! 6.53 G, but side recordings of the emerge ran out of room
//! (`LimitExceeded`) and the emerge took 304.0/295.0 G against 286.7/286.7 G
//! with the loop side at once, so sides go to the loop side at once.
//!
//! Not done: unrolling an `ipairs` loop of a few elements in the root frame
//! of the recording; such a loop inside a loop that a tree can cover is a
//! stepping stone whose outer sides carry it instead (see "Nested loops").
//! Also not done: a loop with its own productive root inside a loop that
//! runs few trips (the root is kept and the outer side pays, as before).
//!
//! # Short generic for loops in calls
//!
//! The latch of a generic for loop (`ForInLoop`) in a call the recording
//! inlined is followed like the jump of a `while` loop
//! (`Recorder::for_in_loop`, `Recorder::may_unroll_while`), within the
//! same budget of [`recorder::MAX_UNROLLED_WHILE_BACKEDGES`] trips, and the
//! abort that turns the first recording away is the same restart
//! (`AbortSite::is_while_jump` covers the latch). An `ipairs` step of such a
//! loop follows the end of the list too: the recording reads the nil that
//! ends it through [`oximond::jit_ir::trace::HelperId::TABLE_ABSENT_INDEX`],
//! which leaves unless the key is still absent from a table with no
//! metatable (`__index` would answer past the end in Lua 5.3 and later, and
//! the interpreter runs it). The call takes a bound table too, not a read
//! past the array's end: that read has no lane form, and a side that binds
//! a table its root rewrites must give its bindings up (`plan_tree`).
//! Before, a recording could not end an `ipairs` loop it followed (the read
//! of the nil missed), so only a stepping stone's outer side could.
//!
//! Its guards start the call over, like a table read inside a call that
//! stored nothing (`Recorder::read_snapshot`): the step's callee guard, the
//! element reads and the end. A list of another length leaves there, and
//! the side that starts at the call records that length (the end's miss is
//! a branch, `HelperId::miss_takes_side`). An `ipairs` step also reads a
//! table the recording met by lane (`ipairs (lists[i])`, `ipairs
//! (t.pieces)`) through the lane, as `GetIndex` does
//! (`Recorder::lane_ipairs_read`); it aborted with `UnsupportedOperand`
//! before.
//!
//! A side that starts at such a call takes the arguments from its parent's
//! registers, which the root writes on every pass. It reads a table there by
//! lane (`SideSeed::lanes`, `native::seed_lane`) when the parent met it at
//! run time, as before, and now also when the argument is a table a body
//! operation made or the entry table of another register (a copy). Before,
//! the side bound the table and `plan_tree` refused it ("register r15 holds
//! a bound table and is written") when the list held strings, whose reads
//! have no lane form. A register the root does not write keeps its entry
//! binding, which is cheaper. I tried the same rule for a side of a side
//! (a register its parent read by lane): the nested `while` repro `n1.lua`
//! then grew a side that had been refused for a type conflict and went from
//! 0.834 G to 0.926 G instructions, so a side of a side binds as before.
//!
//! `for i = 1, 300000 do total = total + sum(lists[i % 4 + 1]) end` with
//! `sum` an `ipairs` loop over lists of two, one, three and no elements ran
//! all 300000 iterations in the interpreter before (889 M instructions,
//! Lua 5.4 profile); now 291522 of them run natively, in a root and three
//! sides, one per list length (128 M).
//!
//! A nest has no recording that links its outer loop to the inner root, as
//! `LuaJIT`'s does. The inner loop is the root, and the work around it joins its
//! tree as outer sides (see "Nested loops"), unless no tree of the inner loop
//! can carry the nest, when the outer root ends in front of it (see "Outer
//! roots"). A recording of the outer header is still tried, because the inner loop may sit on a rare path (the sieve's
//! `while` runs only for primes, and the outer recording that skips it is the
//! common case), and a recording that does meet the inner loop is cut short:
//!
//! - An `InnerLoop` abort at a backward branch that is not a numeric for
//!   latch or a `while` jump (a generic for) or at a numeric loop with too many
//!   trips to unroll rejects the header at the second abort at that
//!   instruction instead of the third: the retry that unrolls changes
//!   nothing there. When that inner loop already has an installed root of
//!   its own, or was rejected, the first abort rejects the header (`LuaJIT`'s
//!   recording ends at the link to the inner trace).
//! - A recording of the outer header that meets an inner loop inside a call
//!   it inlined, and cannot unroll it (a `while`, a generic for, a numeric loop
//!   with many trips), is not restarted for a later iteration any more and
//!   rejects the header at the first abort: whenever the iteration takes the
//!   call, the loop is met at it. Only a loop the next recording can unroll
//!   (a short numeric one) restarts.
//! - A side that reaches the `Return` of a root frame no loop encloses (a
//!   search loop that `return`s its element) ends with an exit in front of the
//!   `Return`, which resumes the interpreter there (the return link of
//!   LuaJIT): it runs natively up to the return, with the registers the
//!   return reads published by the snapshot, and the interpreter makes the
//!   return. A side of a frame whose caller could take the return over
//!   (`SideSeed::caller`, once the tree has run `RETURN_THROUGH_ENTRIES`
//!   entries) runs on in the caller instead, and one that waits for those
//!   entries waits; every other side takes the exit (`SideSeed::return_exit`).
//!
//! Two rules keep a nest's sides from being lost to the policy:
//!
//! - A side that starts at a call finds the callee in a register the root's
//!   recording would have loaded itself. When the last write of that register
//!   in front of the call, back to the loop's header, is a reload that gives
//!   the same closure every iteration (a `GetUpvalue` of a cell the loop
//!   does not write, or a `Move` of a register that is such a reload or is
//!   not written in the loop), the call is guarded on the closure's identity
//!   like the root's, not on its prototype (`Recorder::reloaded_invariant`).
//!   The prototype guard reads the callee's cells through the live closure,
//!   which holds scalars and tables only: a callee that reads a function
//!   upvalue could not be inlined along the other arm, and every side
//!   aborted at the callee's first `GetUpvalue`. The answer only picks the
//!   guard. A callee that is another closure fails the identity guard and
//!   leaves to the interpreter before the call. Only side recordings ask:
//!   a root recording runs the reload itself, and the rule changed which
//!   callees it pinned (a root of Mineclonia's `form_terrain` loop was
//!   rejected in some runs of an earlier version, and compiled in others
//!   with the rule off, so the root keeps the old answer). The reload must
//!   also be on every path: a forward branch in front of it that jumps past
//!   it leaves the carried value, and the callee then varies.
//! - The compile of queued sides is held for a guard that is about to ask for
//!   its own side (`InstalledTrace::holds_for_pending`), for at most
//!   `native::HELD_COMPILE_ENTRIES` entries. A root that runs few trips per
//!   entry refuses entries while a side is queued, so those entries never
//!   came and the hold never ended: the loop ran in the interpreter for good
//!   (447,052 refusals in a 3 s run). A refused entry counts toward the wait
//!   (`held_refused`). The count belongs to one hold: starting, ending or
//!   timing out a hold clears it (`SideBook::hold` and `release`), so the
//!   next hold waits its full length.
//!
//! An exit that rebuilds the frames of an inlined callee (`restore.inner` is
//! not empty) used to start no side, since a side started in the root frame.
//! Such exits show in `interpreter_exits` and `inner_exits`. In the iterator
//! tests they are well under one in twenty iterations, as the guards that
//! resume in the root frame carry the loop. Mineclonia's density loop
//! (`mg_ersatz.lua`, `for y = y2, y1, -1`) is the case where they do not:
//! `augmented_density` writes the upvalues `ssum` and `bsum`, and 14 inlined
//! `sample_density` calls write them again, so every guard after the first
//! write (the recording holds 220 snapshots with inlined frames out of 251)
//! leaves with `restore.inner` filled. Nearly every entry of the trace ends
//! at such a guard (16,603 of 18,008, all `GuardFail`/`Branch`), no side
//! ever starts, and the interpreter runs the rest of the iteration. Such
//! guards now take sides that start inside the calls (see "Sides from exits
//! inside inlined frames"). A recording of an outer loop
//! that inlines a call whose body has loops of its own aborts once with
//! `InnerLoop` at the first loop it meets in it, as the callee's loops are
//! not known before the recording runs. An inner loop of a nest of integers
//! that runs long per entry stays its own root by design (`outer_side_pays`).
//!
//! # Loops in calls
//!
//! A recording of an outer loop that inlines a call whose loop it cannot
//! unroll aborted with `InnerLoop`, and the header was rejected. Mineclonia's
//! `locate_biome_in_area` (biomes.lua:6980) is the case: each trip of its
//! `dx` loop calls the predicate `is_preferred_biome`, which scans a list of
//! biome names with `indexof` (structures.lua:30, `for i, v in ipairs
//! (list) do if v == val then return i end end`). The outer loop, the
//! predicate and the scan all ran in the interpreter.
//!
//! Now the first such abort notes the loop on the site (`Site::callee_loop`:
//! the callee's chunk and the latch), and the next recording
//! (`Recorder::with_callee_loop`) ends at that latch's first backward branch
//! (`Recorder::ends_before_callee_loop`). The end is a terminal exit whose
//! snapshot holds every frame (`Recorder::terminal`): it rebuilds the calls
//! and resumes at the loop's header in the callee, and the root frame stands
//! after the outermost call (`Recorder::root_resume`). The terminal exit
//! takes a loop side, as the one of an outer root does (see "Outer roots"),
//! seeded with the calls' frames (`SideSeed::frames`, `seed_frames_of` with
//! `looping`). The loop side loops in the innermost of them
//! (`Recorder::loop_depth`, `Recorder::in_loop_frame`): it closes at the
//! loop's header in that frame, its poll snapshot holds all the frames (the
//! verifier allows that for a side with `seed_frames` whose terminal frames
//! are the seed's: the same closures, call sites and windows), and the
//! registers of the frame that change in the loop are loop parameters whose
//! initial values are frame inputs (`Recorder::loop_frame_params`). Its exits
//! take seeded sides, as any guard in an inlined call does (see "Sides from
//! exits inside inlined frames"): they run the rest of the calls, return out of
//! them and link back to the outer header. A side that starts at the step
//! that ends an `ipairs` scan reads the nil as `ipairs` does
//! (`leaves_callee_loop` in `Recorder::record_ipairs_step`).
//!
//! Borrowed strings take three rules:
//!
//! - At the terminal exit, each borrow the frames hold (`val` of `indexof`,
//!   read out of an array by the caller) is given a lane
//!   (`Recorder::own_frame_borrows`, `HelperId::STRING_OWN`): a frame input
//!   cannot be a borrow. The element register of the scan is left alone.
//! - The loop side reads the element again at its header
//!   (`Recorder::reload_ipairs_element`), and its seed holds a stand-in for
//!   it (`SeedRegister::Reloaded`), so the element is neither an input nor a
//!   parameter. The same reload lets the root of an `ipairs` scan over
//!   strings compile: carried across the loop-back as a borrow, its element
//!   made the loop type-unstable before.
//! - A loop side reads a string out of an array slot of a table it met by
//!   lane as a borrow (`TraceOp::LaneArrayLoad` of a
//!   `TraceType::StringRef`), not through `TABLE_GET_STRING`, whose lane the
//!   side would have to keep for the root's loop to release (`finalize`
//!   refuses such a helper in a loop side).
//!
//! Two more things made the scan pay. The first build ran the biome loop
//! natively and cost more than the interpreter had: on the emerge the
//! luanti thread spent 1694 native samples in it next to 991 interpreter
//! ones, where the interpreter alone had taken some 1250.
//!
//! - `STRING_EQUAL` cloned both strings through `host::with_values` for
//!   every `v == val`, 44% of the instructions of a scan repro. It now
//!   compares the strings in place (`host::text`), pointer first.
//! - The side of the match (`v == val` holds) was never recorded: its seed
//!   frame held the borrowed `v`, which is no frame input, so `side_seed`
//!   failed and spent every attempt. Every match left for the interpreter,
//!   which ran the rest of the iteration and entered the tree again at the
//!   next backedge. A side that starts at a `return` of fixed results now
//!   holds nil in the registers it does not return (`SeedRegister::Dead`):
//!   nothing in the frame reads them again, and nothing runs there that
//!   could look at them. The chunk's liveness cannot say so, since it keeps
//!   every named local in scope alive for `debug.getlocal`.
//!
//! With both, the stronghold placement's biome search runs natively. Three
//! interleaved runs each, against the branch without this work: Mineclonia
//! startup 61.32G to 55.06G instructions:u (-10.2%), the emerge 216.9G to
//! 215.1G (-0.8%, runs spread by ±3%), devtest unchanged. On the emerge's
//! luanti thread the loop was the nearest loop for 28.3% of interpreter
//! samples and is now for 0.3%, with 2310 entries where there had been
//! 415000. The `idx.lua` repro (`scan` in `trace_callee_loops.rs`, 20000
//! outer trips) went from 16.99G to 6.05G.
//!
//! A numeric `for` loop in a call does not always take a loop side yet: the
//! byte loop of `trace_call_shapes`' `DIGITS` aborts at its first `MultiplyK`
//! (`UnsupportedOperand`), and the root in front of it retires after a few
//! dozen entries.
//!
//! Not done: a loop in a call that holds loops of its own, like the three
//! `while` loops of `region_evaluate` (`mcl_util/shape.lua:434`). The loop
//! side meets the inner loop in its own frame and aborts. The callee rule
//! in `abort_recording_at` leaves such a nest alone on purpose: each level
//! of `region_evaluate` runs about two trips, and the roots, outer roots
//! and unrolls tried before (see "Linking through a return") cost more
//! than they saved. It would take loop sides that nest inside frames, the
//! outer-root machinery carried into inlined calls. The z loop at
//! shape.lua:469 keeps its `LoopNotClosed` retirement for that reason.
//!
//! # Mineclonia's structure loops stay interpreted
//!
//! Two loops of `mcl_levelgen/structures.lua` run in the interpreter on the
//! emerge thread, and the trees above cannot carry either one. This section
//! records why, and what I tried.
//! Loop@744 (`get_structure_starts`) has since compiled with sides, but its
//! trees are evicted for room or retire on the emerge, and random.lua
//! loop@1159 runs natively: see "Calls that take turns between closures"
//! below for what changed and why loop@744 still runs interpreted.
//!
//! `finish_structures` (line 1014) calls `place_structures_in_chunk` from
//! its `dz` loop (line 1030). That call holds two nested `ipairs` loops:
//! `for j, sid in ipairs (structures)` (line 988) and `for _, start in
//! ipairs (starts)` (line 990). The `dz` recording inlines the call and
//! aborts with `InnerLoop` at the latch of the `j` loop. The retry that
//! unrolls it runs past [`recorder::MAX_UNROLLED_GUEST_INSTRUCTIONS`] in
//! the RNG seeding of `random.lua` (`LimitExceeded`), and three `InnerLoop`
//! aborts reject the site. The `j` loop as a root of its own aborts with
//! `TypeUnstable`: `sid` is a string, and the step reads it as a borrowed
//! `StringRef`, which cannot cross the loop back edge. When the step reads
//! it through a lane instead (`Recorder::lane_read`), the `j` loop compiles
//! but retires short, after 282 iterations in 100 entries: the generation
//! steps hold 2, 3, 8 and 25 structures. Nothing in its own chunk encloses
//! it, so it gets no stepping stone, and 25 trips are past the unroll
//! budget. A tree that carries this nest needs a link from the `dz` trace
//! into the `j` trace at the call. The tree has no such link: a nest links
//! only through outer sides and stones, and "Linking through a return"
//! needs 64 entries of a callee root.
//!
//! On the emerge (2026-11-01) the `dz` root at pc 55 restarts with
//! `InnerLoop` at the latch of the `j` loop (structures.lua:987), its
//! unrolled retry stops with `LimitExceeded` in the seeding at :988, and the
//! header is rejected after three `InnerLoop` aborts (later also three
//! `TypeUnstable`). The repro (`finish 48`) runs 9.20 G instructions in the
//! interpreter and 9.38 G with tracing: nothing in this nest pays yet, and
//! the `ipairs` loop over the starts (line 991) gives up as a stepping stone
//! for the same cell alias refusal as loop@744.
//!
//! `get_structure_starts` (line 742) loops over the structure sets
//! (line 744) and calls `set.placement.chunk_test`, a different closure for
//! each set. Its root holds 1782 guest instructions and leaves at the
//! callee guard of that call (33 exits in 33 entries). It retires after 32
//! entries, before any one guard reaches the side threshold. A side at that
//! guard was refused anyway ("a cell alias stands for a name with no window
//! slot"). I tried guarding the prototype instead of the closure and
//! reading the closure's upvalue cells at run time (a `CELL_GET_CALLABLE`
//! helper, closure rows held only for written cells). The repro
//! (`structures-calls.lua starts`) took 0.89 to 0.92 s before and 0.91 to
//! 0.96 s after, so I dropped it. Loop@744 first needs a root that lives
//! long enough to grow sides at its callee guard.
//!
//! A second round (2026-11-01) logged every exit of this root on the
//! 512-block emerge (a temporary print, not in the tree). Of 1757 exits,
//! 796 left at two branch guards the optimizer hoisted into the preheader
//! (origins 18 and 22, the sign tests of the ULL helpers on `cx` and `cz`),
//! before any iteration and with no side key: `cx` and `cz` are arguments
//! of `get_structure_starts`, so they are invariant in the loop and differ
//! between its runs. The rest left at the callee guard of `chunk_test`,
//! one closure per structure set. The second root grew a chain of four
//! sides, one per closure, and retired after 1764 iterations in 1724
//! entries. The first had its side refused for the cell alias above.
//!
//! Four changes went against these, in `structures-calls-round3.patch`
//! next to the repro: a root whose entries leave at a hoisted guard 16
//! times compiles again with its guards kept in the loop; a callee guard
//! inside an inlined call that committed nothing leaves through the
//! snapshot before the outermost call, as a branch there does, so its side
//! starts in the root frame and does not spend the tree's budget for
//! seeded sides (`policy::MAX_SEEDED_GUEST_INSTRUCTIONS`, which eight
//! sides of 1100 to 1800 guest instructions use up); a side that reaches
//! the `ipairs` step ending the root's loop ends there with an exit (the
//! side of the last structure set aborted `TableMiss` at that step before);
//! and the prototype guard of the first attempt, with the cell-alias check
//! made against the grown tree's window. On the repro (`starts 48`)
//! the base runs 15.9 to 16.6 G instructions (interpreter 14.6 G); the 16
//! combinations of the four ran 17.4 to 28.5 G, and 48 to 51 G in other
//! runs, as compile timing decides which sides land. None beat the base.
//! With `OXIMOND_JIT_SYNC=1` the base runs 22.4 G and the patched builds
//! ran past 120 s: once the cell alias no longer refuses the sides, a tree
//! of the root and eight per-closure sides charges 15 to 17 MB of metadata
//! (about 1.8 MB per side, mostly snapshots of six inlined frames), more
//! than the cache's whole `cache::MAX_TRACE_META_BYTES` of 16 MB. The
//! install is refused for room, which spends none of the side's attempts,
//! so the guard records the same side again and the tree compiles again
//! without end (145 recordings of one side in one run). The prototype
//! guard ran 55235 of the repro's 138240 iterations natively, in 13907
//! entries, and still cost more than the base. So sides per closure do not
//! fit this call site: fifteen closures, each side the size of the root.
//! What it needs is the prototype guard with entries that cost less than
//! four iterations of this body, a bound on retries of a side refused for
//! room, and a metadata charge that does not keep a snapshot of every
//! inlined frame per guard.
//!
//! I also tried a set of rules for `ipairs` loops in the root frame: an
//! inner loop's end read through `TABLE_ABSENT_INDEX` as in a call, the
//! recorded loop's own end reported as `LoopNotClosed` so the policy sees a
//! short loop, the lane read of a string element, the unroll budget counted
//! from the first unrolled trip, and two policy rules that reopen and
//! shorten such loops. A loop of 50000 trips around an `ipairs` over four
//! numbers then ran 45212 of them natively in Lua 5.4, against none before.
//! The Mineclonia emerge workload got slower: 282.5 G instructions on
//! average over 9 runs without the rules, 284 to 300 G with them (median
//! about 286 G), and the run to run spread grew. Each rule alone stayed
//! within the noise, about 2 G. The string lane read cost the most: it
//! lets the `j` loop compile, and that trace retires short. Neither target
//! loop ran natively with the rules, so I kept none of them.
//!
//! # Calls that take turns between closures
//!
//! A call site whose callee is another closure of one prototype on each
//! run (`fork:next_integer` in random.lua loop@1159, a new closure per
//! fork; `placement.chunk_test` under structures.lua loop@744, one closure
//! per structure set) keeps the identity guard in every recording. It is
//! guarded on the prototype, with the callee's cells read through the live
//! closure (an `InlinedFrame::dynamic` frame), only in a recording that
//! names it in `policy::Site::poly_calls`. A site gets there two ways. A
//! root of its loop retires, and its identity guards of that call left
//! `policy::POLY_CALL_EXITS` times or more in the window, counting the root
//! and its chain of per-callee sides together, with a guard that a side
//! continues counted once (`InstalledTrace::callee_exit_paths`). Or a side
//! is asked for at an identity guard of that call while
//! `policy::POLY_CALL_SIDES` sides of the tree already continue such guards
//! of it (`InstalledTrace::poly_side_path`): the third closure at the site.
//! The policy then removes the tree without spending a recording attempt,
//! names the site, and the loop records again once hot
//! (`TracePolicy::rerecord_for_poly_call`). The second way exists because a
//! tree whose sides continue its callee guards never retires on them: on
//! the emerge, loop@744's tree grew a side per `chunk_test` closure until
//! another install evicted it for room (see below). The site
//! is named by the call pcs of the inlined frames down to it, outermost
//! first (`policy::PolyCallPath`), and only the recording whose frames
//! match the path guards it on the prototype (`Recorder::poly_call`). The
//! field read that loads such a callee goes through the lane instead of the
//! entry binding of the table's field (`Recorder::reads_poly_callee`), so
//! the binding does not pin the first closure either.
//!
//! The rules that make such a frame carry the code it calls:
//!
//! - A cell of a dynamic frame that holds a string is read with
//!   `CELL_GET_STRING` into a lane of the run's bank, as a table read of a
//!   string field is (`TABLE_GET_STRING`): the bank roots the string for
//!   the rest of the run, and the read misses for another type or a full
//!   bank. The `id` of the concentric ring `chunk_test`
//!   (structures.lua:404) is such a cell, a different string per closure.
//! - A cell of a dynamic frame that holds a function is read with
//!   `CELL_GET_CALLABLE`: a closure through the bank's lanes, the native the
//!   recording saw by its pinned lane, anything else a miss. The call of the
//!   value is guarded on its identity when the cell is shared by every
//!   closure of the reader (an upvalue of an upvalue, `bor` and `floor` of
//!   the ULL helpers), and on its prototype when each closure of the reader
//!   captured the cell anew (`UpvalueSource::ParentRegister`, the `fn` of
//!   each fork; `Recorder::instance_callees`).
//! - A closure row holds only the cells the trace writes
//!   (`written_cell_slots`); a window slot the trace only reads holds the
//!   cell's value for the whole run, as the row does.
//! - A callee whose cell is also a window variable of the recording (a
//!   window slot or a captured local of the root frame) keeps the identity
//!   guard when either side writes that cell (the window, or a `SetUpvalue`
//!   in the callee's chunk), or when the callee indexes it with `GetTabUp`
//!   or `SetTabUp` (`Recorder::shares_window_cell`). A cell that only gets
//!   read, such as the ULL helpers every `next_integer` closure shares, no
//!   longer keeps the call off the prototype guard.
//! - The captured-locals refusal looks at writes: a trace with dynamic
//!   frames is refused only if it writes a captured local of the root frame
//!   that a closure could read, or writes a cell through a dynamic frame
//!   while it holds a captured local in a register
//!   (`SideSeed::dynamic_cell_writes`, `SideSeed::captured_written`). A
//!   frame that only reads the shared helpers no longer refuses the
//!   recording.
//! - A side can start inside a dynamic frame: the exit pushes the frame
//!   with its live closure, and the register after the chunk's holds that
//!   closure as a frame input of the side (`seed_frames_of`,
//!   `Recorder::start_seed_frames`). The frame's pinned lane still names the
//!   prototype.
//! - A cell alias may stand for a window slot that a side added to the
//!   tree, not only for one of the root's (`verify_cell_aliases`). This is
//!   the "a cell alias stands for a name with no window slot" refusal of
//!   loop@744's sides.
//!
//! Two policy rules came with it:
//!
//! - A root whose entries left `policy::HOISTED_GUARD_EXITS` times or more
//!   at branch guards the optimizer hoisted in front of the loop (exits
//!   with no side key whose snapshot is the loop's entry state) retires
//!   with `Site::keep_guards` set, and the next recording keeps its branch
//!   guards in the loop (`TraceBuilder::set_keep_branch_guards`). These are
//!   the two guards on `cx` and `cz` that took 796 of 1757 exits of
//!   loop@744 on the emerge: arguments of `get_structure_starts`, constant
//!   in a run of the loop and different between runs.
//! - A side whose tree install is refused for room spends one of its
//!   `MAX_SIDE_ATTEMPTS` (`end_side_compile_refused`). It spent none before,
//!   and the same side was recorded and compiled again 145 times in one run
//!   of the structures repro (1785 recordings in
//!   `a_side_whose_tree_is_refused_for_room_is_recorded_a_bounded_number_of_times`).
//! - A recording that aborts inside a frame it guarded on the prototype
//!   clears the loop's `Site::poly_calls` and sets `Site::poly_refused`: the
//!   loop keeps identity guards from then on. A cell holding a coroutine or
//!   a userdata still aborts so; a string no longer does. In Lua 5.2 and
//!   later a global read in such a frame (`GetTabUp` of `_ENV`) aborts too.
//!
//! `a_loop_that_leaves_at_hoisted_guards_compiles_again_with_them_in_the_loop`
//! runs the shape of loop@744's `cx`/`cz` tests: with the rule the second
//! root and its sides run more than 80000 of 96000 iterations; with
//! `HOISTED_GUARD_EXITS` out of reach it compiles no side and the test
//! fails. The trace log now names evictions ("evicted the trace at pc ...
//! for room", with the metadata parts and the trace's runs), sides held
//! back while their tree's code runs, sides and trees refused for room, and
//! which part of a side's window differs from its root's.
//!
//! Why this does not pay the costs of the dropped attempts above: those
//! guarded every call read from a table or cell on its prototype once the
//! prototype had made a few closures (`structures-744-partB.patch` doubled
//! the cost of a plain loop that way), or grew a side per closure (the
//! round3 tree of eight per-closure sides outgrew the 16 MB metadata cap).
//! Here a monomorphic call site keeps its identity guard and its pinned
//! callee, and pays nothing: the prototype guard and the reads through the
//! closure appear only after a root retired on that site's callee exits.
//! One prototype-guarded root then carries every closure, so the tree does
//! not grow with the number of closures, and its sides start where the
//! closures' code diverges.
//!
//! Measured on 2026-11-02 against 3724ce7d (pinned `instructions:u`,
//! `--compat=luajit21`, load 3.4 to 10). The xoroshiro repro
//! (`target/flow-luanti-native/repros/poly-callees.lua forks 100000`, the
//! verbatim self-test) went from 5736 G to 934 G, and its root no longer
//! retires. `structures-744.lua` went from 22.37 G to 18.88 G and
//! `poly-callees.lua starts 48` from 22.38 G to 18.89 G: kept guards let
//! the root of loop@744 live, and the alias rule lets its per-closure sides
//! join. n1, c3, c4 and density stayed within 0.1%. shape-repro went
//! from 4.411 G to 4.418 G (+0.17%) for another reason: its loop at
//! shape.lua:469 left at hoisted guards and at a call that takes turns, so
//! both rules fired, and it now runs 74720 iterations natively against 95,
//! at about the interpreter's cost. On the
//! 512-block emerge, random.lua loop@1159 now runs natively (15338
//! iterations in 47 entries per run of the self-test, against a retire
//! after 17 entries).
//!
//! Before the second round below, loop@744 did not run natively on the
//! emerge, though both rules fired there. The attribution's "retired (`TableMiss` at
//! structures.lua:744)" is the last abort of a site whose trees came and
//! went: the `TableMiss` is a recording that aborts at the `ipairs` call on
//! line 743 (`ForInCall`) or at a `GetField` on line 744, and it is not
//! what removes the trees. Seven traced
//! runs of the emerge (`OXIMOND_JIT_TRACE`, 2026-11-02) show two fates:
//!
//! - The first root leaves at the hoisted `cx`/`cz` guard 24 to 32 times
//!   in its window ("exit 6 (member 0, guard Some(22) Some(Branch), frame 1
//!   pc 14) 32x") and retires with "left at hoisted guards". The next root
//!   keeps the guards in the loop; none of its exits after that is at the
//!   loop's entry state. In four runs its tree then grows a side per
//!   `chunk_test` closure (five to seven sides of 1000 to 3400 guest
//!   instructions each) until it holds most of the 16 MB of metadata, and
//!   another trace's install evicts it: "evicted the trace at pc 14 ...
//!   for room" at 15.0 MB and again at 9.4 MB in one run, and at 16.3 MB
//!   in another (4.9 MB of IR, 3.8 MB of lowered IR and 7.6 MB of restore
//!   plans over 8 members). Each time the site goes back to counting,
//!   records again, aborts at the `TableMiss`, and compiles once more. The
//!   log lines are in `target/flow-luanti-native/poly-callees-fix-744-log.txt`.
//! - In three runs the second root retired on its callee guards first and
//!   named the call (`[23, 9]`, `chunk_test` in
//!   `structure_starts_in_chunk_p`). One prototype-guarded recording
//!   aborted at the `GetUpvalue` of `id` in the concentric ring
//!   `chunk_test` (structures.lua:404, a string cell, which a dynamic frame
//!   cannot read), and the header was then rejected after three such
//!   aborts. A recording that aborts inside a dynamic frame now clears the
//!   site's `poly_calls` and sets `Site::poly_refused`, so the loop goes
//!   back to identity guards and never names a call again
//!   (`a_prototype_guarded_call_that_reads_a_coroutine_cell_goes_back_to_identity_guards`;
//!   a string cell is read now, see the second round below).
//!   Two prototype-guarded roots compiled and retired within a window:
//!   966 iterations in 787 entries with 30 exits at a branch guard five
//!   frames deep, and 379 in 260 with a side refused as "another window
//!   than its root" (its pins were the root's 15 in another order). The
//!   site had used its attempts by then.
//!
//! So the cost that keeps loop@744 interpreted is now the size of a tree
//! that carries the structure sets' code once per closure, against the
//! cache's metadata cap, and a prototype-guarded root that leaves at the
//! sets' own branches before sides for them land. The `=(load)` rows that
//! fell under loop@744 in the function table moved, in part, to loops 1030
//! and 988: a function row names the last loop that called it, and every
//! `=(load):N` of every loaded chunk shares one row. The `near%` column of
//! the attribution's per-loop table counts each sample once, for the
//! nearest loop on its stack, and is the number to compare.
//!
//! Measured on 2026-11-02, 3724ce7d against this rule set, one attribution
//! run each (`luanti.py bench --attribution`, load 4.4 and 4.4) and two
//! counted emerges each (load 4.2 to 7.5). On Emerge-0, loop@744's
//! `near%` went from 3.9% of 3613 interpreter samples to 5.0% of 3146,
//! with 7 and 53 native samples: it runs natively for a while and then
//! loses its tree. The xoroshiro loops are where the interpreter samples
//! went: random.lua loop@1159 from 8.9% (retired, 2 native) to 1.6%
//! (compiled, 63 native), and loop@1386, the same self-test's second
//! loop, from 8.2% (retired) to 1.0% (compiled, 47 native). Emerge-0 had
//! 983 native samples before and 1149 after. The emerge went from
//! 283.5/281.4 G to 274.8/277.5 G (268.2/274.7 G in an earlier pair of
//! the same code without the attribution field).
//!
//! The repo's tracing bench suite (`benchmarks/icount.sh -r tracing`,
//! pinned to CPU 4 under `heavy.sh -x`, load 3.4 to 4.0, 89 programs)
//! moved within 0.9% except three: `opt/closure_protos` 744.3 M to 435.9 M
//! (one prototype through many closures, this section's case),
//! `string/naive_concat` 292.6 M to 286.6 M and `allocation/gc_cycles`
//! 300.4 M to 307.8 M. The last two run the same single trace on both
//! builds and differ in its entries: `gc_cycles` takes 77 GC-poll exits
//! where it took 58 (3150 fewer native iterations), `naive_concat` enters
//! 30 times where it entered 23. I read that as the collector's pacing,
//! which the CLI's byte-counting allocator ties to every allocation, the
//! installed trace's second exit-count vector among them; I did not
//! confirm it further.
//! Every count repeated to the 0.1 M over three runs.
//!
//! ## Second round: loop@744's prototype-guarded root
//!
//! Traced emerges after the first round showed the prototype-guarded root
//! of loop@744 aborting or retiring for five reasons, each fixed here:
//!
//! - A tree whose sides continued the callee guards of `chunk_test` never
//!   named the call: the `POLY_CALL_SIDES` rule above. It fires only at an
//!   identity guard (`guards_on_prototype` skips prototype guards) and only
//!   when the live callee shares the pinned closure's prototype
//!   (`InstalledTrace::callee_shares_prototype`). Different prototypes at
//!   one site keep their per-callee sides, as the tests in
//!   `trace_divergent_calls` and `trace_closure_callees` expect.
//! - The callee of a `CallOne` is register B; A is its destination.
//!   `Recorder::reads_poly_callee` compared A, so it never matched a
//!   `CallOne`. Both it and the new check go through `callee_register`.
//! - The concentric ring's `id` cell: `CELL_GET_STRING`, above.
//! - The ring's `chunk_test` reads globals. A dynamic frame guards its
//!   closure's environment against the pinned closure's once per frame
//!   (`CLOSURE_SAME_ENV`, `Recorder::guard_dynamic_environment`) and may
//!   then read a global slot. A closure that `setfenv` changed leaves at
//!   that guard. The verifier's "a global slot in a trace with effects"
//!   rule no longer counts cell reads and writes as effects
//!   (`HelperId::is_cell_access`).
//! - `is_start[hash]` (structures.lua:406) reads a key that is absent from
//!   a lane table with no metatable. The recording emits
//!   `TABLE_ABSENT_INDEX`, which leaves when the key turns up or the table
//!   gains a metatable, and continues with nil instead of aborting with
//!   `TableMiss`.
//!
//! With those, the prototype-guarded root records the concentric side and
//! runs, but loop@744 averages 1.2 loop-backs per entry on the emerge
//! (641 in 552 entries), one or two starts per chunk. The productivity bar
//! of eight retired it. `native::min_iterations_of` now counts each inlined
//! call as `INLINED_CALL_WEIGHT` operations, and a body of twice
//! `TRACE_ENTRY_WORK` or more is held to `TRACE_MIN_ITERATIONS_HEAVY`, one
//! loop-back per entry. In the traced emerge after that change the root ran
//! 10919 entries with three sides and no retirement. Then another install
//! evicted it for room at 5.3 MB of metadata (1.7 MB IR, 1.2 MB lowered,
//! 2.4 MB restore plans over 4 members), against 16.3 MB for the
//! identity-guarded tree of round 3. The log lines are in
//! `target/flow-luanti-native/poly-callees-fix2-744-log.txt`.
//!
//! Two gaps remain. Sides whose recordings overlapped add their pins at the
//! same lane indexes, and the tree refuses the later ones as "another
//! window than its root: pins" (31 against 28). And the root still uses a
//! third of the metadata cap, so a busy emerge evicts it, and the
//! recordings after that can abort at the `ipairs` call on line 743
//! (`TableMiss`) until the site is rejected.
//!
//! Measured on 2026-11-02 against 83e689ab, the first round (pinned
//! `instructions:u` on CPU 4 under `heavy.sh -x`, `--compat=luajit21`,
//! load 5.9 to 6.6, two runs each that agreed to 0.03%):
//! `structures-744.lua` went from 18.88 G to 15.45 G and
//! `poly-callees.lua starts 48` from 18.58 G to 15.93 G. n1, shape-repro,
//! c3 and c4 stayed within 0.05%. density-repro went from 2006.3 M to
//! 2019.8 M (+0.67%); I did not trace which rule costs it. On Emerge-0 of
//! one attribution run (load 11.4), loop@744's `near%` went from 5.0% of
//! 3146 interpreter samples to 4.9% of 2877, with 53 and 59 native
//! samples, and Emerge-0 had 1149 native samples before and 1257 after.
//! random.lua loop@1159 went from 1.6% to 0.6%. Two counted emerges gave
//! 274.3 G and 275.7 G (load 12.0 at the start, 4.3 at the end), against
//! 274.8 G and 277.5 G for 83e689ab: within the noise of the pair.
//!
//! ## The heavy bar and the credited loop-back
//!
//! The productivity window credits every run one loop-back, also a run
//! that left before its first (`credited` in `TracePolicy::run_pinned`).
//! Against a bar of eight or three that credit hardly matters. Against
//! `TRACE_MIN_ITERATIONS_HEAVY`, one per entry, it passed every window, so a
//! heavy trace stayed installed whatever its runs did. That was the
//! regression of `shape-469.lua` from 16.91 G to 20.01 G
//! (`OXIMOND_JIT_SYNC=1`) at the commit above. Bisected with a toggle per
//! rule in a scratch copy, it came from this bar alone. The prototype guard
//! after a retire, `POLY_CALL_SIDES`, the kept hoisted guards, the
//! `callee_register` fix and the count of inlined calls each changed the
//! repro by less than 0.4% when switched off on their own. The trace that
//! stayed was the repro's driver loop (`for it = 1, 40000`, `equal_p` and
//! the sturdy-face checks inlined, the region nest's short loops unrolled):
//! it left at a branch guard four frames deep in each of 38000 entries, and
//! the interpreter rebuilt those frames and finished the calls each time.
//!
//! A window now keeps a second sum, `window_loop_backs`, that a heavy body's
//! bar is judged by: the loop-backs a run did, or the credited one when the
//! run left in the loop's own frame, and nothing for a run that left inside
//! an inlined call before its first loop-back. Any other bar keeps the
//! credited count. `structures-744.lua`, the shape the heavy bar was made
//! for, stays at 15.45 G.
//!
//! Measured on 2026-11-02 (pinned `instructions:u` on CPU 4 under
//! `heavy.sh -x`, `--compat=luajit21`, `OXIMOND_JIT_SYNC=1`, binaries at
//! paths of equal length, load 7.2 to 8.1, two runs each): `shape-469.lua`
//! 20.01 G to 16.99 G; n1, shape-repro, c3, c4, density and
//! `structures-744.lua` within 0.02%.
//!
//! On the 512-block emerge this rule does not touch loop@469: none of its
//! roots is held to the heavy bar. `retire` logs a trace's bar now, and in
//! two traced emerges every one of the 21 roots of loop@469 that retired
//! had a bar of three (`TRACE_MIN_ITERATIONS_FLOOR`). The native samples at
//! loop@469 swing from run to run: one traced run of this commit retired
//! roots that had run 311610 iterations in 33186 entries and 137815 in
//! 10769, the next only roots of at most 324 iterations, and with the old
//! credit switched back on a traced run ran 690 native iterations there
//! against 16600 in the run of this commit beside it.
//!
//! What retires those roots is the entry types (the causes under "Not
//! done" above). `entry_mismatched` logs it now ("N entries matched none of
//! its header's versions"): `lx_dom`, `rx_dom`, `ly_dom` and `ry_dom` are nil
//! or a number, up to nine combinations against four versions. A burst of
//! [`native`]'s `TRACE_IDLE_MISMATCH_LIMIT` entries in a row that no
//! version takes retires the newest version, and the next ones follow,
//! since a version retired beside others closes the header to new
//! versions (`Site::versions_closed`, until the last of them retires; see
//! "Mineshaft corridors"). Its recordings also end with
//! `LoopNotClosed` when `op` returns true from inside the loop (line 482)
//! or on the last of its two to five trips, and the `y` loop around it
//! (line 459) is rejected for `InnerLoop` at line 497. The attribution's
//! "Divergence at shape.lua:437" of one run is the last abort of the site.
//!
//! Three policy changes against the version churn were measured on the
//! emerge (2026-11-02, `instructions:u`, load 5 to 14) and are not in the
//! tree. Three emerges of this commit gave 269.5, 271.7 and 268.4 G; with
//! the old credit, 330.5 and 277.1 G. One emerge in four or five lands
//! near 330 G with or without the changes below, so their counts say
//! little; the attributions say more.
//!
//! - `MAX_TRACE_VERSIONS` of 12: 268.9, 273.9 and 275.1 G; loop@469 still
//!   retired, at 22.6% `near%` with 61 native samples.
//! - An entry that runs any version clears the newest version's count of
//!   idle mismatches: 330.8, 270.6, 280.4 and 326.4 G. Loop@469 got 451
//!   native samples in one attribution run, and the trace of
//!   `mineshaft.lua:299` stayed installed and took 3284 native samples
//!   (168 to 224 in the runs without these changes).
//! - A version with productivity credit spends a window's bar of it on a
//!   burst of mismatches instead of retiring: 323.9, 312.0 and 312.6 G,
//!   loop@469 at 20.3% with 1 native sample, `mineshaft.lua:299` at 2894
//!   native samples.
//!
//! Keeping versions alive can keep that mineshaft trace installed too (in
//! the second attribution run with the idle change it took 191 samples),
//! and neither change made loop@469 native. Loop@469 needs a register that
//! holds nil or a number at entry: an entry input and a snapshot slot of
//! either type, with the type test as a guard in the loop body that a side
//! can continue, and outer sides that link back with the other type. That
//! is the untyped entry input under "Linking through a return" below (ported
//! from `flow/ull-shape`), lowered by Cranelift only.
//!
//! For the MIR backend (left to the session that owns `jit/trace/mir/`):
//! `mir::lower` refuses a root with an entry input of type `Any` ("entry
//! input of any type") and a `GuardType` on an untyped value, so every
//! tree that takes such a register goes to Cranelift whole. To take these
//! trees MIR needs what `emit.rs` does: skip the tag check of an `Any`
//! input at entry (`Lowerer::lower_entry`), lower its `GuardType` as a load
//! of the input slot's tag and bits with a guard exit
//! (`Lowerer::guard_any_input`), read the root's slot from a side
//! (`any_slots`), store a link's value, tag and bits, into the root's slot
//! and enter at the header (`Lowerer::store_untyped_link`), and leave the
//! slots in the exit area on every run (`CompiledTrace::untyped_writebacks`).
//!
//! Emerge, 2026-11-02, one build of each (ef374807's `native.rs` against
//! this one, same tree otherwise): two counted emerges gave 286.6 G and
//! 270.5 G before (load 14.1 to 11.1) and 269.9 G and 267.1 G after (load
//! 10.8 to 9.4), within the run to run spread. On Emerge-0 of one
//! attribution run each, loop@469's `near%` was 17.6% (131 native samples)
//! before and 18.3% (0 native) after, rejected and then retired with
//! `LoopNotClosed` both times; as above, that difference is the spread
//! between runs, not the bar.
//!
//! # Mineshaft corridors
//!
//! Three loops of Mineclonia's `mcl_levelgen/mineshaft.lua` ran interpreted
//! on the emerge thread: loop@299 (`fill_box_rotated`, an `ipos3` loop over
//! the cells of a box), loop@1018 (the sections of `corridor_place`) and
//! loop@1045 (its floor patch, another `ipos3` loop). The repros under
//! `target/flow-luanti-native/repros/` (`mineshaft-299.lua`,
//! `mineshaft-1018.lua`, `mineshaft-1045.lua`) run verbatim slices of
//! random.lua, util.lua, terrain.lua, structures.lua, nodeprops.lua and
//! mineshaft.lua lines 1-1072, loaded as `=mineshaft.lua` so the line
//! numbers match. Their driver places a few dozen corridors and staircases
//! per mapchunk. The Lua-level tests are in `tests/trace_mineshaft.rs`, the
//! recorder tests of table and function keys in
//! `vm/tests/trace_object_keys.rs`, and the hoisted-guard threshold has a
//! unit test beside it in native.rs (`leaves_often_at_hoisted_guards`).
//!
//! Three rules came out of the emerge trace log (2026-11-03):
//!
//! - loop@1045 calls `get_sturdy_faces`, whose cache `ordinal_sturdy[shape]`
//!   (nodeprops.lua:705) is keyed by the shape table. The recorder took only
//!   string and number keys and aborted with `UnsupportedOperand` (10
//!   aborts there in one emerge). A table or function key
//!   (`TableKey::Object`) is now read through the lookup call of a table met
//!   by lane (`Recorder::lane_read`), which leaves before the instruction
//!   unless the key is present with a value of the recorded type, so a
//!   missing key never skips `__index`. A store under such a key still
//!   aborts.
//! - `fill_box_rotated` takes `chance`, 1.0 on one call and 0.8 on the next
//!   (mineshaft.lua:1007 and :1009). The guard `chance == 1.0` was hoisted
//!   out of the loop, and a hoisted guard takes no side, so every 0.8 call
//!   left at entry ("exit 0 ... frame 1 pc 68 8400x" of 13538 entries). A
//!   root that leaves at hoisted guards in a quarter of its entries or more
//!   compiles again with its branch guards in the loop
//!   (`InstalledTrace::keep_hoisted_guards`), a tree or side compile
//!   already in flight keeps that choice when it lands (`keep_guards_of`),
//!   and two productivity windows are forgiven while that code is on its
//!   way (`guard_move_grace`). A root that loops of its own frame enclose is
//!   left out: its exits at the entry state are its loop's ends, which take
//!   outer sides once it is a stepping stone (see "Short loops give way").
//!   Without that exception the repro driver's own `prepare_mapchunk` nest
//!   retired four times and the repro ran 18.3 G instructions against 7.2 G.
//! - Sides at the iterator's row advance aborted with `LoopNotClosed` when
//!   the row was the box's last, and spent all their attempts (see "Side
//!   traces").
//!
//! Those emerge numbers for loop@299 (near% 11.5% to 2.9% and 4.8%) came
//! from binaries that still had the 16 MB metadata cap. Three attribution
//! runs of the commit that raised it (2026-11-03, load 5 to 10) all ended
//! with loop@299 `rejected (Retired)` at 11.7%, 12.7% and 11.9% near% (354,
//! 388 and 415 native samples). The trace log of those runs shows why, and
//! it is not the cache:
//!
//! - Every retirement of the header is by idle mismatches ("the trace at pc
//!   68 retires: N entries matched none of its header's versions"), seven
//!   in a run, over four attempts.
//! - The entries turned away hold nil in register 4 where the version wants
//!   a table: `rng`, which staircases pass as nil (mineshaft.lua:664) and
//!   corridors as a table. The mismatch log now names the first input the
//!   entry check turned down, of any type ("first failed input (4, Table,
//!   0)"); it used to list nil and number registers only, so these lines
//!   named no register.
//! - In the first attempt the header gets a version for nil beside the one
//!   for the table. One of them retires beside the other, which closes the
//!   header to new versions (`Site::versions_closed`), and the rest follow.
//!   The flag outlived them: in attempts 1 to 3 the log shows "no other
//!   version for the header at pc 68 ...: 1 of 4 versions, closed true", so
//!   each attempt kept the one version it recorded, the other signature's
//!   entries retired it, and the fourth retirement rejected the site.
//!
//! `TracePolicy::retire` now opens the header again when the last version
//! retires: closing protects a sibling, and none is left. The unit test is
//! `the_last_version_retiring_opens_the_header_to_versions_again` in
//! `vm/tests/trace_cache.rs`; `tests/trace_mineshaft.rs` runs the corridor
//! and staircase shape in every profile.
//!
//! Measured on 2026-11-03 against the commit before (same tree with the
//! line switched off, binaries at paths of equal length). Emerge-0
//! attribution, loop@299: `rejected (Retired)` at 11.7%, 12.7% and 11.9%
//! near% before; `compiled-exited` in all three runs after, at 7.4% (642
//! native samples), 3.0% (258) and 10.7% (856), with three versions
//! installed in the traced run. In the 10.7% run 8357 of the trace's 8553
//! entries left at a branch guard into the interpreter (`GuardFail/Branch`),
//! so the native samples doubled while the interpreter share barely moved.
//! loop@1045 stays `retired` (2.7% to 3.3% before, 3.5% to 4.0% after) and
//! loop@1018 `rejected (InnerLoop)` at mineshaft.lua:842 (3.8% to 3.9%,
//! then 3.4% to 4.5%). Emerge, interleaved, one rep each (load 4.2 to 5.9): 279.1 and
//! 280.1 G before, 285.8 and 272.5 G after, inside the run to run spread.
//! The loop runs 1.5 to 3 iterations per entry, so its native samples cost
//! about what the interpreter samples they replace did. Pinned counts (CPU
//! 4, `heavy.sh -x`, two runs each): n1, shape-repro, c3, c4 and density
//! within 0.02%; `mineshaft-299.lua` 13.11 G to 13.13 G (its driver never
//! retires the header, so the rule does not fire there).
//!
//! Three more rules were tried and dropped, each measured on the emerge with
//! the others on and off (1 rep each, spread about 4%, and the base itself
//! lands at 320 to 337 G in a quarter of its runs):
//!
//! - Keeping a version through a burst of mismatches no longer than the
//!   iterations it had run: 319 and 322 G alone, against 266 to 282 G for
//!   the base. A version that paid once was never retired by mismatches
//!   again, and paid entry checks for no work.
//! - Letting the idle-mismatch rule wait while another version is asked
//!   for, recording or compiling: 334 and 325 G alone.
//! - Reading string fields of bound tables at run time once a version was
//!   turned away for one (`reorientate_coords` compares `self.dir` with four
//!   strings): 326 and 284 G alone, 295 G median with the kept rules against
//!   285 G without it.
//!
//! The emerge with the kept rules first ran 283, 292 and 287 G against 268,
//! 271 and 327 G for the base, interleaved. Its trees for mineshaft.lua:299,
//! :872 and :1057 hold 3.5 to 4.2 MB of metadata each, and with shape.lua:359
//! (6 MB) and structures.lua:744 (4.8 MB) the 16 MB cap evicted 96 trees for
//! room, against 44 before, among them `mg_ersatz.lua:119`'s tree after 1.98
//! million iterations; its inner loop@123 was then rejected and took 8.6% of
//! the interpreter samples. `cache::MAX_TRACE_META_BYTES` is now 48 MB: no
//! eviction in the emerge, and 281, 281 and 277 G against 321, 266 and 279 G
//! for the base, interleaved.
//!
//! loop@1045 still retires. Past nodeprops.lua:705 a recording aborts at
//! `get_node_shape` (nodeprops.lua:410, `UnsupportedOperand`) in some runs,
//! and the versions split on the rng (nil or a table) and on the string
//! constant `self.dir` that the entry checks.
//!
//! loop@1018 is not fixed. Each section inlines `generate_supports`, whose
//! ceiling test `for x = min_x, max_x` (mineshaft.lua:842) is an inner loop
//! with an early `return`. The recording that unrolls it passes
//! [`recorder::MAX_UNROLLED_GUEST_INSTRUCTIONS`] in the first section's
//! `get_block` calls (`LimitExceeded`), and the recordings after it abort
//! with `InnerLoop` there until the site is rejected. One section runs eight
//! `essay_cobweb` calls, each with a `next_float` of 64-bit LCG arithmetic,
//! so an unrolled iteration would be far past the cap, and the loop runs 2 to
//! 4 sections per entry, too few to repay a root. Its work belongs to a tree
//! rooted further out, in `place_structures_in_chunk` (see "Mineclonia's
//! structure loops stay interpreted").
//!
//! # Operands the emerge refused
//!
//! The Emerge-0 attribution of 2026-11-03 named four loops that aborted with
//! `UnsupportedOperand`. The trace log of the 512-block emerge
//! (`OXIMOND_JIT_TRACE`) gives the instruction of each. Two were recorder
//! rules, now general:
//!
//! - structures.lua loop@855 (`collect_structure_references`) aborted at the
//!   `GetTable` of `starts[hash]` (:856) three times and was rejected. The
//!   start cache is keyed by `struct_hash`, `bor (lshift (x + 4096, 13), z +
//!   4096)`, a sparse number in the hash part, and its values are records.
//!   `Recorder::number_hash_read` took only a number or a boolean there. A
//!   table, a Lua closure, a string or a userdata under such a key of a
//!   bound table is now read through the lookup call (`Recorder::lane_read`,
//!   `rawget`'s read), which leaves before the instruction unless the key
//!   holds a value of the recorded type, so a key gone missing never skips
//!   `__index`. A function trace still refuses (one table could have two
//!   lanes there). `beardify_1`'s loop@1327 reads the same cache and aborted at
//!   :1328 the same way.
//! - The record's field `.starts` is then a row read of a table the lookup
//!   gave, and such rows were never filled outside trees with lane array
//!   views (see the lookup-call rows under "Tables"): every run left at the
//!   field, one iteration per entry. Cranelift now fills the row after the
//!   call, and the MIR backend makes the same fill (the `fills_table`
//!   condition in `mir/lower/calls.rs`), so these trees compile under MIR.
//! - The inner `for _, start in ipairs (local_starts)` (:857) then aborted
//!   the `z` loop with `TableMiss`: most chunks hold no start, the step at
//!   index 1 reads nil, and a recording that does not unroll could not
//!   follow the end of an `ipairs` loop. The first step of an `ipairs` loop
//!   other than the recording's own (its latch jumps to a body that is not
//!   the header) that ends the loop runs no trip, so the recording follows
//!   it as it follows an `if`, behind the `TABLE_ABSENT_INDEX` check that
//!   the key is still absent from a table without a metatable
//!   (`Recorder::ends_inner_generic_for`). A list that holds starts leaves at
//!   that check; a step of a loop that has run a trip still meets the inner
//!   back-edge (`InnerLoop`) and the unrolling retry as before.
//! - structures.lua loop@1263 (`apply_terrain_adaptation`) aborted at the
//!   `Call` of `mathmax (0, piece_bbox[1] - x, x - piece_bbox[4])` (:1271):
//!   the native numeric call took exactly its arity. `math.max` and
//!   `math.min` of more than two numbers now fold the two-argument form from
//!   the left (see "Library calls").
//!
//! The mineshaft unit left loop@1045 retired on `UnsupportedOperand` at
//! nodeprops.lua:410, `static_node_shapes[cid]` in `get_node_shape`, and
//! `is_cid_solid[cid]` (:1111) refused its sides the same way: `get_block`
//! gives no content id outside the chunk, and `t[nil]` had no key form.
//! `t[nil]` of a table without a metatable is nil, so a root trace reads it
//! as the constant nil behind one `TABLE_ABSENT_INDEX` call with the nil key
//! (`Recorder::read_nil_key`), which leaves if the table has gained a
//! metatable since. The key's type is the trace's own; a key that turns into
//! a number leaves at the guard that typed it. A bound table takes the call
//! too: a binding alone names no entry input, which the verifier refuses.
//!
//! Two stay interpreted, and no recorder rule would change that:
//!
//! - biomes.lua loop@7012 (`get_biomes_chebyshev`) aborts at the
//!   `GetUpvalue` of `terrain_persist` at pc 0 of `mcl_mapgen_models/v7.lua`'s
//!   `base_terrain_level` (:107), a value-noise userdata. Every iteration
//!   calls `preset:index_biomes_cached`, whose ersatz form calls
//!   `mapgen_model.get_biome_override` (three `get_2d` noise methods and two
//!   `core.get_value_noise`) and `core.get_biome_data`: C API functions of
//!   Luanti. Reading the userdata alone would only move the abort to the
//!   next instruction; a trace can now make the C API calls itself (see
//!   "Calling host functions").
//! - templates.lua loop@718 (`register_template_pool`) aborts at the `Call`
//!   of `io.open` in `read_structure_template` (:83). Each iteration opens
//!   and reads a file, decompresses it (`core.decompress`, C API) and calls
//!   `core.deserialize`, which compiles the text with `loadstring`
//!   (serialize.lua:245) and runs the chunk once. Its interpreter samples
//!   are the parse (serialize.lua:245) and the one run of the loaded chunk
//!   (`=(load):0`), neither of which native code could do.
//!
//! The repro is `repros/unsupported-operands.lua` (verbatim
//! `collect_structure_references`, `apply_terrain_adaptation` and their
//! helpers). Tests: `tests/trace_unsupported_operands.rs` (interpreter-equal
//! results in every profile, native runs, the exits) and the recorder tests
//! in `vm/tests/trace_object_keys.rs`.
//!
//! Measured on 2026-11-03 with the final rules (`eea44831`) against
//! `109cfac0`: one Luanti build relinked against each library, runs
//! interleaved, gated `-x`, load average 3 to 14. Emerge-0 attribution, three
//! runs each: loop@855 went from rejected at structures.lua:856 (2.6%, 2.5%,
//! 2.4% near%) to compiled-exited (0.2%, 0.1%, 0.1%, about 3500 entries),
//! loop@1263 from rejected at structures.lua:1271 (2.2%, 2.3%, 1.9%) to
//! compiled-exited (1.2%, 1.0%, 0.5%). `UnsupportedOperand` at nodeprops.lua
//! named 10, 7 and 4 attribution lines before and none after; loop@1045 now
//! compiles or retires on its exits (2.0%, 3.7%, 5.2%, against 4.1%, 3.8%,
//! 3.1% rejected). Interpreter and native samples were 2358/2315,
//! 2288/1311 and 2634/1647 before, 2492/1595, 2306/1483 and 1882/1651 after;
//! loop@299 swings between compiled and retired in both builds and moves
//! those totals more than this change does. In one `--profile` run each the
//! Emerge-0 interpreter share fell from 14.0% to 11.2% and JIT code rose
//! from 2.5% to 3.7%.
//!
//! The emerge itself did not get cheaper. Six reps each: 295.6, 282.2,
//! 267.6, 282.2, 295.8 and 271.9 G before (median 282.2 G), 289.4, 281.0,
//! 291.2, 289.3, 297.6 and 285.8 G after (median 289.4 G), inside the
//! run-to-run spread but leaning up. Sampling `instructions:u` per thread
//! (two runs each) puts the difference on the compile thread: `oximond-jit`
//! took 28.3 and 31.4 G before and 32.3 and 55.7 G after, while Emerge-0
//! took 170.6 and 178.6 G before and 167.4 and 189.0 G after, and the main
//! thread 78 G in all four. The loops that now compile, and the versions
//! loop@1045 compiles before it retires, cost compile work that the few
//! iterations per entry do not pay back yet. Pinned counts (CPU 4, two runs
//! each, load 4.5 to 5.3): n1 0.836 G, shape-repro 14.00 G, c3 0.646 G, c4
//! 0.042 G and density 1.173 G, all within 0.01% of the base;
//! structures-744, shape-469 and mineshaft-299 within 0.01% too; the
//! repro's 57.07 G went to 57.17 G (+0.18%). Its collect part stays near
//! 14.5 G: most of its time is the interpreted calls around the loops, not
//! the loops. `benchmarks/icount.sh` on fourteen table-heavy tracing
//! benchmarks gave the same counts to 0.1 M for both builds.
//!
//! # Short loops give way
//!
//! Unrolling only starts once a recording of the outer loop has met the inner
//! loop and aborted, and by then the inner loop may have a tree of its own
//! (see "Nested loops"). A tree for a loop of a few trips pays the loop's
//! entry checks on every run and a loop-back on every trip: 205 instructions
//! per outer iteration at eight trips (`for outer = 1, N do for i = 1, 8 do
//! sum = sum + cells[i] * outer end end`, Lua 5.1), where the same nest
//! unrolled costs 30. `TracePolicy::defers_to_outer` makes the inner loop give
//! way when
//!
//! - the frame is not in a coroutine (a top-level chunk or a function), and
//!   the loops around the short one up to the first that is not short are
//!   numeric for loops with nothing else inside (`Recorder::nest_plan`), and
//!   none of those loops calls: a body that calls is bound by the call, and
//!   each unrolled trip would be another call site. In a function the
//!   iterations the outer loop has left are those of this call;
//! - the short loop has at most [`policy::PREFER_UNROLL_TRIPS`] trips per
//!   run, the same number at two closings in a row, and the nest unrolled
//!   fits [`policy::PREFER_UNROLL_GUEST_INSTRUCTIONS`] and
//!   [`policy::PREFER_UNROLL_BACKEDGES`];
//! - the iterations the outer loop has left repay a compile of that many
//!   guest instructions ([`policy::COMPILE_REPAYING_GUEST_WORK`]), the
//!   judgment its recording faces anyway.
//!
//! The first closing only reads the count. The second turns the root away
//! (counted as a `ShortLoop` abort) and makes the outer loop record next with
//! unrolling on. A count that changes needs a side trace for each value,
//! which compile one after the other, and costs more than the tree for the
//! first few hundred thousand iterations (`control/short_inner`: 3 to 6 trips,
//! 45M against 34M at scale 3), so such a loop keeps its tree. If the outer
//! recording ends without a trace the short loop records on its own again,
//! and never gives way again.
//!
//! # Linking through a return
//!
//! A loop that calls a function whose body is a tree (`for _ = 1, n do
//! advance(dt) end` in `nbody`) used to enter the tree once per call: the
//! call's entry and exit cost more than a short nest runs. A tree can now
//! *link through the return* of the function it was entered in, so that the
//! loop around the call and the nest it calls are one native loop. This is
//! the link between a loop and a tree that runs in a callee: the callee's
//! tree continues, in native code, into the code of the caller that comes
//! after the call and calls the function again.
//!
//! The link is a side like any other (one function, a jump into the side, no
//! indirect link cell, see "Side traces"), and it differs from the sides above
//! in what its recording follows:
//!
//! - An outer side that reaches the `Return` of the root frame, in a tree
//!   that has run [`native`]'s `RETURN_THROUGH_ENTRIES` entries, goes on in the
//!   frame that called it. The caller's code is recorded as a *real frame*, as an
//!   inlined callee's is ([`oximond::jit_ir::trace::Entry::Caller`]): its
//!   registers are a frame of the recording, its stores, table constructors,
//!   library calls and global reads are recorded like any other frame's, and
//!   a guard in it leaves through a snapshot whose first frame above the root
//!   is the caller. The recorder refuses what would have to rebuild the
//!   caller or leave it: its own `Return`, a closure, varargs and open
//!   results, a tail call, instructions that close upvalues, a write to an
//!   upvalue, and a write to a register a closure captured.
//! - An exit in the caller's code *pops the root frame*, as the `Return` the
//!   interpreter would have run does: it writes the caller's registers (those
//!   the snapshot names, the results of the call among them, tables the
//!   region made built from their recipes, the rest left as the interpreter
//!   holds them), sets the caller's pc and last pc, and retires the root
//!   frame (`InnerPlan` in `native`, `restore_snapshot` in `deopt`). Every
//!   effect the region committed before the guard is in the tables, and the
//!   snapshot's `After` phase says where the caller goes on. An exit in the
//!   root frame needs no pop; the frame stack below it was never changed.
//! - The caller's registers are window slots of the recording
//!   ([`oximond::jit_ir::trace::ExternalSlot::CallerRegister`]), named when the
//!   region first reads one and when it writes one the code after the call
//!   can see. Planned with the tree they are entry inputs and loop parameters
//!   like the cell of an upvalue, and every exit writes the slots into the
//!   interpreter's caller frame; compiled apart, the side reads them from the
//!   run's extra inputs and hands the run the ones it writes, which the VM
//!   writes into the caller's frame when the run leaves. Either way an exit
//!   anywhere leaves a complete frame stack: the frames never changed, only
//!   their registers. A table the region made and
//!   stored is a real table by the call, and goes into its slot like any
//!   value. A table that has not escaped and is live after the call refuses
//!   the link. The globals the caller reads are window slots too
//!   ([`oximond::jit_ir::trace::ExternalSlot::CallerGlobal`], or a closure's
//!   global when the caller runs a closure), read once at entry from the
//!   caller frame's environment.
//! - The call at the end of the region must be the call that made the root
//!   frame (same instruction, same result window) and must call the root
//!   closure again, which a guard checks. The recording goes on in the root
//!   function from its first instruction as a fresh activation (the
//!   arguments, then nil) and closes where the root's loop header comes
//!   round, like any link.
//! - A caller that runs straight into its own `Return` after the call
//!   (`return not region_evaluate (...)` in `equal_p` of
//!   `mcl_util/shape.lua`, only computing instructions in between, see
//!   `returns_after_call` in [`native`]) never calls the root again, and the
//!   recording refuses that `Return`: every side that went on into it
//!   aborted there or diverged on the way. Such a side does not go on in the
//!   caller; it ends at the root frame's return like a side that may not
//!   (`SideSeed::return_exit`). In a nest of three loops shaped like
//!   `region_evaluate`, called through such a wrapper, side recordings fell
//!   from 32 to 12 and aborts from 25 to 8, and the program ran 15% fewer
//!   instructions.
//!
//! Not done: the innermost loop of Mineclonia's `region_evaluate`
//! (`shape.lua:469`, the `z` loop of the nest at line 434) still runs in
//! the interpreter in the emerge, also with the rule above. The trace log
//! of a 512-block emerge shows four causes, each enough on its own:
//!
//! - The loop runs two trips per entry. The hot count is taken at the
//!   backedge, so the recording starts on the last trip, which leaves the
//!   loop, and the root aborts with `LoopNotClosed` at the `TestEqual` in
//!   front of it, again and again.
//! - When a root does compile, the guard on the `op` callee is hoisted into
//!   the preheader, where its exits have no side key. The callee streak
//!   retires the root within some 20 entries (`retire` logs 23 iterations
//!   in 23 entries, none mismatched, no stone).
//! - `lx_dom` and `ly_dom` (r16, r25) hold nil on some calls and numbers on
//!   others. Outer sides that link back stop with `TypeUnstable`, and the
//!   tree plan refuses them because r16 has different types in the tree.
//! - Sides recorded by beyond scouts cannot take some of their inputs from
//!   the parent's snapshot, and the stepping stone gives up on its outer
//!   sides (14 of 32 runs left by guards with side attempts). The `y` root
//!   at line 459 unrolls past `MAX_UNROLLED_GUEST_INSTRUCTIONS` instead.
//!
//! I tried four mechanisms against these: a root that aborted with
//! `LoopNotClosed` waits for the loop's next entry and records its first
//! trip; a side that reaches the root's header with a type change records
//! the next trip as a copy of the loop, to run as a loop side; side links
//! check the root's entry types; and a hoisted callee guard that fails at
//! entry compiles the root again with its branch guards kept instead of
//! retiring it. Together they made a small nest with a nil-or-number local
//! run 56k native iterations instead of 960. On the targets they cost more
//! than they saved, with `OXIMOND_JIT_SYNC=1`: a reduced `shape.lua` repro
//! went from 4.40 G to 10.59 G instructions, a `shape.lua:469` repro from
//! 95.7 G to 123.3 G, and the emerge from 292.8-294.8 G to 310.0-310.1 G,
//! with the loop still rejected for `LoopNotClosed` in the attribution. The
//! mechanisms are not in the tree. The diagnostics they needed are: `retire`
//! logs the trace's iterations, entries, mismatches and sides, and the stone
//! that gives up logs its interpreter exits by kind and guard.
//!
//! The `z` loop of `region_op` (`shape.lua:359`, the same nest shape) had a
//! cause of its own: the store of `mark_occupied` went through
//! `TABLE_SET_INDEX` beside the row read of `region.solids`, and every
//! recording was refused (see the lane array store under "Tables"). With
//! the store in place, the 512-block emerge logs no refusal in `shape.lua`
//! (7 remain, in other files), and both loops now stop at the nil-or-number
//! locals alone: the `z` root at line 359 compiles and retires with 48
//! entries mismatched in 32, and the roots at line 469 retire with 128 to
//! 1024 mismatched entries each, on `lx_dom`, `rx_dom`, `ly_dom` and
//! `ry_dom` (16 type combinations, against `cache::MAX_TRACE_VERSIONS` of
//! 4). The emerge stayed at 281.4-283.6 G (282.8-295.3 G before).
//!
//! A root now takes such a register untyped. A version that leaves at entry
//! because a guest register held nil where it was recorded with a number,
//! or the other way round, notes the register, and the site keeps the list
//! (`Site::lazy`). The next recording (`Recorder::with_lazy`) makes each
//! listed register that the loop only reads an entry input of type `Any`.
//! The entry does not check its tag. A type guard at the top of the body
//! loads the tag and the bits from the input's slot and leaves when the
//! type is not the recorded one, and the guard takes a side, which runs the
//! iteration with the other type and links back. When the list grows and the
//! site already has `MAX_TRACE_VERSIONS` versions, the oldest goes, so a
//! version that takes the register untyped can replace it, and the site's
//! stepping stones that gave up (`Site::stone_give_ups`) count from zero
//! again: they typed the register, so their outer sides could not link. Only
//! these guards take sides; the type guards of loaded values still leave to
//! the interpreter (giving those sides too cost the emerge some 2%). A side
//! compiled apart reads such a register from the root's slot as the root
//! does.
//!
//! A side that links back may write such a register (the outer side of the
//! `y` loop computes `ly_dom` and `ry_dom` again, the one of the `x` loop
//! `lx_dom` and `rx_dom`). The root keeps it an input of type `Any` and
//! carries no loop parameter for it: native code stores the link's value,
//! tag and bits, into the root's input slot of the register
//! (`Lowerer::store_untyped_link`) and enters the loop at its header, where
//! the type guard reads the slot again (a link that stored skips the fast
//! and slow inner headers, which follow the guards). The slot is then the
//! register's value, while the interpreter's frame still holds what the
//! run's entry found, and the root's exits keep the register `Unchanged`.
//! So the run leaves those slots in the exit area every time
//! (`CompiledTrace::untyped_writebacks`, after the write-backs of sides
//! compiled apart), and the VM writes them into the frame before it
//! restores any exit (`write_back_links`): an exit's `Unchanged` then means
//! the slot's value, and a snapshot that names the register overrides it.
//! A slot no link wrote holds the entry's value, so writing it back changes
//! nothing. The plan (`plan_tree_with`) takes a register for this when the
//! root or any side in the tree takes it untyped; the root gets an untyped
//! input for one only a side read (a root recorded with `lx_dom` nil never
//! reads `ly_dom` behind `lx_dom and ly_dom`). It refuses the link
//! (`TreeError::LinkValue`) for a value with no tag of its own (a table, a
//! string) and for a side compiled apart, whose link goes through the run,
//! which has no such store (`apart_link_writes` refuses it too). A side
//! without a loop of its own may also take such a register untyped when it
//! writes it: until the write its snapshots keep it `Unchanged`, which the
//! verifier allows for an untyped input (`UnchangedWritten` exempts it), and
//! it guards the type only of the ones its operations read. This replaces
//! the refusal of the first version, where a linking side that wrote any
//! register the site listed stopped with `TypeUnstable`, the root's typed
//! loop parameters included.
//!
//! On `repros/nil-or-number-registers.lua` the first version ran 75184 of
//! 79200 `z` trips natively in 3862 entries (no outer side linked); with the
//! links it runs every trip in 1544 entries, 13 outer sides. The program is
//! too short to repay the sides it compiles: pinned, 0.174 G to 0.343 G.
//! Programs like it pay at entry and in Cranelift, not per trip. The reduced
//! `shape.lua` repro (20000 calls, one or two `z` trips each) went from
//! 8.38 G to 19.66 G with 436323 trips native in 108133 entries (1783 in
//! 5502 before): the `region_evaluate` tree now grows through its outer
//! sides, recompiled 36 times at 323 root instructions. A five times longer
//! run (100000 calls) gave 25.16 G before and 35.07 G after, so each further
//! call costs 193 k instructions against 210 k before and 210 k interpreted:
//! the tree pays only a little per call, and its compiles take the rest.
//! `repros/shape-469.lua` went from 14.12 G to 15.01 G, `shape-359.lua`
//! stayed at 14.46 G, and the nest of `trace_short_nests.rs` whose stone
//! used to give up (24 `j` trips per call) runs natively at 2.6 entries per
//! call, 0.61 G against 0.34 G, interpreted 0.26 G: its entries (table
//! bindings, window slots) cost more than the trips save. n1, c3 and c4
//! stayed within 0.3%, density-repro rose 0.7%.
//!
//! In the emerge, roots at line 469 record with `lx_dom`, `rx_dom`, `ly_dom`
//! and `ry_dom` untyped, and their outer sides record and link. Emerge-0
//! had 1134 native samples (878 and 1113 with the `TypeUnstable` refusal,
//! 1450 and 1584 before untyped inputs), 272 of them in the tree at line
//! 469 (0 and 95 with the refusal, 279 and 360 before): the drop the
//! refusal caused is mostly gone. The interpreter share of `shape.lua:101`
//! (8.8%) and of the function at 434 (5.8% own, 21.5% incl) did not go
//! down: the stone of the version with all four untyped gave up within 64
//! entries, its runs leaving by guards with no side attempts left (15 at
//! branch guards, 48 of another class), and the root of the other instance
//! of the chunk retired at its fourth attempt, which rejects the site. Two
//! runs gave 290.1 G and 286.8 G at load 9-23, against
//! 277.5-283.7 G for the first version at load 13-17 (spread about 4%).
//!
//! These two steps were written on a branch that did not land, and were
//! ported onto the tree of `e2061e2e` (2026-11-03). Two rules came with the
//! port:
//!
//! - A version can type a register the site already lists: its recording
//!   started before the list grew, and the version is installed after. Its
//!   mismatches on that register did not grow the list, so it never made
//!   way, and the emerge's roots at line 469 retired with 512 and more
//!   entries mismatched on `r17` (`site list [26, 17]`, untyped `[26]`).
//!   `TracePolicy::note_lazy` now treats such a version as stale once per
//!   register (`Site::lazy_retaken`), which bounds the churn when no
//!   recording can take the register untyped. `OXIMOND_JIT_TRACE` logs the
//!   registers an entry held another type in at the 1st, 2nd, 4th, ...
//!   mismatch of a version, with its untyped inputs and the site's list.
//! - The MIR backend used to refuse a tree with an untyped input, so
//!   Cranelift compiled it, and the tree grows a side at a time: 65
//!   compiles of up to 681k CLIF instructions in the reduced `shape.lua`
//!   repro. MIR takes such a tree now (the type guard reads the tag from
//!   the root's input slot, and a link stores into it). Only a side of it
//!   compiled apart still goes to Cranelift, at the baseline tier
//!   (`native::takes_untyped`).
//!
//! Measured on 2026-11-03 against `e2061e2e` (pinned `instructions:u` on
//! CPU 4 under `heavy.sh -x`, `OXIMOND_JIT_SYNC=1`, binaries at paths of
//! equal length, load 7.2 to 8.7): `shape-469.lua` 16.99 G to 16.60 G,
//! `shape-repro.lua` 4.42 G to 14.00 G, `nil-or-number-registers.lua`
//! 0.143 G to 0.277 G, `density-repro.lua` and `structures-744.lua` +1.2%
//! and +1.1% (the same traces, entries and CLIF counts), n1, c3, c4 and
//! `shape-359.lua` within 0.3%; the tracing bench suite within 0.01% in
//! total. The shape repro is no warm-up artefact alone: at 100000 calls it
//! takes 31.5 G against 21.2 G, 219 k instructions per further call
//! against 210 k. Its tree runs 2.2 M `z` trips natively in 528 k entries,
//! and the entries cost more than the four trips each saves (table
//! bindings, `admit_table`, exit restores and `write_back_links` lead the
//! profile): 279 k runs leave at branch guards and 249 k at the `z` loop's
//! end, where no outer side carries the run on.
//!
//! On the emerge the outcome swings between runs. Of four attribution runs
//! of the final build and its predecessor, two had loop@469 compiled
//! (`near%` 22.7% before, 10.4% and 2.7% after, 125 native samples before,
//! 251 and 319 after; in the second `shape.lua:101` fell from 14.3% to
//! 3.1%) and two had it retired again (20.9% and 23.4%, 0 native): the
//! stepping stone of the all-untyped version gave up on its outer sides,
//! its runs leaving by guards with no side attempts left, and the site
//! used up its attempts. Where it compiles it is entry-heavy as in the
//! repro: 164683 entries, 139514 of them leaving at the `z` loop's end.
//! Emerge, two reps each: 274.8 and 270.3 G before (load 5.8 to 3.1),
//! 281.9 and 277.4 G after (load 3.0 to 2.8); seven more runs of
//! intermediate builds gave 266.4 to 285.9 G. The mechanism stayed off
//! then: with it on, the shape repro took 14.0 G against 4.4 G, and the
//! emerge did not move (Stage 9 tip, three reps each: 270.4, 265.3 and
//! 270.6 G on, 261.2, 275.4 and 263.1 G off). Two things made it cost what
//! it did. An untyped tree compiled in Cranelift at the baseline tier, and
//! a tree of more than 32 members did so at 50 to 80 ms a compile; the MIR
//! backend takes both now. And the sides of an untyped root were refused
//! over the temporaries its table bindings sit in (see
//! `retype_passed_inputs` in `jit_ir::trace::tree`). With those fixed the
//! shape repro takes 3.5 G on against 4.2 G off under `OXIMOND_JIT_SYNC=1`,
//! and two profiled emerges each way gave the emerge thread 112.2 and
//! 112.8 G on against 117.9 and 116.2 G off (interpreter 20.3 and 20.4 G
//! against 26.5 and 24.9 G, the compile thread unchanged). Six unprofiled
//! emerges each way did not agree: 145.2 to 152.1 G on (medians 147.9 and
//! 151.8 G of three) against 144.4 to 152.6 G off (145.8 and 145.4 G), so
//! it stayed off then.
//!
//! Three more faults kept the nest in the interpreter, typed or not:
//!
//! - The hot count is taken at the backedge, and the last backedge of a run
//!   is the pass whose loop test leaves. A `z` loop of two trips started
//!   every other recording there, each ended `LoopNotClosed` after a few
//!   instructions, and the site ran out of attempts. Such a root recording
//!   (at most [`policy::EMPTY_PASS_INSTRUCTIONS`] instructions) is dropped
//!   uncounted, and the next backedge records again, at most
//!   [`policy::MAX_EMPTY_PASS_RESTARTS`] times in a row
//!   (`TracePolicy::restart_empty_pass`). A header that has versions asks
//!   for the version again instead of going cold.
//! - `op` is one closure per caller. Its guard was hoisted in front of the
//!   loop, where no side continues it, and the root retired on a streak of
//!   entries leaving there. The next recording now keeps the guards in the
//!   loop, as after a retirement at hoisted branch guards.
//! - An outer side was refused over a dead table temporary that a child
//!   had added to a parent whose own parent passed the register on
//!   `Unchanged` (`retype_passed_inputs` now looks up the chain with
//!   `started_type`, and no longer skips loop sides).
//!
//! With the first of them, a root that closes as a copy of one retired for
//! its few trips is rejected as before, and the loop now counts as short
//! there too (`Site::retired_short`), so the loops around it unroll it. And
//! a stepping stone that gave up over entries of a type it had typed, while
//! the site lists the register as untyped, no longer counts as a give-up
//! (`Site::stone_forgiven`, once per site): the stone was recorded before
//! the list named the register and outlived the versions that took it
//! untyped, and the next root takes it.
//!
//! With `OXIMOND_JIT_SYNC=1` the emerge is repeatable to 0.02 G, and the
//! emerge thread takes 110.5 G before these changes, 108.4 G after with
//! untyped inputs, and 119.0 G after without: typed, the four versions a
//! header keeps are a draw from the eight nil-or-number mixes of `lx_dom`
//! and its siblings, and the draw decides how many entries mismatch (2918
//! against 29153 logged for loop@469 in two builds that differ in nothing
//! else). Untyped, loop@469 runs 1.87M iterations in 10640 entries. Eight
//! unprofiled emerges each way: 149.3 G before (143.5 to 152.6), 146.3 G
//! after typed (142.0 to 148.8), 144.6 G after untyped (143.1 to 148.0).
//! Mineclonia's startup takes 36.0 G untyped against 35.8 G, devtest's is
//! unchanged, and no benchmark of the Lua suite takes an untyped input
//! (19852 M instructions either way). So untyped inputs are on unless
//! `OXIMOND_JIT_UNTYPED=0` is set. The compile thread pays for it: 6.0 G
//! against 3.7 G before, because a side of a young tree inside other loops
//! compiles with its whole tree, once per side.
//!
//! A repro of the loop at line 359 (`OXIMOND_JIT_SYNC=1`, 60 jigsaw
//! structures) also showed a wrong result before the lane store: the
//! traced run counted 668 or 674 voxels instead of 678, or failed with
//! `bor` on nil. The fault was not in the value selects. A loop side of
//! the tree (the `z` loop, recorded from the terminal exit of a `y` root)
//! had its invariant guards hoisted into its preheader, and the optimizer
//! gave them the root's entry-state snapshot: every register `Unchanged`.
//! A loop side starts from its parent's native state, which the
//! interpreter's registers do not hold, so a failing hoisted guard resumed
//! the `z` loop with the previous run's locals (a boolean in `ly_dom`,
//! `lz` past `lz_max`). The selects only decided which guards were left to
//! hoist (with `OXIMOND_JIT_SELECTS=0` none were invariant). The entry
//! snapshot of a side's hoisted guards now names every entry input and
//! every loop parameter's first value, and a side that starts inside
//! inlined calls hoists no guard (see "Loop-invariant code motion" in
//! `jit_ir::trace::optimize`). The repro gives 678 with and without the
//! lane store, and `trace_loop_sides.rs` pins the shape with a flag that
//! flips the invariant branch of a loop side every 97th call.
//!
//! Checked again for the select-miscompile unit, with the side entry
//! snapshot switched off in a scratch copy: the repro fails again with
//! `OXIMOND_JIT_SYNC=1` and lane stores off (`bor` on nil), and gives 678
//! with it, in both compile modes and with the lane store on or off. The
//! select's lowering, its registers and the `TABLE_SET_INDEX` call that
//! stores its value are the same code either way, so they are not at
//! fault. The reduced test
//! `a_select_in_front_of_a_loop_side_stored_through_a_raw_set_agrees`
//! (`trace_value_selects.rs`) fails the same way without the entry
//! snapshot and passes with it; its CLI copy fails without it even with
//! `OXIMOND_JIT_SELECTS=0`, where a guard in front of the loop replaces the
//! select and the loop's own guard is just as invariant.
//!
//! The siblings looked at: a hoisted load guard of a loop side would leave
//! through the same snapshot. No test reaches that case: the field read of
//! `a_loop_sides_hoisted_field_guard_that_fails_agrees`
//! (`trace_value_selects.rs`) is not hoisted, since the side's stores to
//! lane tables may write its table, and the test passes with the side's
//! entry snapshot switched off (checked in a scratch copy, 2026-11-02). The
//! two tests named above are the ones that fail without it. A loop side whose loop-back passes a parameter on
//! unchanged runs its preheader at the header, where the parameters'
//! current values are named; one without runs it once, and a side that
//! rejoins it (`LinkLoop`) jumps past it; a root's preheader runs again after
//! a link only when a parameter passes through, and those are named; and
//! the recorder's own snapshots of a side name every register the side
//! starts with. None other leaves through an entry state it did not start
//! from.
//!
//! A tree with a loop side like the `z` loop's is entered only by a root
//! frame that its caller called that way: the same chunk and closure, the
//! call at the same pc, the same result window, and a frame as a fresh
//! call makes it (no tail
//! calls, varargs, captured locals or interior state), and no register of
//! the caller that is captured by a closure other than the ones captured when
//! the sides were recorded (the tree writes registers of the caller that a
//! captured variable would not see). Any other entry is a mismatch and the
//! interpreter runs, so a function called from two places gets a version for
//! each.
//!
//! What happens to the link when its tree changes: the link is part of the
//! tree, so there is no target to patch. A flush, an eviction, a retirement
//! or a retree replaces the tree with its caller context in one piece (the
//! context is rebuilt from the sides' frames each time the tree installs);
//! the caller frame then runs in the interpreter, which finds the call
//! as a plain call. A blacklisted header is never entered again. A side that
//! returns into the caller is compiled apart from its tree (see "Sides that run
//! through the loops around the root, and sides that return"): the link does
//! not compile the tree again, so the gate above does not scale with the
//! size of the tree, and the tree is entered only by the caller it was
//! linked to as soon as the side joins. A tree that is compiled again for
//! another reason (the peak tier, or sides that cost more than the compile)
//! takes the sides inline and widens, as sides that returned did before they
//! could be compiled apart. A coroutine yield, an error, a hook, a
//! callee that is another closure, a type change or a break leave through a
//! guard of the tree. Calls that are not plain (`pcall`, a method with
//! varargs, open results, a tail call), trees with count hooks, trees that
//! resume coroutines, and a root that calls itself do not link.
//!
//! The caller's code is not unrolled or linked further: a loop in it other
//! than the one that calls the function again ends the recording as an inner
//! loop, and a function that is called from a loop that is in a caller's
//! caller is linked one call at a time, not through the frames above.
//!
//! # Flush and lifetime
//!
//! A flush ([`TracePolicy::flush`]) advances the epoch, forgets every site,
//! cancels recording, drops queued requests and bumps the epoch that stale
//! broker results are checked against. Code is freed once no native run
//! uses it (see [`cache`]).
//!
//! # Notes for the compile packages
//!
//! - Fast-path backedges that arm `jit.backedge_skip` or a window of
//!   `jit.rejected_skips` stay inside `interpret_fast` and never reach
//!   [`TracePolicy::at_loop_header`]. While a skip is armed,
//!   [`TracePolicy::credit_skipped_backedges`] counts the skipped window
//!   toward the header's hotness. `backedge_skip` arms while a header waits
//!   for its first compile and no trace is installed (`COMPILE_WAIT_SKIP`),
//!   and counts the backedges of every loop. A rejected header has a window
//!   of its own that counts its backedges only, so it delays no other
//!   header's lookup, whatever traces are installed; its windows double (up to
//!   `TRACING_REJECTED_BACKEDGE_SKIP_MAX`) for as long as the loop stays
//!   rejected. The end of a skip polls the collector, which a skipped
//!   backedge does not.
//! - A trace retired before it ran one productive window is remembered by its
//!   site. If the next recording closes as the same trace and nothing else is
//!   compiling or installed (which could have starved the first one), the
//!   site is rejected without a compile. A trace retired twice that comes
//!   back again is rejected whatever else is compiling.
//! - A trace or tree whose work is mostly calls of the string library's
//!   pattern functions compiles at the baseline tier
//!   ([`native::is_helper_bound`]): the calls cost the same native and
//!   interpreted, so the code around them has little to gain, and the
//!   optimizing tier's compile is a few times the baseline's. A loop of
//!   hundreds of milliseconds would not notice; the loops of a short run do.
//! - `maybe_run_native` consults the policy after the hook, tracing cooldown,
//!   and `InterpretOnce` checks. Keep it there.
//! - `record` runs before an instruction executes. The fast loop's
//!   `RECORD = true` instantiation calls it before dispatch, and the general
//!   loop calls it only for instructions the fast loop did not hook. The
//!   recorder drops a repeated hook for the same unexecuted instruction.
//! - Compilation runs on the broker's worker. [`TracePolicy::queued`] holds
//!   closed traces only while native compilation is switched off.
//! - A native exit restores the frame exactly. The dispatcher then runs the
//!   instruction at the resume pc before it consults the policy again, so
//!   the trace cannot re-enter the state it just left. No marker is needed.
//! - `maybe_run_native` returns to the dispatcher after a trace exit, and the
//!   dispatcher recomputes which frame is on top, since an exit inside an
//!   inlined call pushed frames.
//! - A frame with captured upvalue slots enters a trace. Its registers are
//!   the variables their open cells stand for, so the exit's restore of
//!   registers is also the restore of the cells.
//!
//! # Function traces
//!
//! A function that calls itself directly is hot at its entry. A Lua call that
//! pushes a frame of the calling frame's own chunk sets `JitState::call_landing`
//! (the fast loop settles a cold, counting or rejected entry in place), and
//! the dispatch loop consults the policy for the site of the function's
//! entry: the chunk with [`policy::FUNCTION_ENTRY_PC`], a pc no instruction
//! has, so it never shares a site with a loop header at pc 0. A hot entry
//! records one activation, from pc 0 to its return (see [`recorder`],
//! "Function traces"), and each direct self call in it becomes a nested
//! activation of the same native code. Only self calls consult: a function
//! entered from elsewhere runs its first activation in the interpreter, and
//! its self calls enter the trace.
//!
//! Each activation is a native function call with its own part of the exit
//! area (see [`oximond::jit::CompiledTrace::activation_stride`]). A return
//! hands the value to the calling activation, or, from the entered one, to
//! the VM, which returns from the frame as the interpreter does. An exit in
//! a nested activation unwinds: every activation below it spills its state
//! at its self call, and the VM rebuilds one frame per activation, each
//! stopped at the call that made the next (see [`native`]). A run nests at
//! most [`native::MAX_NATIVE_ACTIVATIONS`] activations, and never more than
//! the call-depth limit leaves room for: the deepest self call leaves before
//! the call, and the interpreter's call enters the trace again one frame
//! further down. Each activation spends one unit of fuel. A function trace
//! is judged by the activations it runs per entry over its life (see
//! [`native::TRACE_MIN_HALF_ACTIVATIONS_PER_ENTRY`]), and waits while a side
//! for one of its guards compiles: until then every run leaves at that guard
//! from deep in the recursion, and the frames each exit rebuilds would enter
//! the trace for little work.
//!
//! # Functions the host calls
//!
//! A function a host calls (`Vm::call_value`, the C API's `lua_call`) has no
//! function-entry site: its entry is never offered to the policy, and only
//! the loops inside it trace. Luanti is the case that asked. Its C++ reads
//! every position through `read_vector` and writes them back through
//! `push_vector` (builtin/common/vector.lua:408 and :416), and the emerge's
//! attribution charged 5.0% of the Emerge-0 interpreter samples to
//! `read_vector`, "entry never-hot". A `perf` call graph of the same emerge
//! (dwarf unwinding) put 0.58% of the thread's samples under `read_v3_aux`,
//! the C++ that calls it, and less than 0.1% in the function's four
//! instructions. The rest of the 5.0% was C++ time: a tick that passes in
//! host code was seen at the next dispatch, the function the host calls
//! next. The sampler now counts those ticks as host samples
//! ([`crate::vm::attribution`]). A trace of a four-instruction function
//! would cost as much to enter as it saves, so the work went into the host
//! boundary instead. A host entry runs on a stack the VM keeps from the last
//! one (`Vm::run`). A host callback parks its caller's stack and builds the
//! debugger's records of it only when asked (`HostSnapshot`). The C API
//! roots its stacks when the collector asks
//! (`Runtime::set_host_roots_refresher`), not at every call, and a pop of
//! values that own nothing only shortens its stack. On a C++ repro of the
//! emerge's conversions (target/flow-luanti-native/repros/vector-entry.cpp)
//! that took 77.6 G instructions to 40.4 G. In the emerge's Emerge-0
//! profile, root building, frame capture and stack resizing fell from about
//! 3.4% of the samples to under 0.5%. A loop that calls a host function
//! still aborts its recording at the call (`UnsupportedOperand`). The host
//! entries inside it are what got cheaper.
//!
//! # Inlined calls
//!
//! The recorder inlines calls of Lua closures with fixed arguments and
//! results, vararg callees included (see [`recorder`]). A trace pins each callee by its generation-aware
//! identity: lane `k` of every run's root bank is pinned closure `k`, and a
//! call-site guard compares the live callee with that lane. Entry fills the
//! bank with the pinned closures first and refuses a frame whose pinned
//! closure no longer exists (a stale handle matches nothing, even when its
//! slot was reused), whose shared cells are not the cells the trace promised
//! (see [`oximond::jit_ir::trace::CellAlias`]), or that has no room under the
//! call-depth limit for the frames an exit inside the deepest callee pushes.
//!
//! A call site whose callee takes turns between closures (locals that swap
//! places, an array of closures) is guarded on the prototype instead: one
//! trace serves every closure of it, and a side trace each other prototype.
//! The callee's cells are read and written through the live closure, by
//! native code where the cell holds a number or a boolean, and a frame an
//! exit rebuilds runs the closure of that run (see [`recorder`]). Entry
//! gives each closure of the bank a row with its prototype and the addresses
//! of its first cells, like the rows of tables, and leaves the row of a
//! closure that shares a cell with a window slot zero, which no prototype
//! matches. A closure the run meets later gets its row when it gets its
//! lane: an array or field load admits it with its row, and a helper that
//! returns a closure (`TABLE_GET_CLOSURE` of a rowless read) fills the row
//! after the call in a tree that reads closure rows. The rows are shared by
//! every trace of the thread and not cleared between runs, so a row nothing
//! filled holds another run's prototype word, which can match the pinned
//! closure and run another prototype's code (a placement field that took a
//! closure made in the loop, `trace_poly_protos.rs`).
//!
//! The successful path materialises no frame. An exit inside a callee
//! restores the root frame as the interpreter would have it stopped at its
//! call and pushes the callee frames above it, in call order, from the pinned
//! closures ([`deopt`]). The interpreter then continues the innermost call
//! exactly once, with a call chain a traceback, `debug.getinfo`, an error or a
//! coroutine yield sees as any other.
//!
//! # Coroutines
//!
//! A loop that resumes coroutines (`coroutine.resume`, or a function
//! `coroutine.wrap` made) runs them from native code. A coroutine is an
//! opaque value like a table: an array read of one (`threads[i]`) gives it a
//! lane of the run's root bank, which roots it, and the call is a helper
//! ([`oximond::jit_ir::trace::HelperId::coroutine_resume`]) that takes the VM
//! and the call stack the trace runs on from the run and resumes the thread
//! through the switch the fast loop uses (`Vm::resume_for_trace`). The thread
//! runs on its own heap stack, so the native stack does not grow, and native
//! execution stays off for it: its loops are not entered or recorded while the
//! trace waits, since the trace owns the engine's state. The collector sees
//! the run in flight: the bank is a root, and the stack the trace runs on is
//! rooted whole.
//!
//! What the thread may do to everything else, the trace checks after each
//! call instead of assuming it away: the array, length and header words of
//! its bindings are read again from the tables, the callables and strings
//! its bindings fixed, the cells and globals it reads and the captured locals
//! it reads are compared with what the run started with, and a hook that
//! appeared declines the next call. A trace that writes a cell or a captured
//! local, stores through a field, reads a table by lane (a lane array view
//! included, since the thread may move the array it read the words of) or
//! holds a header does not record a resume at all
//! ([`oximond::jit_ir::trace::TraceIr::resume_conflict`]).
//!
//! The helper answers only the call the recording saw: `true` and integers,
//! floats or booleans of the recorded kinds. Anything else (an error, a dead
//! thread, another count or type of values, a check that failed) is a miss
//! *after* the thread ran. The VM keeps what the call gave, the trace leaves
//! before the call as for any library call, and the interpreter's execution of
//! the call takes the answer from the VM instead of running the thread again
//! (`Vm::replay_resume`). No guard of these helpers takes a side trace: a side
//! would run the call a second time.
//!
//! A thread's frames show the interpreter state of the trace's entry to
//! `debug.getinfo` and the debug library's stack walkers while it waits, not
//! the state its pc names.
//!
//! # Cells, globals and opaque values
//!
//! The root frame's window is longer than its registers: the upvalue cells
//! and the globals a trace names follow them as extra registers
//! ([`oximond::jit_ir::trace::ExternalSlot`]). The recorder reads and writes
//! them like registers, so loop formation, snapshots and the optimizer treat
//! them alike. A trace holds a cell in a native value for the whole run and
//! writes it back at every exit, which is exact because a trace makes no call
//! and nothing else can see the cell meanwhile. A global is only read, and
//! only while the environment holds a present raw value for it (the same
//! condition under which the interpreter skips `__index`). It is read when
//! the trace is entered, so the trace names no function or table constant and
//! a rebound global is simply another value. Entry refuses a frame whose
//! cells alias each other or a captured local of the frame.
//!
//! A side trace names cells like the root does, a callee's included: it may
//! read and write a cell of an inlined callee that the tree has not named
//! yet (the closure iterator of a generic `for` that keeps its position in
//! captured locals, which a branch the root did not take writes). The side
//! cannot check that the new cell is distinct from the others it holds, but
//! the tree can: the externals and aliases of the side join the tree's, and
//! every later entry checks them all (`upvalues_alias`, `aliases_hold`). A
//! frame that fails the check never enters the tree. Every exit of the side
//! writes the cells back, as the root's do.
//!
//! # Wide integers (int64 and uint64 cdata)
//!
//! `LuaJIT`'s `int64_t` and `uint64_t` cdata are immediates here, not boxes:
//! `Value::Int64` and `Value::UInt64` in the interpreter, and
//! [`oximond::jit_ir::trace::TraceType::Int64`] and `UInt64` in the trace
//! IR, where they live in machine registers like integers. `1ull`, `x *
//! 1ull`, `bit.band (x, 0xffffffffull)` and `tonumber (x)` allocate
//! nothing in either, so there is no box for a trace to sink, and a wide op
//! that cannot fail (`wide_arithmetic` returns `None` only for a zero
//! divisor) takes no guard and no snapshot. What made Mineclonia's
//! `load`-made helpers in `random.lua` expensive was elsewhere:
//!
//! - In the interpreter, each half of a result is `tonumber (band (value,
//!   0xffffffffull))`: a `Call` with open results and a `CallOpen` that
//!   takes them. Both left the fast loop for the dispatch loop, about 400
//!   instructions a pair. A native that returns one value (a `Scalar`,
//!   `Unary` or `Binary` plan) now runs in place for both
//!   (`Vm::open_native_in_window`): the `Call` inside the fast loop, the
//!   `CallOpen` through `native_call_activation`, since a call site in the
//!   `CallOpen` arm cost the loop's register allocation 3% on unrelated
//!   code. Interpreted, the `ull-helpers` repro's `getlong` went from 6.32 G
//!   to 5.46 G instructions and `mineclonia/ull_kernel` from 286 M to 252 M
//!   (2026-11-01).
//! - In traces, the helpers keep their RNG state in `{ lo, hi }` pairs, and
//!   each array read and store of a pair half was a guard with a snapshot,
//!   about one snapshot per three guest instructions. Reads of a slot the
//!   recording already read or stored are now moves with no guard, and
//!   in-place stores into such a slot of the value's type are
//!   [`oximond::jit_ir::trace::TraceOp::ArrayOverwrite`]s with no guard and
//!   no snapshot, which the verifier proves from the IR (slot facts, see
//!   `recorder::slot_facts` for the aliasing and kill rules). `maxsnap`
//!   counts only the snapshots a guard leaves through, so the one
//!   `hoist_callee` takes before each `band` of `tonumber (band (...))`,
//!   which no guard used, is free (see "Guards and snapshots" in
//!   [`recorder`]). A failing chunk test of the `structures-744` repro went
//!   from 352 snapshots to 102 guarded ones for 1089 guest instructions,
//!   and its tracing run from 29.40 G to 23.15 G instructions.
//!
//! On the 512-block emerge (2026-11-01, 8d4b2cdf against this, load
//! 2.5-10), structures.lua loop@744 is no longer rejected for
//! `LimitExceeded`: its root records 1782 guest instructions with 132
//! guarded snapshots and compiles, and a side from its callee guard
//! records 3421 with 233. It then retires: `placement.chunk_test` is
//! another closure of one prototype for each structure set, so every entry
//! leaves at that callee guard ("33 iterations in 33 entries, exit 40
//! (member 0, guard Some(11) Some(Callee), frame 2 pc 9) 33x"), and the side
//! does not join the optimized root ("a cell alias stands for a name with
//! no window slot"). random.lua loop@1159 (Mineclonia's own xoroshiro
//! self-test, run at load) retires on the callee guard of
//! `fork:next_integer`, a new closure for each of its 100 forks ("exit 17
//! (member 0, guard Some(1) Some(Callee), frame 1 pc 1064) 17x"). Neither
//! is about how wide values are held or about a limit; both need a call
//! guarded on the prototype that stays cheap. So the helpers still run
//! interpreted on Emerge-0: the `=(load)` rows of the attribution are
//! 27.8% of interpreter samples before (:12 8.2, :60 4.0, :74 3.5, :81
//! 3.2, :50 2.6, :55 2.3, :33 2.1, :41 1.9; 3803 samples) and 29.3% after
//! (:12 7.2, :33 5.0, :74 3.6, :81 3.5, :55 3.1, :50 2.7, :60 2.3, :41 1.9;
//! 3633 samples), about 1060 samples both times, and the emerge is
//! 287.9/285.2 G against 283.6/282.9 G instructions, inside the run-to-run
//! spread. Native samples on Emerge-0 went from 903 to 1205.
//!
//! The side of loop@744 that the tree refused ("a cell alias stands for a
//! name with no window slot") inlines `placement.chunk_test`, whose cell is
//! a slot only that side names: `verify_cell_aliases` checks an alias
//! against the root's window, not the tree's. I let the check take the
//! slots the sides add (the patch is
//! `target/flow-luanti-native/repros/structures-calls-alias-window.patch`).
//! The side then joins, but the tree still leaves at the callee guard of
//! the next closure on every entry and retires, so the change only buys
//! the compiles of the bigger tree: the repro
//! `target/flow-luanti-native/repros/structures-calls.lua starts` went from
//! 22.45 G to 29.82 G instructions (`finish` 9.38 G to 10.43 G), and the
//! emerge measured 297.3/288.2 G against 288.3/282.8 G. Dropped; the
//! callee of a field that takes turns between closures of one prototype
//! needs a guard that holds across them first (see `trace_poly_protos.rs`
//! for why the prototype guard alone does not pay). I also tried that
//! guard again for this loop, with the closure's cells read through the
//! closure at run time and sides seeded inside dynamic frames and tail
//! calls (`target/flow-luanti-native/repros/structures-calls-poly-attempt.patch`):
//! the `starts` repro ran 1.22 to 1.31 s against 0.92 s, with 23649 entries
//! and 20179 exits inside inlined frames. Its sides pinned the natives
//! `floor` and `ceil` in other orders than the root ("a side trace has
//! another window than its root"), and the root retired at one iteration
//! per entry.
//!
//! # Limits
//!
//! The caps of one recording and of one tree are at `LuaJIT`'s scale
//! ([`recorder::MAX_TRACE_OPS`], [`recorder::MAX_TRACE_IR_OPS`],
//! [`recorder::MAX_TRACE_SNAPSHOTS`], [`policy::MAX_SIDE_TRACES`],
//! [`cache::MAX_INSTALLED_TRACES`]). `jit.opt.start('maxrecord=..',
//! 'maxsnap=..', 'maxside=..', 'maxtrace=..', 'hotloop=..', 'maxmcode=..')`
//! sets them per VM (`TracePolicy::set_jit_options`): the runtime keeps the
//! request, and the VM applies it at its next safe point together with a
//! flush, since traces recorded under other limits are not comparable.
//! `maxrecord` bounds the guest instructions of a recording and, doubled, its
//! IR operations. The other caps (externals, root constants, pinned
//! callables, fields, virtual tables, inlined calls) are fixed. With
//! `OXIMOND_JIT_TRACE` set, an abort for a limit names the cap it hit.
//! `maxsnap` counts the recorder's snapshots that a guard leaves through,
//! not its guards: guards with nothing between them share one, a call of a
//! callee value the recording already guarded takes no guard, and a
//! snapshot taken for a guard that never came (the one before the call that
//! makes open results for `tonumber (band (...))`, when no guard of the
//! call leaves through it) is free, as `LuaJIT` never takes one (see "Guards
//! and snapshots" in [`recorder`]).
//!
//! Tables and closures travel through a trace as lane ids into the per-run
//! root bank ([`TracePolicy::root_values`]). They can be moved, compared for
//! identity and tested for truthiness. A table comparison is guarded before
//! it runs, because two different tables may still be equal through `__eq`,
//! which only the interpreter can evaluate. The bank fills at entry, grows
//! when native code loads a table out of an array (the VM gives the table the
//! next lane, or the lane it already has) and is empty once the exit restored
//! the frame, so no object (and no entry of a weak table) stays alive because
//! a trace once touched it. String
//! constants a loop stores into registers ride to the exits as
//! per-trace root slots resolved from the chunk's constants.

pub(crate) mod cache;
mod coverage;
mod debug_env;
#[allow(
    dead_code,
    reason = "native exits restore through per-exit plans; the generic restore is the specification they are tested against"
)]
pub(crate) mod deopt;
mod helpers;
mod host;
pub(crate) mod native;
pub(crate) mod nest;
mod perf_map;
pub(crate) mod policy;
mod profile;
pub(crate) mod recorder;
mod side_queue;
pub(crate) mod stats;

use crate::bytecode::Chunk;
use crate::compat::{CompatibilityProfile, ExecutionMode};
use crate::jit::{CompilationId, CompiledTrace, OptimizationTier, SubmitError, TraceCompileError};
use crate::jit_ir::trace::{TraceId, TraceIr, TraceSide};
use crate::jit_ir::{observable_registers, register_set_contains};
use crate::runtime::alloc_phase::{self, Phase};
use crate::runtime::{ThreadHandle, Value};
use cache::{
    MAX_CONFIGURED_CODE_BYTES, MAX_CONFIGURED_TRACES, MIN_CONFIGURED_CODE_BYTES, TraceCache,
};
pub(crate) use native::TraceRun;
use native::{BROKER_FULL_SKIPS, COMPILE_REFUSAL_COOLDOWN, QUEUE_FULL_COOLDOWN};
use policy::{
    AbortClass, COMPILE_WAIT_SKIP, COVERED_MIN_ITERATIONS, COVERED_MIN_ITERATIONS_FLOAT,
    COVERED_WINDOW, LONG_COOLDOWN, MAX_RECORDING_RESTARTS, MAX_REPEAT_RESTARTS, MAX_SIDE_TRACES,
    SiteState, SiteTable, TRACE_HOT_THRESHOLD, TRACE_QUEUE_CAPACITY, attempt_limit, classify_abort,
};
use recorder::{ClosedTrace, RecordLimits, Recorder, SharedLiveness, Step, WakeCandidate};
pub(crate) use recorder::{HookPoint, HookSource, Surroundings};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Weak};

/// Chunks whose liveness the policy keeps.
const MAX_LIVENESS_CHUNKS: usize = 16;

pub use coverage::{Coverage, CoverageState};
pub(crate) use nest::{LoopSpan, loops_of};
pub(crate) use policy::FUNCTION_ENTRY_PC;
pub use policy::RejectCause;
pub use stats::{NativeAborts, TraceAbortReason, TraceExitKind, TraceGuardKind, TraceStatistics};

/// Who runs a loop header the interpreter just reached.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TraceDispatch {
    /// The interpreter runs the header.
    Interpret,
    /// Like `Interpret`, for a header the tracer has given up on: the
    /// interpreter may skip the next backedge lookups.
    Skip,
    /// A compiled trace owns the header. Run the trace installed at this
    /// table index with [`TracePolicy::enter`].
    Enter(u32),
}

/// Where a compile request's result goes.
#[derive(Clone, Debug)]
pub(crate) enum CompileTarget {
    /// Code for a freshly recorded trace at a loop header.
    New {
        chunk: Weak<Chunk>,
        header_pc: usize,
    },
    /// Code for the installed trace `index` with one more side trace in its
    /// tree. It replaces the trace's code and its tree, if the trace is
    /// still the one that asked for it (same `serial` and `version`).
    Retree {
        index: u32,
        serial: u64,
        version: u32,
    },
    /// Code for one side of the installed trace `index`, to be called from
    /// the exit it continues. The trace keeps its code and its tree, and
    /// grows by the side, if it is still the trace that asked for it (same
    /// `serial` and `version`). `source` is the side's index in the tree's
    /// sides.
    Side {
        index: u32,
        serial: u64,
        version: u32,
        source: usize,
    },
}

/// The guest instruction a recording met its abort at: its pc in the chunk of
/// the frame the interpreter was running (the root frame, a callee the
/// recording inlined, or the caller it returned into), and whether that
/// instruction returns from its frame.
#[derive(Clone, Copy, Debug)]
struct AbortSite {
    pc: usize,
    is_return: bool,
    /// Whether the instruction is the latch of a numeric for loop, a kind of
    /// inner loop a later recording can unroll.
    is_counted_latch: bool,
    /// Whether the instruction is the backward jump of a `while` loop or the
    /// latch of a generic for loop, which a later recording unrolls when the
    /// loop is in an inlined call.
    is_while_jump: bool,
    /// The chunk and the header the instruction branches back to, when it is
    /// a backward branch.
    branch_back: Option<(*const Chunk, usize)>,
}

impl AbortSite {
    fn of(at: &HookPoint<'_>) -> Self {
        Self {
            pc: at.pc,
            is_return: at.chunk.code.get(at.pc).is_some_and(|instruction| {
                matches!(
                    instruction.opcode(),
                    Ok(crate::bytecode::OpCode::Return
                        | crate::bytecode::OpCode::ReturnOpen
                        | crate::bytecode::OpCode::ReturnVararg
                        | crate::bytecode::OpCode::Halt)
                )
            }),
            is_counted_latch: at.chunk.code.get(at.pc).is_some_and(|instruction| {
                instruction.opcode() == Ok(crate::bytecode::OpCode::NumericForLoop)
            }),
            is_while_jump: at.chunk.code.get(at.pc).is_some_and(|instruction| {
                matches!(
                    instruction.opcode(),
                    Ok(crate::bytecode::OpCode::Jump | crate::bytecode::OpCode::ForInLoop)
                ) && nest::jump_target(at.pc, *instruction).is_some_and(|target| target <= at.pc)
            }),
            branch_back: at
                .chunk
                .code
                .get(at.pc)
                .and_then(|instruction| nest::jump_target(at.pc, *instruction))
                .filter(|target| *target <= at.pc)
                .map(|target| (Arc::as_ptr(at.chunk), target)),
        }
    }
}

/// Why a tree compile did not produce code.
#[derive(Clone, Copy)]
enum SideFailure<'a> {
    /// The broker or cache had no room. Says nothing about the sides.
    Refused,
    /// The backend failed on this tree.
    Failed(&'a std::sync::Arc<crate::jit_ir::trace::TraceIr>),
}

/// The side recording in progress: which guard of which trace it continues.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SideTarget {
    /// The installed trace, by table index and serial.
    pub(crate) index: u32,
    pub(crate) serial: u64,
    /// The key of the guard (see [`native::SideKey`]).
    pub(crate) key: native::SideKey,
}

/// A compile request the policy wants the broker to run.
#[derive(Debug)]
pub(crate) struct CompileRequest {
    pub(crate) ir: Arc<TraceIr>,
    pub(crate) tier: OptimizationTier,
    pub(crate) epoch: u64,
    pub(crate) target: CompileTarget,
    /// Metadata bytes the request reserved in the cache.
    pub(crate) reserved: usize,
    /// Set for a request to compile one side apart from its tree.
    pub(crate) side: Option<crate::jit::SideJob>,
    /// The members of `ir` as planned when the request was made, for a tree
    /// request. The broker compiles them without planning `ir` again.
    pub(crate) planned: Option<Arc<Vec<crate::jit_ir::trace::TreeMember>>>,
    /// Cold traces (index and serial) to evict once the broker accepts the
    /// request, to make room for what it reserved. A refused request evicts
    /// nothing.
    pub(crate) victims: Vec<(u32, u64)>,
}

/// A request the broker accepted, waiting for its result.
#[derive(Debug)]
pub(crate) struct PendingCompile {
    target: CompileTarget,
    epoch: u64,
    reserved: usize,
    /// The IR the request compiles, kept to build the installed trace.
    ir: Arc<TraceIr>,
    /// Requests the broker folded into this one because their IR has the
    /// same content. One compiled trace installs at one target, so each
    /// follower is submitted again once this one finishes.
    followers: Vec<CompileRequest>,
}

/// A loop header the interpreter just reached by a backward step.
pub(crate) struct LoopHeader<'a> {
    pub(crate) chunk: &'a Arc<Chunk>,
    pub(crate) depth: usize,
    pub(crate) pc: usize,
    /// The backward branch that reached the header.
    pub(crate) latch: usize,
    pub(crate) profile: CompatibilityProfile,
    pub(crate) coroutine: Option<ThreadHandle>,
    /// The frame's captured locals: open upvalues, named by absolute stack
    /// slot, which `captured_base` turns into registers. A recording starting
    /// here needs them to see which cells an inlined callee shares with it.
    pub(crate) captured: &'a [crate::vm::stack::OpenUpvalue],
    /// The first stack slot of the frame's registers.
    pub(crate) captured_base: u32,
    /// The closure the frame runs. A loop through tail calls closes only on
    /// a tail call back into this very closure.
    pub(crate) closure: Option<crate::runtime::ClosureHandle>,
    /// Whether this is a function's entry, which a self call reached (`pc`
    /// is then [`policy::FUNCTION_ENTRY_PC`] and the frame stands at pc 0),
    /// rather than a loop header.
    pub(crate) function: bool,
}

/// A recorded trace waiting for compilation.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "the compile packages read the queue; tests read it today"
)]
pub(crate) struct QueuedTrace {
    pub(crate) chunk: Weak<Chunk>,
    pub(crate) header_pc: usize,
    /// The policy epoch the trace was recorded in.
    pub(crate) epoch: u64,
    /// Guest instructions in one iteration.
    pub(crate) guest_instructions: u32,
    pub(crate) ir: TraceIr,
}

/// Owns the trace engine's state for one VM.
#[derive(Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent switches: enabled, compiling, recent entry, link counting"
)]
pub(crate) struct TracePolicy {
    trace_profile: profile::ProfileState,
    /// Whether the engine can record at all: tracing execution on a target
    /// that can run native code.
    enabled: bool,
    /// The recording in progress, if any. One at a time.
    recorder: Option<Box<Recorder>>,
    sites: SiteTable,
    /// Closed traces that wait because native compilation is off, oldest
    /// first.
    queue: VecDeque<QueuedTrace>,
    /// Whether closed traces compile to native code. Only tests turn it off.
    compile: bool,
    /// Compiled traces, indexed by `SiteState::Installed`, with the byte
    /// accounting and eviction.
    cache: TraceCache,
    /// The caps of new recordings (`jit.opt.start`).
    record_limits: RecordLimits,
    /// Side traces one tree may hold (`maxside`).
    max_sides: usize,
    /// Backward branches before a loop is recorded (`hotloop`).
    hot_threshold: u32,
    /// Compile requests not yet handed to the broker.
    outbox: Vec<CompileRequest>,
    /// Requests the broker holds, by its ID.
    pending: FxHashMap<CompilationId, PendingCompile>,
    /// Recording starts left to skip because the broker refused the last
    /// request for lack of room. A trace recorded now would be refused too,
    /// and the recording and its verification are the expensive part. A
    /// completion or an accepted request ends the wait; the count bounds it
    /// when neither comes.
    broker_full_skips: u32,
    /// Reusable entry-slot buffer for [`Self::enter`].
    inputs: Vec<crate::jit::TraceInputSlot>,
    /// Reusable table-binding buffer for [`Self::enter`].
    binding_views: Vec<crate::jit::TraceBinding>,
    /// The string indexes of the small tables a run's hash lookups bind,
    /// reused between runs.
    probe_scratch: crate::runtime::NativeStringProbeScratch,
    /// Where recent entry reads of constant names (a global, or a field of a
    /// table an upvalue holds) found their values, so the repeat of an entry
    /// skips the hash probe while the table is unchanged. Boxed: it is a few
    /// kilobytes.
    entry_reads: Box<crate::runtime::StringReadCache>,
    /// The lanes of the tables a run holds, by identity, emptied at every run.
    lane_index: Box<native::LaneIndex>,
    /// The field slots of the tables a run meets, reused between runs. Boxed
    /// so that a run's reference to it points off the policy: the code a
    /// helper runs in the middle of the run borrows the VM, policy included,
    /// as a whole (see `ResumeHost`).
    table_space: Box<crate::jit::TraceTableSpace>,
    /// The lane of each table of the root bank during a run that meets tables.
    /// Boxed like [`Self::table_space`].
    table_lanes: Box<rustc_hash::FxHashMap<u64, u32>>,
    /// Reusable exit area for [`Self::enter`].
    exit_area: Vec<i64>,
    /// Whether a native run happened since [`Self::take_recent_entry`] last
    /// asked.
    entered_recently: bool,
    /// The root bank: every table and closure a native run's lanes name. It
    /// fills at entry and empties when the run's exit has restored the frame,
    /// so it holds nothing between runs. See [`Self::root_values`].
    bank: Vec<Value>,
    /// Whether a native run is in flight: set from its entry checks until
    /// its exit has restored the frame. See [`Self::run_in_flight`].
    in_flight: bool,
    /// The address of the VM that is entering a trace, for a trace that
    /// resumes coroutines (zero when none is). See [`Self::set_resume_host`].
    resume_vm: usize,
    /// Whether any debug hook is installed, as the VM last told the policy.
    /// Every hook is then a count hook (the VM refuses traces for any
    /// other), so new recordings count guest instructions, and only counted
    /// traces enter.
    hooks_installed: bool,
    /// The live count of the running thread's count hook, when it has one:
    /// the VM sets it before an entry and reads what the run left.
    hook_budget: Option<i64>,
    /// Advances on every flush.
    epoch: u64,
    next_trace_id: u32,
    /// Number of loop-header consults, so tests can see the dispatch point.
    consults: u64,
    /// Liveness per chunk address for the recorder. The weak handle keeps the
    /// address from being reused while the entry lives.
    liveness: FxHashMap<usize, (Weak<Chunk>, SharedLiveness)>,
    /// The guard the recording in progress continues, when it is a side
    /// recording. `recorder` is set exactly when this or a root recording is
    /// in progress.
    side_target: Option<SideTarget>,
    /// A guard that just reached the exit count at which a side recording
    /// starts, set by an exit and taken by the entry that ran it.
    side_candidate: Option<native::SideCandidate>,
    /// For a trace that made a host call itself and must leave with the call
    /// complete (see "Calling host functions"): the pc after the call and
    /// the pc of the call. The exit takes the frame as the call left it, in
    /// place of a restore.
    vmcall_landed: Option<(usize, usize)>,
    /// The error a native exit met building a virtual table, until the VM
    /// takes it.
    restore_error: Option<crate::runtime::RuntimeError>,
    /// The profile of the latest recording, which side recordings reuse.
    profile: Option<CompatibilityProfile>,
    /// Whether compiled code counts its jumps into side traces. Costs a few
    /// instructions per jump, so only runs that report statistics (and
    /// tests) ask for it.
    count_links: bool,
    /// The optimizations new traces may get, from the VM's
    /// [`oximond::compat::TracingOptLevel`].
    opt_flags: crate::compat::TracingOptFlags,
    /// The tier new traces compile at. Tests that check the unoptimized
    /// lowering guard by guard select `Baseline`.
    tier: OptimizationTier,
    /// Whether recordings leave a branch guard inside a call through a
    /// snapshot of the callee's frames. Only tests turn it on.
    rebuild_callee_frames: bool,
    /// Whether a loop of a top-level chunk whose remaining work cannot repay
    /// a compile is left uncompiled (see [`Self::runs_too_short`]). The VM's
    /// unit tests run small hand-assembled loops as top-level chunks to test
    /// what traces do, so it is off in them.
    judge_one_shot_loops: bool,
    /// Whether the root of a short inner loop with one loop around it is
    /// installed as a placeholder while the sides of its tree are recorded in
    /// the interpreter (see [`Self::stage_root`]). Off, the root compiles at
    /// the baseline tier as a stepping stone, as it does for deeper nests.
    stage_roots: bool,
    /// Whether a side recorded for an installed tree is compiled as a
    /// function of its own that the exit it continues calls, when the tree
    /// can take it that way (see [`native::InstalledTrace::plan_detached`]).
    /// Off, every side joins its tree in a compile of the whole tree.
    detach_sides: bool,
    /// Whether a small tree the MIR backend compiled takes its sides in a
    /// compile of the whole tree (see `InstalledTrace::joins_for_less`).
    join_small_trees: bool,
    /// Whether recordings turn pure two-arm branches into value selects
    /// (see "Value selects" above). Off, every branch keeps its guard.
    value_selects: bool,
    /// Whether a side compiled apart may link back to the loop when it
    /// writes registers the root does not carry as loop parameters: the run
    /// writes them into the entry inputs and the VM into the frames when it
    /// leaves (see [`oximond::jit_ir::trace::apart_link_writes`]). Off, such a
    /// side leaves at the loop header, as before sides could do that.
    detach_link_writes: bool,
    /// Whether a site lists registers its entries found nil on some entries
    /// and a number on others, for its next root to take untyped (see
    /// `note_lazy`). On unless `OXIMOND_JIT_UNTYPED=0`.
    untyped_inputs: bool,
}

/// Whether the run `closed` was recorded in has fewer iterations left than a
/// trace needs per entry (see [`native::min_iterations_of`]). A prepared
/// pattern loop like that keeps the helper-bound wait: its code would run a
/// few iterations per entry, and the entries cost more than the interpreter's
/// own pattern paths. Measured on `string/pattern_hot` (scale 3, level 1,
/// `instructions:u`, load 4): 549.7M with this clause, which is the level 0
/// count, and 616.8M without it (+12%).
#[allow(clippy::cast_precision_loss)]
fn runs_per_entry_too_short(closed: &ClosedTrace) -> bool {
    closed
        .remaining_trips
        .is_some_and(|remaining| remaining + 1.0 < native::min_iterations_of(&closed.ir) as f64)
}

impl TracePolicy {
    /// Builds the policy for an execution mode. Interpreter execution never
    /// traces.
    pub(crate) fn new(mode: ExecutionMode) -> Self {
        let enabled = matches!(mode, ExecutionMode::Tracing) && cfg!(not(target_arch = "wasm32"));
        Self {
            enabled,
            trace_profile: profile::ProfileState::default(),
            recorder: None,
            sites: SiteTable::default(),
            queue: VecDeque::new(),
            compile: true,
            cache: TraceCache::default(),
            record_limits: RecordLimits::default(),
            max_sides: MAX_SIDE_TRACES,
            hot_threshold: TRACE_HOT_THRESHOLD,
            outbox: Vec::new(),
            pending: FxHashMap::default(),
            broker_full_skips: 0,
            inputs: Vec::new(),
            binding_views: Vec::new(),
            probe_scratch: crate::runtime::NativeStringProbeScratch::default(),
            entry_reads: Box::default(),
            lane_index: Box::default(),
            table_space: Box::new(crate::jit::TraceTableSpace::new()),
            table_lanes: Box::default(),
            exit_area: Vec::new(),
            entered_recently: false,
            bank: Vec::new(),
            in_flight: false,
            resume_vm: 0,
            hooks_installed: false,
            hook_budget: None,
            epoch: 0,
            next_trace_id: 0,
            consults: 0,
            liveness: FxHashMap::default(),
            side_target: None,
            side_candidate: None,
            vmcall_landed: None,
            restore_error: None,
            profile: None,
            count_links: cfg!(test) || debug_env::stats(),
            opt_flags: crate::compat::TracingOptFlags::default(),
            tier: OptimizationTier::Optimized,
            rebuild_callee_frames: false,
            judge_one_shot_loops: !cfg!(test),
            stage_roots: true,
            detach_sides: !debug_env::detach_disabled(),
            join_small_trees: debug_env::joins_small_trees(),
            value_selects: !debug_env::selects_disabled(),
            detach_link_writes: true,
            untyped_inputs: debug_env::untyped_enabled(),
        }
    }

    /// Selects the tier new traces compile at, for tests.
    #[cfg(test)]
    pub(crate) const fn set_compile_tier(&mut self, tier: OptimizationTier) {
        self.tier = tier;
    }

    /// Makes recordings leave branch guards inside calls through snapshots of
    /// the callee's frames, for tests of the exits that rebuild them.
    #[cfg(test)]
    pub(crate) const fn set_rebuild_callee_frames(&mut self, rebuild: bool) {
        self.rebuild_callee_frames = rebuild;
    }

    /// Whether a loop of a top-level chunk that cannot repay a compile is
    /// left to the interpreter (see [`Self::runs_too_short`]).
    pub(crate) const fn set_judge_one_shot_loops(&mut self, judge: bool) {
        self.judge_one_shot_loops = judge;
    }

    /// Whether a stepping stone's tree is recorded ahead of its compile (see
    /// [`Self::stage_root`]). Tests of the stepping stone turn it off.
    #[cfg(test)]
    pub(crate) const fn set_stage_roots(&mut self, stage: bool) {
        self.stage_roots = stage;
    }

    /// Turns value selects on or off, for tests whose subject is the branch
    /// guards and sides of two-arm branches.
    pub(crate) const fn set_value_selects(&mut self, select: bool) {
        self.value_selects = select;
    }

    /// Turns untyped entry inputs for nil-or-number registers on or off.
    pub(crate) const fn set_untyped_inputs(&mut self, untyped: bool) {
        self.untyped_inputs = untyped;
    }

    /// Turns off the rule that a small tree takes its sides in a compile of
    /// the whole tree: every side the policy allows is compiled apart.
    pub(crate) const fn keep_sides_apart(&mut self) {
        self.join_small_trees = false;
    }

    /// Turns the compile of sides apart from their trees on or off, for tests.
    #[cfg(test)]
    pub(crate) const fn set_detach_sides(&mut self, detach: bool) {
        self.detach_sides = detach;
    }

    /// Lets sides compiled apart link back with writes the root does not carry,
    /// for tests.
    #[cfg(test)]
    pub(crate) const fn set_detach_link_writes(&mut self, allow: bool) {
        self.detach_link_writes = allow;
    }

    /// Turns link counting on or off for code compiled from now on.
    pub(crate) const fn set_count_links(&mut self, count: bool) {
        self.count_links = count;
    }

    /// Lets traces call host C functions themselves, or not (see
    /// `OXIMOND_JIT_NO_VMCALL`).
    pub(crate) const fn set_vmcall(&mut self, enabled: bool) {
        self.record_limits.vmcall = enabled;
    }

    /// Sets the guest instructions a loop needs for its trace to call host C
    /// functions itself (see `VMCALL_MIN_GUEST`).
    pub(crate) const fn set_vmcall_work(&mut self, work: u32) {
        self.record_limits.vmcall_work = work;
    }

    /// Applies the limits of `jit.opt.start` to recordings and compiles from
    /// now on. The caller flushes the traces recorded under other limits.
    pub(crate) fn set_jit_options(&mut self, options: crate::runtime::JitOptions) {
        let clamp = |value: u32| usize::try_from(value).unwrap_or(usize::MAX);
        if let Some(record) = options.max_record {
            self.record_limits = self.record_limits.with_record(clamp(record.max(1)));
        }
        if let Some(snap) = options.max_snap {
            self.record_limits.snapshots = clamp(snap).max(4);
        }
        if let Some(side) = options.max_side {
            self.max_sides = clamp(side);
        }
        if let Some(hot) = options.hot_loop {
            self.hot_threshold = hot.max(1);
        }
        let mut limits = self.cache.limits();
        if let Some(trace) = options.max_trace {
            limits.traces = clamp(trace).clamp(1, MAX_CONFIGURED_TRACES);
        }
        if let Some(kilobytes) = options.max_mcode {
            limits.code_bytes = clamp(kilobytes)
                .saturating_mul(1024)
                .clamp(MIN_CONFIGURED_CODE_BYTES, MAX_CONFIGURED_CODE_BYTES);
        }
        self.cache.set_limits(limits);
        if debug_env::trace() {
            eprintln!(
                "jit-trace: jit.opt.start set {options:?}: record {:?}, sides {}, hot {}, cache {:?}",
                self.record_limits, self.max_sides, self.hot_threshold, limits
            );
        }
    }

    /// Selects the optimizations for code compiled from now on.
    pub(crate) const fn set_opt_flags(&mut self, flags: crate::compat::TracingOptFlags) {
        self.opt_flags = flags;
    }

    /// The optimizations code compiled from now on may get.
    pub(crate) const fn opt_flags(&self) -> crate::compat::TracingOptFlags {
        self.opt_flags
    }

    /// What native code compiled for this policy counts and optimizes.
    pub(crate) fn compile_options(&self) -> crate::jit::TraceCompileOptions {
        // A trace may call a helper of ours: the backend needs its address.
        native::install_helpers();
        crate::jit::TraceCompileOptions {
            count_links: self.count_links,
            opt_flags: self.opt_flags,
        }
    }

    /// How many times the interpreter consulted the policy at a loop header.
    #[allow(dead_code)]
    pub(crate) const fn consults(&self) -> u64 {
        self.consults
    }

    /// Whether the policy may record or run traces.
    pub(crate) const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Whether every loop inside the loop that the branch at `latch` closes at
    /// `header` is rejected (a loop with none counts). The skip a rejected
    /// loop arms while no trace is installed swallows the backedges of the
    /// loops inside it, so another state there is still counting or waiting
    /// on them.
    pub(crate) fn inner_loops_rejected(
        &mut self,
        chunk: &Chunk,
        header: usize,
        latch: usize,
    ) -> bool {
        Recorder::inner_loop_headers(chunk, header, latch).all(|inner| {
            self.sites
                .existing_mut_at(chunk, inner)
                .is_some_and(|site| site.state == SiteState::Rejected)
        })
    }

    /// Whether a loop around the loop that the branch at `latch` closes at
    /// `header` has no verdict yet (it is still counting backedges, waiting
    /// out a cooldown, or never reached a lookup), and `header` has not spent
    /// its [`policy::UNSKIPPED_LOOKUPS`]. Spends one. Whatever turned `header`
    /// away (too few trips, or a call no recording takes), the loop around
    /// it may still record a root that ends in front of it (see "Outer
    /// roots"), so it gets to count.
    pub(crate) fn enclosing_loop_unsettled(
        &mut self,
        chunk: &Chunk,
        header: usize,
        latch: usize,
    ) -> bool {
        let spent = self
            .sites
            .existing_mut_at(chunk, header)
            .is_none_or(|site| site.unskipped >= policy::UNSKIPPED_LOOKUPS);
        if spent {
            return false;
        }
        let unsettled = nest::enclosing_loops(chunk, nest::LoopSpan { header, latch })
            .iter()
            .any(|outer| {
                self.sites
                    .existing_mut_at(chunk, outer.header)
                    .is_none_or(|site| {
                        matches!(
                            site.state,
                            SiteState::Cold { .. } | SiteState::Cooldown { .. }
                        )
                    })
            });
        if unsettled && let Some(site) = self.sites.existing_mut_at(chunk, header) {
            site.unskipped += 1;
        }
        unsettled
    }

    /// Whether the tracer rejected the function entry of the chunk at
    /// address `chunk`, as far as the memo of
    /// [`Self::remember_rejected_entry`] knows. Its self calls skip the
    /// consult.
    #[inline]
    pub(crate) fn entry_rejected(&self, chunk: usize) -> bool {
        self.sites.is_rejected_entry(chunk)
    }

    /// Remembers `chunk` for [`Self::entry_rejected`] if the tracer rejected
    /// its function entry. Called after a self call's consult.
    pub(crate) fn remember_rejected_entry(&mut self, chunk: *const Chunk) {
        self.sites.remember_rejected_entry(chunk);
    }

    /// Whether the tracer rejected the header at pc 0 of the chunk at
    /// address `chunk`, as far as the memo of [`Self::remember_rejected_tail`]
    /// knows. A tail call that lands there skips the lookup.
    #[inline]
    pub(crate) fn tail_header_rejected(&self, chunk: usize) -> bool {
        self.sites.is_rejected_tail(chunk)
    }

    /// Remembers `chunk` for [`Self::tail_header_rejected`] if the tracer
    /// rejected its header at pc 0. Called after a tail call landing there
    /// found nothing to enter or record.
    pub(crate) fn remember_rejected_tail(&mut self, chunk: *const Chunk) {
        self.sites.remember_rejected_tail(chunk);
    }

    /// Whether any trace is installed.
    pub(crate) const fn traces_live(&self) -> bool {
        self.cache.live() != 0
    }

    /// Whether a recording is in progress.
    #[inline(always)]
    pub(crate) const fn recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// Whether the recorder is sleeping through a host call and the
    /// instruction about to run belongs to code that call runs (a Lua
    /// function it calls back), `depth` host boundaries deep. The recorder
    /// has nothing to do with it.
    #[inline]
    pub(crate) fn skips_nested_host(&self, depth: usize) -> bool {
        self.recorder
            .as_ref()
            .is_some_and(|recorder| recorder.skips_nested_host(depth))
    }

    /// Hands the recorder the number of host boundaries the call it sleeps
    /// through is made at, once it has just seen the call.
    #[inline]
    pub(crate) fn latch_host_depth(&mut self, depth: usize) {
        if let Some(recorder) = self.recorder.as_mut() {
            recorder.latch_host_depth(depth);
        }
    }

    /// Records that a host call a trace made is complete and the trace must
    /// leave: the frame stands at `resume`, the call was at `call`.
    pub(crate) fn set_vmcall_landed(&mut self, resume: usize, call: usize) {
        self.vmcall_landed = Some((resume, call));
    }

    /// As [`Self::set_vmcall_landed`], for a call that raised an error: the
    /// run ends as a failure, and the entry takes the error from the VM.
    pub(crate) fn fail_vmcall_landed(&mut self, resume: usize, call: usize) {
        self.vmcall_landed = Some((resume, call));
        self.restore_error = Some(crate::runtime::RuntimeError::message("a host call failed"));
    }

    /// Traces that closed and wait for compilation, oldest first.
    #[allow(dead_code)]
    pub(crate) const fn queued(&self) -> &VecDeque<QueuedTrace> {
        &self.queue
    }

    /// The longest window of skipped backedges the header at `pc` may earn:
    /// the check of a covered header comes every [`COVERED_WINDOW`] backedges,
    /// so a window of that length makes the next consult the check, and a
    /// header in a cooldown must be consulted when the cooldown ends. Any
    /// other header the policy skips may skip the longest windows.
    pub(crate) fn skip_window_cap(&mut self, chunk: &Arc<Chunk>, pc: usize) -> u32 {
        match self.sites.existing_mut(chunk, pc).map(|site| site.state) {
            Some(SiteState::Covered) => COVERED_WINDOW,
            // A window that ends with the cooldown, not past it.
            Some(SiteState::Cooldown { remaining }) => remaining.saturating_sub(1).max(1),
            _ => u32::MAX,
        }
    }

    /// Counts `skipped` backedges the fast path will not report for the
    /// header at `pc`, because the tracer rejected it (or a long cooldown
    /// stands for a rejection, or a tree covers it) and the VM armed a window
    /// of `jit.rejected_skips`. A site still warming up advances as if it had
    /// seen them.
    pub(crate) fn credit_skipped_backedges(&mut self, chunk: &Arc<Chunk>, pc: usize, skipped: u32) {
        if !self.enabled {
            return;
        }
        if let Some(site) = self.sites.existing_mut(chunk, pc) {
            match &mut site.state {
                SiteState::Cold { backedges } => {
                    *backedges = backedges.saturating_add(skipped);
                }
                SiteState::Cooldown { remaining } => {
                    *remaining = remaining.saturating_sub(skipped).max(1);
                }
                // The outer iterations the tree did not run, which its next
                // check weighs.
                SiteState::Covered => {
                    site.cover_consults = site.cover_consults.saturating_add(skipped);
                }
                _ => {}
            }
        }
    }

    /// Whether a trace ran since the last time this was asked, and forgets
    /// it. The VM asks before it arms the global backedge skip, which a
    /// recent native run cancels.
    pub(crate) fn take_recent_entry(&mut self) -> bool {
        std::mem::take(&mut self.entered_recently)
    }

    /// The state of the only site at `pc`, for tests.
    #[cfg(test)]
    pub(crate) fn site_state_any(&self, pc: usize) -> Option<SiteState> {
        self.sites.state_at_pc(pc)
    }

    /// The state of a site, for tests.
    #[cfg(test)]
    pub(crate) fn site_state(&self, chunk: &Arc<Chunk>, pc: usize) -> Option<SiteState> {
        self.sites.state(chunk, pc)
    }

    /// The answer of [`Self::at_loop_header`] for the header at `pc` of
    /// `chunk` if it is one recently found rejected, or `None`. Either answer
    /// changes nothing but the consult count, so the fast loop settles it in
    /// place instead of leaving for the lookup.
    #[inline]
    pub(crate) fn rejected_header_dispatch(
        &mut self,
        chunk: &Chunk,
        pc: usize,
    ) -> Option<TraceDispatch> {
        if !self.enabled || self.recorder.is_some() || !self.sites.is_last_rejected_at(chunk, pc) {
            return None;
        }
        self.consults += 1;
        Some(TraceDispatch::Skip)
    }

    /// The consult of [`Self::at_loop_header`] for a header whose site can
    /// only count this backedge and answer `Interpret`: a cold header below
    /// the hot threshold, a cooldown that is not over (unless a long one on a
    /// leaf loop would be spent as a rejection), or a trace that waits for
    /// its first compile. The fast loop then resumes in place. `None`
    /// changes nothing and leaves the header to the full consult, which also
    /// creates sites, starts recordings, enters traces, and finds out whether
    /// a loop is a leaf. `Some(skip)` is the number of the next backedges the
    /// caller may skip: the header only waits for its compile.
    pub(crate) fn count_backedge(&mut self, chunk: *const Chunk, pc: usize) -> Option<u32> {
        if !self.enabled || self.recorder.is_some() {
            return None;
        }
        let traces_live = self.cache.live() != 0;
        let site = self.sites.existing_mut_at(chunk, pc)?;
        let skip = match &mut site.state {
            SiteState::Cold { backedges } if backedges.saturating_add(1) < self.hot_threshold => {
                *backedges += 1;
                0
            }
            SiteState::Cooldown { remaining }
                if *remaining > 1
                    && (*remaining <= LONG_COOLDOWN
                        || site.leaf == Some(false)
                        || (site.leaf == Some(true) && traces_live)) =>
            {
                *remaining -= 1;
                0
            }
            // Nothing but the end of the compile changes this. A queued
            // trace waits for nothing (native compilation is off).
            SiteState::Queued => 0,
            // The skip is global: armed in a callee's loop, it runs on into
            // the caller's backedges. With a trace installed it would delay
            // that trace's entries, and an outer trace entered late in every
            // period fails its productivity window while this compile is in
            // flight. So only skip while nothing is installed.
            SiteState::Compiling if site.head.is_none() => {
                if traces_live {
                    0
                } else {
                    COMPILE_WAIT_SKIP
                }
            }
            _ => return None,
        };
        self.consults += 1;
        Some(skip)
    }

    /// Called when the interpreter reaches a loop header (a backward step to
    /// `header.pc`). Counts the backedge and decides who owns the header.
    pub(crate) fn at_loop_header(
        &mut self,
        stats: &mut TraceStatistics,
        header: &LoopHeader<'_>,
    ) -> TraceDispatch {
        self.consults += 1;
        let fallback = TraceDispatch::Interpret;
        // A header reached mid-recording is the closing visit or a branch
        // the recorder already judged.
        if self.recorder.is_some() {
            return TraceDispatch::Interpret;
        }
        // A rejected header costs one pointer compare, not a hash lookup.
        if self.sites.is_last_rejected(header.chunk, header.pc) {
            return TraceDispatch::Skip;
        }
        let Some(site) = self.sites.site_mut(header.chunk, header.pc) else {
            return fallback;
        };
        match site.state {
            SiteState::Cold { backedges } => {
                let backedges = backedges.saturating_add(1);
                site.state = SiteState::Cold { backedges };
                if backedges
                    < if self.trace_profile.recording {
                        1
                    } else {
                        self.hot_threshold
                    }
                {
                    return fallback;
                }
            }
            SiteState::Cooldown { remaining } if remaining > 1 => {
                site.state = SiteState::Cooldown {
                    remaining: remaining - 1,
                };
                // A long wait on a loop with nothing inside it (a few thousand
                // backedges, for a table that is still filling) is as cheap
                // to spend as a rejection: the skipped backedges count toward
                // it, and nothing inside the loop starves for the skip.
                if remaining > LONG_COOLDOWN {
                    let leaf = *site.leaf.get_or_insert_with(|| {
                        !Recorder::has_inner_loop(header.chunk, header.pc, header.latch)
                    });
                    if leaf {
                        return TraceDispatch::Skip;
                    }
                }
                return fallback;
            }
            SiteState::Cooldown { .. } => {}
            SiteState::Installed { index } => {
                if !site.rerecord {
                    return TraceDispatch::Enter(index);
                }
                // Another version for the types the header sees now.
                site.rerecord = false;
            }
            // The existing versions keep running while a new one compiles.
            SiteState::Compiling => {
                if let Some(head) = site.head {
                    return TraceDispatch::Enter(head);
                }
                return fallback;
            }
            SiteState::Queued => return fallback,
            SiteState::Covered => {
                site.cover_consults += 1;
                if site.cover_consults < COVERED_WINDOW {
                    // The backedges up to the next check run unwatched.
                    return TraceDispatch::Skip;
                }
                let (index, mark) = (site.cover_index, site.cover_mark);
                let iterations = self
                    .cache
                    .get(index)
                    .map(native::InstalledTrace::iterations);
                let bar = if self
                    .cache
                    .get(index)
                    .is_some_and(native::InstalledTrace::integers)
                {
                    COVERED_MIN_ITERATIONS
                } else {
                    COVERED_MIN_ITERATIONS_FLOAT
                };
                let carried = iterations
                    .is_some_and(|now| now.saturating_sub(mark) >= u64::from(COVERED_WINDOW) * bar);
                // The tree is still growing the sides that make it carry the
                // loop (a loop with several exits needs one side per exit,
                // and each takes a recording and a compile of milliseconds,
                // which are many outer iterations). The iterations that came
                // back meanwhile say nothing about the finished tree.
                let growing = !carried
                    && self
                        .cache
                        .get(index)
                        .is_some_and(native::InstalledTrace::outer_side_pending);
                let mut own_turn = false;
                if let Some(site) = self.sites.existing_mut(header.chunk, header.pc) {
                    if carried || growing {
                        site.cover_consults = 0;
                        site.cover_mark = iterations.unwrap_or(0);
                        return TraceDispatch::Interpret;
                    }
                    site.cover_revoked = true;
                    // A loop the tracer never judged on its own (the tree
                    // covered it before it got hot) gets its own trace: an
                    // outer loop whose iterations rarely run the inner one
                    // is a loop like any other.
                    own_turn = !site.covered_rejected;
                }
                // Most outer iterations came back through the interpreter:
                // the tree does not run this loop.
                if debug_env::trace() {
                    eprintln!(
                        "jit-trace: covered outer loop at pc {} is not carried by its tree, {}",
                        header.pc,
                        if own_turn {
                            "counting again"
                        } else {
                            "rejected"
                        }
                    );
                }
                self.forget_covered(index, header.pc);
                stats.covered_revoked += 1;
                stats.uncovered_loops += 1;
                if own_turn {
                    if let Some(site) = self.sites.existing_mut(header.chunk, header.pc) {
                        site.state = SiteState::Cold { backedges: 0 };
                    }
                    return TraceDispatch::Interpret;
                }
                stats.negative_evictions +=
                    self.sites
                        .reject(header.chunk, header.pc, RejectCause::Uncovered);
                return TraceDispatch::Skip;
            }
            SiteState::Rejected => {
                self.sites.remember_rejected(header.chunk, header.pc);
                return TraceDispatch::Skip;
            }
            // No recorder is running, so a stale marker counts as cold.
            SiteState::Recording => {
                site.state = site.head.map_or(SiteState::Cold { backedges: 0 }, |index| {
                    SiteState::Installed { index }
                });
                return fallback;
            }
        }
        self.begin_recording(stats, header);
        TraceDispatch::Interpret
    }

    /// The shared liveness cache of `chunk`, creating it on first use.
    fn liveness_of(&mut self, chunk: &Arc<Chunk>) -> SharedLiveness {
        if self.liveness.len() >= MAX_LIVENESS_CHUNKS {
            self.liveness
                .retain(|_, (chunk, _)| chunk.strong_count() > 0);
            if self.liveness.len() >= MAX_LIVENESS_CHUNKS {
                self.liveness.clear();
            }
        }
        Arc::clone(
            &self
                .liveness
                .entry(policy::chunk_address(chunk))
                .or_insert_with(|| (Arc::downgrade(chunk), SharedLiveness::default()))
                .1,
        )
    }

    fn begin_recording(&mut self, stats: &mut TraceStatistics, header: &LoopHeader<'_>) {
        let _phase = alloc_phase::enter(Phase::Recording);
        if self.broker_full_skips > 0 {
            self.broker_full_skips -= 1;
            self.back_off_refused(header.chunk, header.pc, QUEUE_FULL_COOLDOWN);
            return;
        }
        let Some(site) = self.sites.site_mut(header.chunk, header.pc) else {
            return;
        };
        // A function's entry records from pc 0 of the frame its self call
        // pushed, which needs the closure the self calls must call.
        let function = match (header.function, header.closure) {
            (false, _) => None,
            (true, Some(closure)) => Some(closure),
            (true, None) => return,
        };
        let unroll_loops = site.inner_loop_met && !site.unroll_overflowed;
        let callee_loop = site.callee_loop.clone().filter(|_| function.is_none());
        let rowless = site.rowless;
        let no_vmcall = site.no_vmcall;
        let lane_fields = site.lane_fields;
        let poly_calls = site.poly_calls.clone();
        let keep_guards = site.keep_guards;
        let lazy = site.lazy.clone();
        if debug_env::trace() {
            eprintln!(
                "jit-trace: recording the loop at pc {}{} (unrolling short loops: {unroll_loops}, untyped registers: {lazy:?})",
                header.pc,
                debug_env::at_line(header.chunk, header.pc)
            );
        }
        stats.recording_starts += 1;
        site.state = SiteState::Recording;
        // A recording that unrolls short loops follows the `while` jumps of
        // the loops of the root frame that cannot have a productive root of
        // their own. A loop with one (or none yet) is left to that root,
        // which the recording ends at.
        let short_loops = if unroll_loops && function.is_none() {
            self.short_loops_of(header.chunk)
        } else {
            Vec::new()
        };
        // The loops inside that no tree of their own carries: the recording
        // ends in front of the first one it enters, which becomes a loop
        // side of this root's tree (see "Outer roots").
        let free_loops = if function.is_none() && !self.hooks_installed {
            self.free_loops_of(header.chunk, header.pc, header.latch)
        } else {
            Vec::new()
        };
        self.profile = Some(header.profile);
        let liveness = self.liveness_of(header.chunk);
        let recorder = Recorder::new(
            Arc::clone(header.chunk),
            header.depth,
            if function.is_some() { 0 } else { header.pc },
            header.latch,
            header.coroutine,
            header.profile,
        );
        let recorder = match function {
            Some(closure) => recorder.for_function(closure),
            None => recorder,
        }
        .with_loop_unrolling(unroll_loops)
        .with_short_loops(short_loops)
        .with_free_loops(free_loops)
        .with_callee_loop(callee_loop)
        .with_rowless(rowless)
        .with_no_vmcall(no_vmcall)
        .with_lane_fields(lane_fields)
        .with_poly_calls(poly_calls)
        .with_kept_guards(keep_guards)
        .with_lazy(lazy)
        .with_limits(self.record_limits);
        self.recorder = Some(Box::new(
            recorder
                .with_liveness(liveness)
                .with_counting(self.hooks_installed)
                .with_home_closure(header.closure)
                .with_rebuilt_callee_frames(self.rebuild_callee_frames)
                .with_value_selects(self.value_selects)
                .with_captured(
                    header
                        .captured
                        .iter()
                        .filter_map(|open| {
                            Some((
                                u8::try_from(open.slot - header.captured_base).ok()?,
                                open.cell,
                            ))
                        })
                        .collect(),
                ),
        ));
    }

    /// The deepest call depth a recording still hooks (see
    /// [`Recorder::hook_depth_limit`]): deeper instructions would only be
    /// skipped.
    pub(crate) fn record_depth_limit(&self) -> usize {
        self.recorder
            .as_ref()
            .map_or(usize::MAX, |recorder| recorder.hook_depth_limit())
    }

    /// Called before the interpreter executes an instruction while a
    /// recording is in progress. Returns whether the recording continues. When
    /// it returns `false` the recording is over and the instruction runs
    /// unobserved.
    pub(crate) fn record(&mut self, stats: &mut TraceStatistics, at: &HookPoint<'_>) -> bool {
        let _icount = crate::jit::icount::scope("record");
        let _phase = alloc_phase::enter(Phase::Recording);
        let id = TraceId(self.next_trace_id);
        let Some(recorder) = self.recorder.as_mut() else {
            return false;
        };
        match recorder.step(at, id) {
            Step::Skipped => true,
            Step::Recorded => {
                match at.source {
                    HookSource::Fast => stats.fast_path_records += 1,
                    HookSource::General => stats.general_path_records += 1,
                }
                // A tail-call cycle that lands in a chunk whose header
                // already has a trace is that trace rotated: the installed
                // one runs the whole cycle, and the interpreter enters it at
                // its own header first, so this recording would be compiled
                // and never entered. Only a step that entered a new
                // activation can land there.
                if let Some(recorder) = self.recorder.as_mut()
                    && recorder.take_entered_activation()
                    && self.side_target.is_none()
                    && let Some(recorder) = self.recorder.as_ref()
                    && recorder.tail_call_chunks().any(|chunk| {
                        self.sites
                            .existing_mut(chunk, 0)
                            .is_some_and(|site| site.head.is_some())
                    })
                {
                    if debug_env::trace() {
                        eprintln!(
                            "jit-trace: the cycle lands in a chunk with a trace, aborted at pc {}",
                            at.pc
                        );
                    }
                    self.abort_recording_at(stats, TraceAbortReason::RootTailCall, Some(at));
                    return false;
                }
                true
            }
            Step::Closed(closed) => {
                let (chunk, header) = self
                    .side_target
                    .and_then(|target| {
                        self.cache
                            .get(target.index)
                            .and_then(native::InstalledTrace::profile_site)
                    })
                    .unwrap_or_else(|| (Arc::clone(at.chunk), closed.header_pc));
                let closure = self
                    .recorder
                    .as_ref()
                    .and_then(|recorder| recorder.home_closure());
                self.capture_profile(
                    at,
                    &chunk,
                    header,
                    &closed.ir,
                    closure,
                    self.side_target.is_none(),
                );
                self.finish_closed(stats, *closed, at);
                // A recording that began behind the one that just closed
                // takes this very instruction as its first.
                if self.recorder.is_some() {
                    return self.record(stats, at);
                }
                if self.trace_profile.recording
                    && let Some(around) = at.surroundings
                {
                    let trees: Vec<_> = self
                        .outbox
                        .iter()
                        .filter_map(|request| {
                            let (chunk, header) = match request.target {
                                CompileTarget::New {
                                    ref chunk,
                                    header_pc,
                                } => (chunk.upgrade()?, header_pc),
                                CompileTarget::Retree { index, .. }
                                | CompileTarget::Side { index, .. } => {
                                    self.cache.get(index)?.profile_site()?
                                }
                            };
                            Some((chunk, header, Arc::clone(&request.ir)))
                        })
                        .collect();
                    for (chunk, header, ir) in trees {
                        self.remember_profile_tree(&chunk, header, &ir, around.runtime);
                    }
                }
                false
            }
            Step::Wake(wake) => {
                let scout = self.recorder.take();
                let started = scout
                    .as_ref()
                    .is_some_and(|scout| self.wake_side(scout, wake));
                if started {
                    // The new recording takes this very instruction first.
                    return self.record(stats, at);
                }
                // Nothing to record after all: the sides queued so far
                // compile without it.
                if let Some(target) = self.side_target.take() {
                    self.abort_side(stats, target, TraceAbortReason::Divergence, false, false);
                }
                false
            }
            Step::Abort(reason) => {
                if let Some(callable) = self
                    .recorder
                    .as_mut()
                    .and_then(|recorder| recorder.take_abort_callable())
                {
                    stats.native_aborts.add(&callable_name(at, &callable));
                }
                if let Some(name) = self
                    .recorder
                    .as_mut()
                    .and_then(|recorder| recorder.take_abort_metamethod())
                {
                    stats.native_aborts.add(name);
                }
                // A recording that failed on a table still filling up says
                // nothing about the next attempt at this instruction.
                let waiting = reason == TraceAbortReason::TableMiss
                    && self
                        .recorder
                        .as_ref()
                        .is_some_and(|recorder| recorder.waits_for_table());
                let root = self.recorder.as_ref().map(|recorder| recorder.site_pc());
                let side = self.side_target.is_some();
                let restarted = self.restart_recording(stats, reason, at);
                if debug_env::trace() {
                    eprintln!(
                        "jit-trace: {}recording of the loop at pc {root:?} {} at pc {} ({:?}) {}:{}: {reason:?}",
                        if side { "side " } else { "" },
                        if restarted { "restarted" } else { "aborted" },
                        at.pc,
                        at.chunk
                            .code
                            .get(at.pc)
                            .map(|instruction| instruction.opcode()),
                        String::from_utf8_lossy(at.chunk.debug.source_name().as_bytes()),
                        at.chunk.debug.line_at(at.pc).unwrap_or(0),
                    );
                }
                if !restarted {
                    self.abort_recording_at(stats, reason, (!waiting).then_some(at));
                }
                false
            }
        }
    }

    /// A recording that unrolled loops ran out of room: the nest is too big
    /// for one trace, so the tree records again without unrolling. That is a
    /// restart, not an abort of the loop: it is not counted and spends none
    /// of the attempts of the site or the side. Returns whether it did.
    fn restart_unroll_overflow(
        &mut self,
        stats: &mut TraceStatistics,
        reason: TraceAbortReason,
    ) -> bool {
        let Some(recorder) = self.recorder.as_ref() else {
            return false;
        };
        if reason != TraceAbortReason::LimitExceeded || !recorder.has_unrolled_loops() {
            return false;
        }
        let (chunk, header) = (Arc::clone(recorder.chunk()), recorder.site_pc());
        let Some(site) = self.sites.existing_mut(&chunk, header) else {
            return false;
        };
        site.unroll_overflowed = true;
        stats.unroll_restarts += 1;
        if let Some(target) = self.side_target.take() {
            self.recorder = None;
            // Flushed: the side tries again at once, without unrolling.
            self.abort_side(stats, target, TraceAbortReason::Flushed, false, false);
            stats.side_aborts = stats.side_aborts.saturating_sub(1);
            return true;
        }
        if site.head.is_some() {
            return false;
        }
        site.state = SiteState::Cold {
            backedges: self.hot_threshold - 1,
        };
        stats.recording_starts = stats.recording_starts.saturating_sub(1);
        self.recorder = None;
        true
    }

    /// Drops a root recording that left its loop before it recorded a trip
    /// and lets the site record again at the next backedge, a few times in a
    /// row (see [`policy::MAX_EMPTY_PASS_RESTARTS`]). The hot count is taken
    /// at the backedge, and the last backedge of a run is the pass whose
    /// loop test leaves: a loop of two trips starts every other recording
    /// there. Such a pass says nothing about the loop, so it spends no
    /// attempt and is not counted as a start. Only a pass that leaves by the
    /// loop's test counts. Returns whether it did.
    fn restart_empty_pass(
        &mut self,
        stats: &mut TraceStatistics,
        reason: TraceAbortReason,
        at: &HookPoint<'_>,
    ) -> bool {
        // A return or the end of the chunk leaves the function, not the
        // loop, and a numeric for loop ends once per run of a count its
        // close estimates: both abort and retry as before.
        let leaves_by_test = at.chunk.code.get(at.pc).is_some_and(|instruction| {
            !AbortSite::of(at).is_return
                && !matches!(
                    instruction.opcode(),
                    Ok(crate::bytecode::OpCode::NumericForLoop)
                )
        });
        if !leaves_by_test {
            return false;
        }
        if reason != TraceAbortReason::LoopNotClosed || self.side_target.is_some() {
            return false;
        }
        let Some(recorder) = self.recorder.as_ref() else {
            return false;
        };
        if recorder.is_function()
            || recorder.in_inlined_call()
            || recorder.guest_instructions() > policy::EMPTY_PASS_INSTRUCTIONS
        {
            return false;
        }
        let (chunk, header) = (Arc::clone(recorder.chunk()), recorder.site_pc());
        let Some(site) = self.sites.existing_mut(&chunk, header) else {
            return false;
        };
        if site.empty_passes >= policy::MAX_EMPTY_PASS_RESTARTS {
            return false;
        }
        site.empty_passes += 1;
        // A header with versions asked for another one: it asks again at its
        // next consult, and keeps running the versions it has.
        site.state = match site.head {
            Some(head) => {
                site.rerecord = true;
                SiteState::Installed { index: head }
            }
            None => SiteState::Cold {
                backedges: self.hot_threshold - 1,
            },
        };
        stats.recording_starts = stats.recording_starts.saturating_sub(1);
        self.recorder = None;
        true
    }

    /// The headers of the loops of `chunk` whose own root cannot pay: it was
    /// retired twice for running too few trips per entry, or the header was
    /// given up because every recording of it ended on its last trip. A recording of a loop around them unrolls their `while` jumps.
    pub(crate) fn short_loops_of(&mut self, chunk: &Arc<Chunk>) -> Vec<usize> {
        nest::loops_of(chunk)
            .into_iter()
            .filter(|span| {
                self.sites
                    .existing_mut(chunk, span.header)
                    .is_some_and(|inner| inner.retired_short)
            })
            .map(|span| span.header)
            .collect()
    }

    /// The headers of the loops inside the loop that the branch at `latch`
    /// closes at `header` that no tree of their own carries, so that a root
    /// recording of the outer loop ends in front of them (see "Outer roots"):
    /// the inner loop's header was given up, or its root is installed (or on
    /// its way) but no outer side of its tree can run the loop around it
    /// ([`nest::can_cover`] says no, for sibling inner loops for one). An
    /// inner loop whose root may still carry the nest (one that is counting,
    /// recording, cooling down, a stepping stone, or covering the loops
    /// around it) is left to that root, and the recording aborts at it as
    /// before. So is an inner loop with loops of its own, whose loop side
    /// would abort at their backward branches on every recording, and one
    /// given up for its few trips (its last abort was `ShortLoop` or
    /// `LoopNotClosed`), whose loop side would not pay for its entry.
    pub(crate) fn free_loops_of(
        &mut self,
        chunk: &Arc<Chunk>,
        header: usize,
        latch: usize,
    ) -> Vec<usize> {
        nest::inner_loops(chunk, nest::LoopSpan { header, latch })
            .into_iter()
            .filter(|inner| {
                if !nest::inner_loops(chunk, *inner).is_empty() {
                    return false;
                }
                let Some(site) = self.sites.existing_mut(chunk, inner.header) else {
                    return false;
                };
                match site.state {
                    // A loop given up for its few trips would run a loop side
                    // that leaves after a trip or two on every outer
                    // iteration.
                    SiteState::Rejected => !site.last_abort().is_some_and(|(reason, _)| {
                        matches!(
                            reason,
                            TraceAbortReason::ShortLoop | TraceAbortReason::LoopNotClosed
                        )
                    }),
                    SiteState::Installed { .. } | SiteState::Queued | SiteState::Compiling => {
                        !nest::can_cover(chunk, *inner)
                    }
                    _ => false,
                }
            })
            .map(|inner| inner.header)
            .collect()
    }

    /// Drops a root recording that met an inner loop inside an inlined call
    /// and lets its site record again at the next iteration, a few times. The
    /// call is on a path the iteration recorded took, which says little about
    /// the next one. Returns whether it did. The dropped recording does not
    /// count as a start: the same site starts again.
    fn restart_recording(
        &mut self,
        stats: &mut TraceStatistics,
        reason: TraceAbortReason,
        at: &HookPoint<'_>,
    ) -> bool {
        if self.restart_unroll_overflow(stats, reason) || self.restart_empty_pass(stats, reason, at)
        {
            return true;
        }
        if reason != TraceAbortReason::InnerLoop || self.side_target.is_some() {
            return false;
        }
        let Some(recorder) = self.recorder.as_ref().filter(|r| r.in_inlined_call()) else {
            return false;
        };
        // A loop in the callee that no recording can unroll (a `while`, a
        // generic for, a numeric loop with many trips) meets the next
        // recording at the same call whenever the call is taken, and a call
        // the loop makes on most iterations is taken on the next one too:
        // the header is turned away at this abort instead of after two
        // restarts and the aborts that follow them.
        let site_at = AbortSite::of(at);
        if recorder.met_long_inner_loop() || !(site_at.is_counted_latch || site_at.is_while_jump) {
            return false;
        }
        let (chunk, header) = (Arc::clone(recorder.chunk()), recorder.site_pc());
        let Some(site) = self.sites.site_mut(&chunk, header) else {
            return false;
        };
        if site.head.is_some() || site.restarts >= MAX_RECORDING_RESTARTS {
            return false;
        }
        site.restarts += 1;
        site.inner_loop_met = true;
        site.state = SiteState::Cold {
            backedges: self.hot_threshold - 1,
        };
        stats.recording_starts = stats.recording_starts.saturating_sub(1);
        self.recorder = None;
        true
    }

    /// Whether the recording in progress sleeps through a resume: the thread
    /// that runs meanwhile may yield without ending the recording.
    pub(crate) fn recording_waits_for_resume(&self) -> bool {
        self.recorder
            .as_ref()
            .is_some_and(|recorder| recorder.waits_for_resume())
    }

    /// Whether the recording expects the error the interpreter is about to
    /// raise, which a protected call of the trace catches.
    pub(crate) fn recording_expects_error(&self) -> bool {
        self.recorder
            .as_ref()
            .is_some_and(|recorder| recorder.expects_error())
    }

    /// Stops the recording in progress, if any, for `reason`, and applies the
    /// site's retry policy.
    pub(crate) fn abort_recording(
        &mut self,
        stats: &mut TraceStatistics,
        reason: TraceAbortReason,
    ) {
        let _phase = alloc_phase::enter(Phase::Recording);
        self.abort_recording_at(stats, reason, None);
    }

    /// [`Self::abort_recording`] for an abort the recorder met at guest
    /// instruction `at`, which the site remembers (see
    /// [`policy::MAX_SAME_ABORTS`]).
    fn abort_recording_at(
        &mut self,
        stats: &mut TraceStatistics,
        reason: TraceAbortReason,
        at: Option<&HookPoint<'_>>,
    ) {
        let Some(recorder) = self.recorder.take() else {
            return;
        };
        let site = at.map(AbortSite::of);
        let pc = site.map(|site| site.pc);
        if let Some(target) = self.side_target.take() {
            // The side ran to the return of the root frame. The instruction
            // is the one of the frame the interpreter was running, which is
            // a callee's or the caller's when the recording had left the root
            // frame, so it is read from that frame's chunk, not the root's.
            let at_return = reason == TraceAbortReason::LoopNotClosed
                && site.is_some_and(|site| site.is_return);
            // The side ran to the root's own latch in the root frame and the
            // loop ended there: the guard it continued also fails on
            // iterations that go on (a stateful iterator's next row and its
            // last one take the same arm), so the next failure may close.
            let loop_ended = reason == TraceAbortReason::LoopNotClosed
                && !recorder.in_inlined_call()
                && site.and_then(|site| site.branch_back)
                    == Some((Arc::as_ptr(recorder.chunk()), recorder.site_pc()));
            self.abort_side(stats, target, reason, at_return, loop_ended);
            return;
        }
        let chunk = Arc::clone(recorder.chunk());
        if reason == TraceAbortReason::LimitExceeded
            && recorder.has_unrolled_loops()
            && let Some(site) = self.sites.site_mut(&chunk, recorder.site_pc())
        {
            site.unroll_overflowed = true;
        }
        if recorder.refused_rows()
            && let Some(site) = self.sites.site_mut(&chunk, recorder.site_pc())
        {
            site.rowless = true;
        }
        // The next recording aborts at a host call the trace could not make
        // itself.
        if recorder.refused_vmcall()
            && let Some(site) = self.sites.site_mut(&chunk, recorder.site_pc())
        {
            site.no_vmcall = true;
        }
        // A host call beside field bindings is recorded again with the fields
        // read through lanes. If the lane reads fail that recording (an
        // operand they cannot take), the next recording aborts at the call.
        // Any other abort says nothing about the lane reads, so the call
        // stays and the site tries the bound fields again.
        if let Some(site) = self.sites.site_mut(&chunk, recorder.site_pc()) {
            if recorder.wants_lane_fields() {
                site.lane_fields = true;
            } else if recorder.lane_fields() {
                site.lane_fields = false;
                if reason == TraceAbortReason::UnsupportedOperand {
                    site.no_vmcall = true;
                }
            }
        }
        // A frame that reads its closure's cells through the live closure
        // cannot read every value a cell holds (a string: the `id` of
        // Mineclonia's concentric ring `chunk_test`, structures.lua:404). The
        // loop goes back to identity guards and does not name a call again.
        if recorder.in_dynamic_frame()
            && let Some(site) = self.sites.site_mut(&chunk, recorder.site_pc())
            && !site.poly_calls.is_empty()
        {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the loop at pc {} aborted inside a call guarded on the prototype; its calls keep identity guards",
                    recorder.site_pc()
                );
            }
            site.poly_calls.clear();
            site.poly_refused = true;
        }
        // A loop in an inlined call that the unrolling recording could not
        // follow: the next recording ends in front of it, and the loop runs
        // as a loop side in the call's frames (see "Loops in calls"). Only a
        // loop no other loop of the callee encloses: the inner loop of a
        // nest in a call has a root of its own, whose sides link through the
        // return to the caller (see "Linking through a return").
        if reason == TraceAbortReason::InnerLoop
            && recorder.met_long_inner_loop()
            && recorder.in_inlined_call()
            && !recorder.is_function()
            && !self.hooks_installed
            && let Some(at) = at
            && let Some(header) = Recorder::loop_latch_target(at.chunk, at.pc)
            && nest::enclosing_loops(
                at.chunk,
                nest::LoopSpan {
                    header,
                    latch: at.pc,
                },
            )
            .is_empty()
            && let Some(site) = self.sites.site_mut(&chunk, recorder.site_pc())
            && site.head.is_none()
            && site.callee_loop.is_none()
        {
            site.callee_loop = Some(policy::CalleeLoop::new(at.chunk, at.pc));
            site.state = SiteState::Cold {
                backedges: self.hot_threshold - 1,
            };
            stats.recording_aborts[reason.index()] += 1;
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the loop at pc {} records again in front of the loop in a call at pc {}{}",
                    recorder.site_pc(),
                    at.pc,
                    debug_env::at_line(at.chunk, at.pc)
                );
            }
            return;
        }
        // An inner loop that is not a numeric for loop (a `while`, a generic
        // for) is not unrolled by the next recording either, so it ends the
        // retries one abort sooner, and so does a numeric one with too many
        // trips ahead to unroll.
        let mut repeats = 0;
        // A `while` loop is unrolled when it is in an inlined call, or in the
        // root frame once its own root cannot pay.
        let short_while = site.is_some_and(|site| {
            site.is_while_jump
                && (recorder.in_inlined_call()
                    || site
                        .branch_back
                        .and_then(|(met, header)| self.sites.existing_mut_at(met, header))
                        .is_some_and(|inner| inner.retired_short))
        });
        if reason == TraceAbortReason::InnerLoop
            && (recorder.met_long_inner_loop()
                || site.is_some_and(|site| !(site.is_counted_latch || short_while)))
        {
            repeats = 2;
            // A loop inside a call the recording inlined is met at that call
            // on every iteration that takes it.
            if recorder.in_inlined_call() {
                repeats = 1;
            }
            // The inner loop has a root of its own, or was given up: the next
            // recording ends at it the same way.
            let owned = site
                .and_then(|site| site.branch_back)
                .and_then(|(met, header)| self.sites.existing_mut_at(met, header))
                .is_some_and(|inner| {
                    matches!(
                        inner.state,
                        SiteState::Installed { .. } | SiteState::Rejected
                    )
                });
            if owned {
                repeats = 1;
            }
        }
        self.settle_site_as(
            stats,
            &chunk,
            recorder.site_pc(),
            reason,
            pc,
            repeats,
            at.map(|at| at.chunk),
        );
    }

    /// Applies an abort for `reason` to the site at `pc`: the retry or
    /// rejection policy. `at` is the guest instruction the recording failed
    /// at, when one did.
    fn settle_site(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        pc: usize,
        reason: TraceAbortReason,
        at: Option<usize>,
    ) {
        self.settle_site_as(stats, chunk, pc, reason, at, 0, None);
    }

    /// [`Self::settle_site`]; a non-zero `repeats` says the abort is one
    /// that a retry does not change the way of, so the `repeats`th of them
    /// at one instruction rejects the header instead of the third.
    /// `met_in` is the chunk of the frame the recording met the abort in,
    /// when it is not the site's own.
    #[allow(clippy::too_many_arguments)]
    fn settle_site_as(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        pc: usize,
        reason: TraceAbortReason,
        at: Option<usize>,
        repeats: u8,
        met_in: Option<&Arc<Chunk>>,
    ) {
        stats.recording_aborts[reason.index()] += 1;
        if reason != TraceAbortReason::Flushed {
            self.release_deferred(chunk, pc);
        }
        let Some(site) = self.sites.site_mut(chunk, pc) else {
            return;
        };
        // The same abort at the same instruction again says the loop meets
        // it on every iteration a recording starts on: retrying waits for
        // nothing.
        site.empty_passes = 0;
        let repeated =
            site.note_abort_in(reason, at, met_in) || (repeats > 0 && site.same_aborts >= repeats);
        site.inner_loop_met |= reason == TraceAbortReason::InnerLoop;
        // A header that already has versions keeps running them. Its failed
        // attempt only spends the budget for more versions.
        if let Some(head) = site.head {
            site.state = SiteState::Installed { index: head };
            match classify_abort(reason, site.attempts) {
                AbortClass::Neutral => {}
                AbortClass::Permanent => site.versions_closed = true,
                AbortClass::Retry { .. } => {
                    site.attempts = site.attempts.saturating_add(1);
                    if site.attempts >= attempt_limit(reason) || repeated {
                        site.versions_closed = true;
                    }
                }
            }
            // Another version needs a fresh streak of mismatches.
            if let Some(newest) = self.cache.get_mut(head) {
                newest.mismatch_streak = 0;
            }
            return;
        }
        let rejected = match classify_abort(reason, site.attempts) {
            AbortClass::Permanent => true,
            AbortClass::Neutral => {
                site.state = SiteState::Cold { backedges: 0 };
                false
            }
            AbortClass::Retry { cooldown } => {
                site.attempts = site.attempts.saturating_add(1);
                if site.attempts >= attempt_limit(reason) || repeated {
                    if repeated && debug_env::trace() {
                        eprintln!(
                            "jit-trace: the header at pc {pc} aborted {} times in a row with {reason:?} at the same instruction, rejected",
                            policy::MAX_SAME_ABORTS
                        );
                    }
                    true
                } else {
                    site.state = SiteState::Cooldown {
                        remaining: cooldown,
                    };
                    false
                }
            }
        };
        if rejected {
            stats.negative_evictions += self.sites.reject(chunk, pc, RejectCause::Aborts);
            // A loop whose recordings keep ending on its last trip is short.
            if reason == TraceAbortReason::LoopNotClosed {
                if let Some(site) = self.sites.existing_mut(chunk, pc) {
                    site.retired_short = true;
                }
                self.reopen_outer_loops(chunk, pc);
            }
        }
    }

    /// Lets the loops around the loop at `header` that were turned away for
    /// meeting an inner loop record again. The loop at `header` has no root
    /// that pays (it was given up, or retired as too short), so the next
    /// recording of an outer loop unrolls its `while` jumps (see
    /// [`Self::short_loops_of`]) instead of ending at it.
    pub(crate) fn reopen_outer_loops(&mut self, chunk: &Arc<Chunk>, header: usize) {
        let backedges = self.hot_threshold.saturating_sub(1);
        for outer in nest::loops_of(chunk)
            .iter()
            .filter(|span| span.header < header && span.latch > header)
        {
            self.sites
                .reopen_for_inner_loop(chunk, outer.header, backedges);
        }
    }

    /// How many times a trace like `ir` (up to its id) was retired from this
    /// header for never running a productive window.
    fn retired_repeats(&mut self, chunk: &Arc<Chunk>, header_pc: usize, ir: &TraceIr) -> usize {
        let Some(site) = self.sites.existing_mut(chunk, header_pc) else {
            return 0;
        };
        site.retired_unproductive
            .iter()
            .filter(|retired| {
                let mut same = ir.clone();
                same.id = retired.id;
                ***retired == same
            })
            .count()
    }

    /// Whether `ir` compiles to calls of prepared pattern twins only: the
    /// flag is on and every pattern call has a constant pattern (see
    /// [`native::pattern_calls_are_prepared`]). Such a trace does not wait out
    /// [`policy::HELPER_BOUND_WAIT`].
    fn runs_prepared_patterns(&self, ir: &TraceIr) -> bool {
        if !self
            .opt_flags
            .contains(crate::compat::TracingOptFlags::STRING_CONSTANTS)
        {
            return false;
        }
        // The twins are known once installed, which the first compile does.
        native::install_helpers();
        // A loop that rewrites with `string.gsub` compiled to no gain:
        // `string/pattern_gsub` (scale 3, level 1, `instructions:u`) counts
        // 72.0M compiled and 71.5M waiting, which is the level 0 count.
        !native::calls_gsub(ir) && native::pattern_calls_are_prepared(ir)
    }

    /// The tier a new trace compiles at: the policy's, except for a trace
    /// whose work is the pattern calls it makes (see
    /// [`native::is_helper_bound`]).
    fn tier_for(&self, ir: &TraceIr) -> OptimizationTier {
        if self.tier == OptimizationTier::Optimized && native::is_helper_bound(ir) {
            OptimizationTier::Baseline
        } else {
            self.tier
        }
    }

    fn finish_closed(
        &mut self,
        stats: &mut TraceStatistics,
        closed: ClosedTrace,
        at: &HookPoint<'_>,
    ) {
        let Some(recorder) = self.recorder.take() else {
            return;
        };
        if let Some(target) = self.side_target.take() {
            self.finish_side(stats, target, &recorder, closed, at);
            return;
        }
        if self.defers_to_outer(stats, recorder.chunk(), &closed) {
            return;
        }
        if self.runs_too_short(recorder.chunk(), &closed) {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the loop at pc {} has {:?} iterations left in its run, too few to repay a trace",
                    closed.header_pc, closed.remaining_trips
                );
            }
            let chunk = Arc::clone(recorder.chunk());
            self.settle_site(
                stats,
                &chunk,
                closed.header_pc,
                TraceAbortReason::ShortLoop,
                Some(closed.header_pc),
            );
            return;
        }
        stats.traces_recorded += 1;
        stats.recorded_instructions += u64::from(closed.guest_instructions);
        stats.slot_reads_forwarded += u64::from(closed.slot_reads_forwarded);
        stats.array_overwrites += u64::from(closed.slot_overwrites);
        // Later activations of the root frame (tail calls in it) are no
        // inlined calls.
        stats.inlined_calls += closed
            .ir
            .frames
            .iter()
            .filter(|frame| frame.inlined.is_some())
            .count() as u64;
        if debug_env::trace_ir() {
            for (pc, instruction) in recorder.chunk().code.iter().enumerate() {
                eprintln!(
                    "  {pc}: {:?} a={} b={} c={} d={}",
                    instruction.opcode(),
                    instruction.a(),
                    instruction.b(),
                    instruction.c(),
                    instruction.d()
                );
            }
            eprintln!("jit-trace: recorded {:#?}", closed.ir);
        }
        self.next_trace_id = self.next_trace_id.wrapping_add(1);
        if self.compile {
            let chunk = Arc::clone(recorder.chunk());
            // A loop of prepared pattern calls with iterations to spare skips
            // the wait (see `runs_prepared_patterns`).
            // Whether a trace pays depends on its siblings too: while another
            // compile is in flight or another trace is installed, the retired
            // trace may only have lost its window to the interpreter waiting
            // on them, so it gets its compile. A function trace is judged on
            // the activations each entry runs, which no sibling changes.
            // A trace that came back twice, and twice failed a window of its
            // own, does not owe its failure to the siblings either.
            if self.judge_one_shot_loops
                && closed.ir.function.is_none()
                && native::is_helper_bound(&closed.ir)
                && (!self.runs_prepared_patterns(&closed.ir) || runs_per_entry_too_short(&closed))
                && let Some(site) = self.sites.site_mut(&chunk, closed.header_pc)
                && !site.helper_bound_waited
            {
                site.helper_bound_waited = true;
                site.state = SiteState::Cooldown {
                    remaining: policy::HELPER_BOUND_WAIT,
                };
                if debug_env::trace() {
                    eprintln!(
                        "jit-trace: the trace at pc {} is helper-bound, waiting {} backedges before it compiles",
                        closed.header_pc,
                        policy::HELPER_BOUND_WAIT
                    );
                }
                return;
            }
            let repeats = self.retired_repeats(&chunk, closed.header_pc, &closed.ir);
            let alone = !self.has_compile_work() && self.cache.live() == 0;
            if repeats > 0 && (alone || closed.ir.function.is_some() || repeats >= 2) {
                // The iteration this recording ran took the path of a trace
                // that never paid, and the next may take another: record
                // again before turning the site away.
                if closed.ir.function.is_none()
                    && let Some(site) = self.sites.site_mut(&chunk, closed.header_pc)
                    && site.repeat_restarts < MAX_REPEAT_RESTARTS
                {
                    site.repeat_restarts += 1;
                    site.state = SiteState::Cold {
                        backedges: self.hot_threshold - 1,
                    };
                    stats.recording_starts = stats.recording_starts.saturating_sub(1);
                    if debug_env::trace() {
                        eprintln!(
                            "jit-trace: the trace at pc {} repeats one that never paid, recording again",
                            closed.header_pc
                        );
                    }
                    return;
                }
                if debug_env::trace() {
                    eprintln!(
                        "jit-trace: the trace at pc {} repeats one that never paid, rejected",
                        closed.header_pc
                    );
                }
                stats.negative_evictions +=
                    self.sites
                        .reject(&chunk, closed.header_pc, RejectCause::RepeatedUnpaid);
                // The same trace of a loop that retired for its few trips
                // would retire for them again: the loop is short, and the
                // loops around it record again and unroll it.
                if closed.ir.function.is_none()
                    && self
                        .sites
                        .existing_mut(&chunk, closed.header_pc)
                        .is_some_and(|site| {
                            site.retired_short |= site.short_retirements >= 1;
                            site.retired_short
                        })
                {
                    self.reopen_outer_loops(&chunk, closed.header_pc);
                }
                return;
            }
            // The recording closed, so the aborts before it are no streak:
            // a loop of one trip per entry starts every other recording on
            // the pass that leaves, and three of those spread over the
            // traces it recorded in between would reject it.
            if let Some(site) = self.sites.site_mut(&chunk, closed.header_pc) {
                site.same_aborts = 0;
                site.empty_passes = 0;
            }
            let tier = if self.is_stepping_stone(&chunk, &closed) {
                // The tree's sides are recorded in the interpreter, and the
                // root compiles with them (or alone, if they never land).
                if self.stage_root(stats, &chunk, &closed, &recorder, at) {
                    return;
                }
                stats.baseline_roots += 1;
                OptimizationTier::Baseline
            } else {
                self.tier_for(&closed.ir)
            };
            self.request_compile(stats, &chunk, closed.header_pc, closed.ir, tier);
            return;
        }
        if let Some(site) = self.sites.site_mut(recorder.chunk(), closed.header_pc) {
            site.state = SiteState::Queued;
        }
        if self.queue.len() >= TRACE_QUEUE_CAPACITY {
            // The evicted trace will never compile, so let its site count
            // again instead of staying Queued for good.
            if let Some(evicted) = self.queue.pop_front()
                && let Some(chunk) = evicted.chunk.upgrade()
                && let Some(site) = self.sites.site_mut(&chunk, evicted.header_pc)
                && site.state == SiteState::Queued
            {
                site.state = SiteState::Cold { backedges: 0 };
            }
        }
        self.queue.push_back(QueuedTrace {
            chunk: Arc::downgrade(recorder.chunk()),
            header_pc: closed.header_pc,
            epoch: self.epoch,
            guest_instructions: closed.guest_instructions,
            ir: closed.ir,
        });
    }

    /// Whether a root trace closed at a numeric for loop's latch cannot repay
    /// its compile. For a loop inside other loops of its function that no
    /// tree could carry (a tree's outer side keeps a loop it carries from
    /// leaving between runs), or whose stepping stone gave up: the run has fewer iterations left than a trace
    /// needs per entry to pay for entering and leaving (see
    /// [`native::min_iterations_of`]), or all the iterations the program has
    /// left for the loop do less work than a compile
    /// ([`policy::COMPILE_REPAYING_GUEST_WORK`]). For a loop with no loop
    /// around it, only the second, which is known only in a top-level chunk:
    /// in a function, one run says little about runs to come.
    fn runs_too_short(&self, chunk: &Arc<Chunk>, closed: &ClosedTrace) -> bool {
        let Some(remaining) = closed.remaining_trips else {
            return false;
        };
        let latch = closed.ir.poll_snapshot.and_then(|poll| {
            match closed.ir.snapshots[poll.index()].phase {
                crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => Some(pc),
                crate::jit_ir::trace::SnapshotPhase::Before { .. } => None,
            }
        });
        let Some(latch) = latch else {
            return false;
        };
        let span = nest::LoopSpan {
            header: closed.header_pc,
            latch,
        };
        let cannot_repay_compile = self.judge_one_shot_loops
            && closed.remaining_work.is_some_and(|work| {
                work * f64::from(closed.guest_instructions) < policy::COMPILE_REPAYING_GUEST_WORK
            });
        if nest::enclosing_loops(chunk, span).is_empty() {
            // A loop of a top-level chunk with no loop around it runs once:
            // when the rest of that run is shorter than a compile, it ends
            // before the trace is installed (a table filled once).
            if closed.remaining_work.is_some() || !self.judge_one_shot_loops {
                return cannot_repay_compile;
            }
            // A function's loop runs again on every call, and each call
            // enters the trace once. One run says little about the next, so
            // this only turns the loop away for a few attempts (a short loop
            // is rejected after MAX_RECORDING_ATTEMPTS of them), but a loop
            // whose whole run has fewer iterations than the trace needs per entry
            // loses every call to the interpreter, and compiling it first
            // would only be thrown away by the productivity window.
            #[allow(clippy::cast_precision_loss)]
            let bar = native::min_iterations_of(&closed.ir) as f64;
            // The estimate counts from zero: a loop from one has one fewer.
            return closed.run_trips.is_some_and(|trips| trips - 1.0 < bar);
        }
        // A loop whose stepping stone gave up on its outer sides
        // (`Site::stone_gave_up`) is not carried by a tree after all.
        let gave_up = self
            .sites
            .site(chunk, closed.header_pc)
            .is_some_and(policy::Site::stone_gave_up);
        if nest::can_cover(chunk, span) && !gave_up {
            return false;
        }
        #[allow(clippy::cast_precision_loss)]
        let bar = native::min_iterations_of(&closed.ir) as f64;
        remaining + 1.0 < bar || cannot_repay_compile
    }

    /// Whether the root that just closed gives way to the loop around it: a
    /// loop with few, steady trips per run inside a loop with many iterations
    /// left is better run unrolled in one trace of the outer loop (see
    /// `Recorder::nest_plan`) than as a trace of its own that the outer loop
    /// enters and leaves on every iteration. The tree of the short loop pays
    /// its entry checks and a loop-back per trip on every run. The unrolled
    /// trace pays neither, and a nest of short loops multiplies the saving.
    ///
    /// The trip count has to be the same twice in a row. A count that
    /// changes needs a side trace for each value, which compile one after the
    /// other, and costs more than the tree for the first few hundred
    /// thousand iterations (`control/short_inner`: trips of 3 to 6).
    /// The first closed root is turned away like a short loop and tells the
    /// count; the second decides. The loop around it records next, with
    /// unrolling on, and the short loop waits [`DEFERRAL_COOLDOWN`]
    /// backedges. If the outer recording fails, the short loop is let go
    /// at once and never gives way again.
    fn defers_to_outer(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        closed: &ClosedTrace,
    ) -> bool {
        let Some(plan) = closed.nest_plan else {
            return false;
        };
        if !self.compile
            || closed.ir.function.is_some()
            || plan.root_trips > policy::PREFER_UNROLL_TRIPS
            || plan.guest_instructions > policy::PREFER_UNROLL_GUEST_INSTRUCTIONS
            || plan.backedges > policy::PREFER_UNROLL_BACKEDGES
            // The recording of the loop around faces the same judgment as
            // any root of a top-level chunk (`runs_too_short`): the iterations
            // it has left must repay a compile.
            || plan.target_remaining * plan.guest_instructions < policy::COMPILE_REPAYING_GUEST_WORK
        {
            return false;
        }
        let header = closed.header_pc;
        let Some(site) = self.sites.existing_mut(chunk, header) else {
            return false;
        };
        if site.no_deferral || site.deferrals >= policy::MAX_DEFERRALS {
            return false;
        }
        // The loop around has to be undecided, and has not run out of room
        // for a nest before.
        if let Some(target) = self.sites.existing_mut(chunk, plan.target.header)
            && (target.unroll_overflowed
                || target.cover_revoked
                || !matches!(
                    target.state,
                    SiteState::Cold { .. } | SiteState::Cooldown { .. }
                ))
        {
            return false;
        }
        let Some(site) = self.sites.existing_mut(chunk, header) else {
            return false;
        };
        let steady = match site.trips_seen.replace(plan.root_trips) {
            Some(seen) if (seen - plan.root_trips).abs() < f64::EPSILON => true,
            Some(_) => {
                // The count changes: this loop keeps its own tree.
                site.no_deferral = true;
                site.trips_seen = None;
                return false;
            }
            None => false,
        };
        site.deferrals += 1;
        site.state = SiteState::Cooldown {
            remaining: if steady {
                policy::DEFERRAL_COOLDOWN
            } else {
                policy::RETRY_COOLDOWN_BASE + 1
            },
        };
        stats.recording_aborts[TraceAbortReason::ShortLoop.index()] += 1;
        if debug_env::trace() {
            eprintln!(
                "jit-trace: the loop at pc {header} has {} trips per run: {}",
                plan.root_trips,
                if steady {
                    format!("the loop at pc {} records the nest", plan.target.header)
                } else {
                    "waits for a second look".to_owned()
                }
            );
        }
        if steady {
            let target = self.sites.site_mut(chunk, plan.target.header);
            if let Some(target) = target {
                target.inner_loop_met = true;
                target.deferred_from = Some(header);
                target.state = SiteState::Cold {
                    backedges: self.hot_threshold - 1,
                };
            }
        }
        true
    }

    /// The recording of `header`, a loop that short loops gave way to, ended
    /// without a trace. Those loops record on their own again.
    fn release_deferred(&mut self, chunk: &Arc<Chunk>, header: usize) {
        let Some(from) = self
            .sites
            .existing_mut(chunk, header)
            .and_then(|site| site.deferred_from.take())
        else {
            return;
        };
        if let Some(inner) = self.sites.existing_mut(chunk, from) {
            inner.no_deferral = true;
            if matches!(inner.state, SiteState::Cooldown { .. }) {
                inner.state = SiteState::Cold {
                    backedges: self.hot_threshold - 1,
                };
            }
        }
    }

    /// Whether a closed root only leads to the tree that will replace it.
    /// A loop inside other loops that a tree can carry, and that runs fewer
    /// iterations per entry than a trace needs to pay for one, is installed
    /// so that its exit can take the outer side (see "Nested loops" in the
    /// module documentation). It runs a few dozen entries, until that side
    /// lands and the tree compiles over it, so its code is compiled at the
    /// baseline tier: the same trace at the optimized tier costs a second
    /// compile of the whole loop to be replaced by a third.
    ///
    /// A root without `remaining_trips` (a `while` or generic-for loop) has
    /// no trip count to judge here: it compiles at the policy's tier, and
    /// its first productivity window decides whether it is a stone (see
    /// `InstalledTrace::keep_stone`).
    fn is_stepping_stone(&self, chunk: &Arc<Chunk>, closed: &ClosedTrace) -> bool {
        let Some(remaining) = closed.remaining_trips else {
            return false;
        };
        if closed.ir.function.is_some() || self.tier != OptimizationTier::Optimized {
            return false;
        }
        let Some(latch) = closed.ir.poll_snapshot.and_then(|poll| {
            match closed.ir.snapshots[poll.index()].phase {
                crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => Some(pc),
                crate::jit_ir::trace::SnapshotPhase::Before { .. } => None,
            }
        }) else {
            return false;
        };
        let span = nest::LoopSpan {
            header: closed.header_pc,
            latch,
        };
        #[allow(clippy::cast_precision_loss)]
        let bar = native::min_iterations_of(&closed.ir)
            .max(native::OUTER_SIDE_MAX_ITERATIONS_PER_ENTRY) as f64;
        remaining + 1.0 < bar
            && !nest::enclosing_loops(chunk, span).is_empty()
            && nest::can_cover(chunk, span)
    }

    /// A side recording ended without a trace. The guard it continued keeps
    /// exiting, and gets another try while its attempts last. A flush or an
    /// error that is not the guard's fault costs nothing.
    // Each abort reason is its own documented case; `InnerLoop`'s arm keeps
    // its data-dependent note even though the retry budget matches.
    #[allow(clippy::match_same_arms)]
    fn abort_side(
        &mut self,
        stats: &mut TraceStatistics,
        target: SideTarget,
        reason: TraceAbortReason,
        at_return: bool,
        loop_ended: bool,
    ) {
        stats.side_aborts += 1;
        if debug_env::trace() {
            eprintln!("jit-trace: side recording aborted: {reason:?}");
        }
        // What the interpreter did differently this time may not repeat; an
        // arm the recorder cannot follow will.
        let outer_ended = reason == TraceAbortReason::LoopNotClosed
            && (loop_ended
                || self
                    .cache
                    .get(target.index)
                    .is_some_and(|trace| trace.serial == target.serial && trace.encloses));
        let waits_for_link = outer_ended
            && at_return
            && self
                .cache
                .get(target.index)
                .is_none_or(|trace| trace.entries() < native::RETURN_THROUGH_ENTRIES);
        // A side that met an inner loop its recording could not follow is
        // recorded again with short loops unrolled, as its root would be
        // (`Site::inner_loop_met`, which `begin_side` reads): the first such
        // abort marks the root's site and costs the guard no attempt. A loop
        // the unrolling retry cannot follow either spends attempts as before.
        let unrolls_next = reason == TraceAbortReason::InnerLoop
            && self
                .cache
                .get(target.index)
                .filter(|trace| trace.serial == target.serial)
                .and_then(|trace| Some((trace.chunk()?, trace.header_pc())))
                .and_then(|(chunk, header)| {
                    let site = self.sites.existing_mut(&chunk, header)?;
                    let first = !site.inner_loop_met && !site.unroll_overflowed;
                    site.inner_loop_met = true;
                    first.then_some(())
                })
                .is_some();
        let attempts = match reason {
            TraceAbortReason::Flushed => 0,
            _ if unrolls_next => 0,
            // Waits for the tree to be able to run on past the return; the
            // guard costs nothing until then. A tree that has run enough
            // entries to link through the return and still met a return the
            // recorder cannot follow (no caller to link, a caller it
            // refuses) spends an attempt, or the guard would record the
            // same abort every time it fails.
            _ if waits_for_link => 0,
            // The recorded outer iteration was the last one: another may
            // have inner trips. Or the root's loop ended on the occurrence
            // recorded, and another occurrence of the guard goes on.
            _ if outer_ended => 1,
            // A key outside the array on the iteration recorded is as likely
            // to be inside it on another one.
            TraceAbortReason::Error
            | TraceAbortReason::Yield
            | TraceAbortReason::Hook
            | TraceAbortReason::Divergence
            | TraceAbortReason::TableMiss
            | TraceAbortReason::DepthChange => 1,
            // Whether the arm runs an inner loop often depends on the data
            // (a loop whose trip count is zero for most iterations).
            TraceAbortReason::InnerLoop => 1,
            _ => policy::MAX_SIDE_ATTEMPTS,
        };
        if let Some(trace) = self.cache.get_mut(target.index)
            && trace.serial == target.serial
        {
            trace.end_side_recording(target.key, attempts);
            if waits_for_link {
                trace.hold_side_for_return(target.key);
            } else if outer_ended {
                trace.retry_side_soon(target.key);
            }
            // The side this loop side was recorded behind waits for it.
            let behind = target.key.parent == Some(native::PREVIOUS_SIDE);
            let beyond = behind && target.key.origin != crate::jit_ir::trace::TERMINAL_GUARD;
            // A scout that ran out of instructions would again.
            if beyond && reason == TraceAbortReason::LimitExceeded {
                self.submit_queued(stats, target.index);
            } else if beyond {
                if let Some(chunk) = self
                    .cache
                    .get(target.index)
                    .and_then(native::InstalledTrace::chunk)
                {
                    self.retry_beyond_side(stats, target, &chunk);
                }
            } else if trace.release_held() || behind {
                self.submit_queued(stats, target.index);
            }
            self.recompile_stone(target.index);
        }
    }

    /// The recording of the code beyond the nest behind a queued outer side
    /// ended without the side wanted. It watches for the next place to start
    /// while attempts last, and otherwise lets the queued sides compile.
    fn retry_beyond_side(
        &mut self,
        stats: &mut TraceStatistics,
        target: SideTarget,
        chunk: &Arc<Chunk>,
    ) {
        let Some(trace) = self.cache.get_mut(target.index) else {
            return;
        };
        trace.end_chained_recording(target.key);
        let anchor = trace.take_beyond_try().and_then(|frame| {
            let queued = trace.last_queued()?;
            let anchor =
                native::beyond_anchor(chunk, trace.tree_ir(), trace.header_pc(), &queued.ir)?;
            Some((frame, anchor))
        });
        if let Some((frame, anchor)) = anchor
            && self.arm_beyond_side(stats, target, Arc::clone(chunk), frame, anchor)
        {
            return;
        }
        self.submit_queued(stats, target.index);
    }

    /// A baseline-tier root whose outer side will not land stays the code
    /// that runs, so it is compiled again at the policy's tier. The recompile
    /// is a tree compile of the root alone, which holds sides recorded
    /// meanwhile back until it lands.
    fn recompile_stone(&mut self, index: u32) {
        let epoch = self.epoch;
        if !self.has_stepping_stones() {
            return;
        }
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        if !trace.stone_given_up() {
            return;
        }
        // A tree that is mostly pattern calls is as good at the baseline tier.
        if trace.is_helper_bound() {
            return;
        }
        let tier = self.tier;
        let serial = trace.serial;
        let version = trace.tree_version();
        let ir = trace.begin_stone_recompile();
        self.outbox.push(CompileRequest {
            ir,
            tier,
            epoch,
            target: CompileTarget::Retree {
                index,
                serial,
                version,
            },
            reserved: 0,
            side: None,
            planned: None,
            victims: Vec::new(),
        });
    }

    /// A stepping stone whose code hoisted a branch guard into the preheader
    /// is compiled again with its branch guards kept in the loop (see
    /// [`native::InstalledTrace::begin_keep_guards`]). The tree keeps its
    /// tier.
    pub(super) fn consider_kept_guards(&mut self, index: u32) {
        let epoch = self.epoch;
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        let Some(ir) = trace.begin_keep_guards() else {
            return;
        };
        if debug_env::trace() {
            eprintln!(
                "jit-trace: the tree at pc {} compiles again with its branch guards in the loop",
                trace.header_pc()
            );
        }
        let tier = trace.compiled_tier();
        let serial = trace.serial;
        let version = trace.tree_version();
        self.outbox.push(CompileRequest {
            ir,
            tier,
            epoch,
            target: CompileTarget::Retree {
                index,
                serial,
                version,
            },
            reserved: 0,
            side: None,
            planned: None,
            victims: Vec::new(),
        });
    }

    /// A tree whose sides compiled apart have cost more than the tree would
    /// to compile again with them inline is compiled again that way (see
    /// [`native::InstalledTrace::wants_relink`]). The tree keeps its tier.
    pub(super) fn consider_relink(&mut self, index: u32) {
        let epoch = self.epoch;
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        if !trace.wants_relink() {
            return;
        }
        if debug_env::trace() {
            eprintln!("jit-trace: trace {index} runs its sides apart too often, compiling them in");
        }
        let tier = trace.compiled_tier();
        let serial = trace.serial;
        let version = trace.tree_version();
        let ir = trace.begin_relink();
        self.outbox.push(CompileRequest {
            ir,
            tier,
            epoch,
            target: CompileTarget::Retree {
                index,
                serial,
                version,
            },
            reserved: 0,
            side: None,
            planned: None,
            victims: Vec::new(),
        });
    }

    /// A tree at the optimized tier that has run long enough is compiled
    /// again at the peak tier (see [`native::PEAK_ITERATIONS`]), as a tree compile of
    /// the tree it is, which holds sides recorded meanwhile back until it
    /// lands. Asked for once.
    pub(super) fn consider_peak(&mut self, index: u32) {
        if self.tier != OptimizationTier::Optimized {
            return;
        }
        let epoch = self.epoch;
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        if !trace.wants_peak() {
            return;
        }
        if debug_env::trace() {
            eprintln!(
                "jit-trace: trace {index} ran {} iterations, compiling it at the peak tier",
                trace.iterations()
            );
        }
        let serial = trace.serial;
        let version = trace.tree_version();
        let ir = trace.begin_stone_recompile();
        self.outbox.push(CompileRequest {
            ir,
            tier: OptimizationTier::Peak,
            epoch,
            target: CompileTarget::Retree {
                index,
                serial,
                version,
            },
            reserved: 0,
            side: None,
            planned: None,
            victims: Vec::new(),
        });
    }

    /// A side recording closed. The side joins the trace's queue. If no tree
    /// compile is in flight the queue is built into a tree and compiled, and
    /// the trace keeps running its current code meanwhile; otherwise the side
    /// waits for the compile in flight to land, so that the sides recorded in
    /// the meantime compile together.
    fn finish_side(
        &mut self,
        stats: &mut TraceStatistics,
        target: SideTarget,
        recorder: &Recorder,
        closed: ClosedTrace,
        at: &HookPoint<'_>,
    ) {
        // A recording behind a queued outer side that does not leave the
        // nest in front of a loop is not the one wanted: it followed the
        // outer loop round, or left the nest with nothing to run in place.
        if target.key.parent == Some(native::PREVIOUS_SIDE)
            && target.key.origin != crate::jit_ir::trace::TERMINAL_GUARD
            && closed.entered_loop.is_none()
        {
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: the side beyond the nest from guard {:?} ends in front of no loop, dropped",
                    target.key
                );
            }
            self.retry_beyond_side(stats, target, recorder.chunk());
            return;
        }
        stats.side_traces_recorded += 1;
        if closed.guest_instructions == 0 && closed.ir.guards.is_empty() {
            stats.empty_sides += 1;
        }
        stats.recorded_instructions += u64::from(closed.guest_instructions);
        stats.slot_reads_forwarded += u64::from(closed.slot_reads_forwarded);
        stats.array_overwrites += u64::from(closed.slot_overwrites);
        // A side that starts inside inlined calls counts against the tree's
        // budget for such sides (see `MAX_SEEDED_GUEST_INSTRUCTIONS`).
        if closed.ir.seed_frames > 0
            && let Some(trace) = self.cache.get_mut(target.index)
            && trace.serial == target.serial
        {
            trace.note_seeded_side(closed.guest_instructions);
        }
        let closed_crossed = closed.crossed.clone();
        if debug_env::trace() {
            eprintln!(
                "jit-trace: side trace recorded from guard {:?}{}: {} guest instructions, loops entered {:?}",
                target.key,
                debug_env::at_line(recorder.chunk(), closed.header_pc),
                closed.guest_instructions,
                closed.crossed
            );
        }
        let chunk = Arc::clone(recorder.chunk());
        // The registers live at the header of the tree's loop. A loop side
        // closes at a header of its own, which says nothing about them.
        let tree_header = self
            .cache
            .get(target.index)
            .and_then(native::InstalledTrace::profile_site)
            .map_or(closed.header_pc, |(_, header)| header);
        let live = self.registers_live_at(
            &chunk,
            if target.key.origin == crate::jit_ir::trace::TERMINAL_GUARD {
                tree_header
            } else {
                closed.header_pc
            },
        );
        let Some(trace) = self.cache.get_mut(target.index) else {
            return;
        };
        if trace.serial != target.serial {
            return;
        }
        // A side that ended in front of a loop: the loop is recorded next,
        // from the state the side's exit restores, so that both join the
        // tree in one compile. An outer side that leaves the outermost loop
        // around the root likewise gets the code beyond that loop, when the
        // loop is nearly done.
        let beyond = (closed.entered_loop.is_none()
            && recorder.captured().is_empty()
            && self.compile)
            .then(|| native::beyond_anchor(&chunk, trace.tree_ir(), trace.header_pc(), &closed.ir))
            .flatten()
            .filter(|anchor| outer_loop_nearly_done(&chunk, anchor.outer, at));
        let chain = closed
            .entered_loop
            .filter(|_| recorder.captured().is_empty() && self.compile);
        let side = TraceSide {
            parent: target.key.parent.map(|parent| parent as usize),
            guard: target.key.origin,
            ir: Box::new(closed.ir),
        };
        if debug_env::trace_ir() {
            eprintln!("jit-trace: side recorded {side:#?}");
        }
        trace.side_recorded(
            target.key,
            side,
            live,
            closed_crossed,
            recorder.returned_into(),
        );
        if let Some(entered) = chain
            && self.start_chained_loop_side(stats, target, recorder, entered)
        {
            return;
        }
        if let Some(anchor) = beyond {
            let frame = (recorder.depth(), recorder.captured(), recorder.coroutine());
            if let Some(trace) = self.cache.get_mut(target.index) {
                trace.begin_beyond_scouting(frame.clone());
            }
            if self.arm_beyond_side(stats, target, Arc::clone(recorder.chunk()), frame, anchor) {
                return;
            }
        }
        let Some(trace) = self.cache.get_mut(target.index) else {
            return;
        };
        // A guard of the root that is about to ask for its side too (the
        // first trip of the inner loop, which the outer side hands to the
        // root's header) gets it into the same compile.
        if trace.holds_for_pending() {
            return;
        }
        self.submit_queued(stats, target.index);
    }

    /// Starts the recording of the loop side behind the side that just
    /// closed in front of its loop (see [`native::PREVIOUS_SIDE`]). The
    /// interpreter is at the loop's header, and the recorder takes the
    /// instruction there as its first. Returns whether it started.
    fn start_chained_loop_side(
        &mut self,
        stats: &mut TraceStatistics,
        target: SideTarget,
        recorder: &Recorder,
        (header, latch): (usize, usize),
    ) -> bool {
        let key = native::SideKey {
            parent: Some(native::PREVIOUS_SIDE),
            origin: crate::jit_ir::trace::TERMINAL_GUARD,
        };
        let Some(trace) = self.cache.get(target.index) else {
            return false;
        };
        let (Some(profile), Some((tree, planned))) = (trace.profile(), trace.seed_plan()) else {
            return false;
        };
        let Some(parent) = tree.sides.len().checked_sub(1) else {
            return false;
        };
        let place = native::SeedPlace {
            key: native::SideKey {
                parent: u32::try_from(parent).ok(),
                origin: crate::jit_ir::trace::TERMINAL_GUARD,
            },
            pc: header,
            last: Some(latch),
            header: trace.header_pc(),
            latch,
            encloses: true,
            beyond: false,
            captured: Vec::new(),
            closure: None,
            caller: None,
            return_exit: false,
            depth: recorder.depth(),
            inner: Vec::new(),
            root_registers: Vec::new(),
            root_cells: Vec::new(),
            root_environment: None,
            lazy: trace.lazy.clone(),
        };
        let Some((seed, header, latch)) = native::side_seed(&tree, &planned, &place) else {
            return false;
        };
        let chunk = Arc::clone(recorder.chunk());
        let liveness = self.liveness_of(&chunk);
        let loop_recorder =
            Recorder::new_side(chunk, recorder.depth(), header, latch, profile, seed)
                .with_limits(self.record_limits)
                .with_captured(Vec::new())
                .with_liveness(liveness)
                .with_rebuilt_callee_frames(self.rebuild_callee_frames)
                .with_value_selects(self.value_selects);
        self.recorder = Some(Box::new(loop_recorder));
        self.side_target = Some(SideTarget {
            index: target.index,
            serial: target.serial,
            key,
        });
        if let Some(trace) = self.cache.get_mut(target.index) {
            trace.begin_chained_recording(key);
        }
        stats.side_recordings += 1;
        true
    }

    /// Arms the recording of the code beyond the outermost loop around the
    /// root, behind the outer side that just closed (see
    /// [`native::PREVIOUS_SIDE`]): a scout watches the interpreter finish the
    /// loop, and the side starts at the guard of the closed side whose
    /// failure edge the interpreter takes (see [`Self::wake_beyond_side`]).
    /// Returns whether it is armed.
    fn arm_beyond_side(
        &mut self,
        stats: &mut TraceStatistics,
        target: SideTarget,
        chunk: Arc<Chunk>,
        (depth, captured, coroutine): native::ScoutFrame,
        anchor: native::BeyondAnchor,
    ) -> bool {
        let Some(trace) = self.cache.get(target.index) else {
            return false;
        };
        let (Some(profile), header) = (trace.profile(), trace.header_pc()) else {
            return false;
        };
        let scout = Recorder::new(chunk, depth, header, anchor.outer.latch, coroutine, profile)
            .with_limits(self.record_limits)
            .with_captured(captured)
            .scouting(
                (anchor.latch, header),
                anchor.candidates,
                native::EAGER_BEYOND_WAIT,
            );
        let key = native::SideKey {
            parent: Some(native::PREVIOUS_SIDE),
            origin: u32::MAX - 1,
        };
        self.recorder = Some(Box::new(scout));
        self.side_target = Some(SideTarget {
            index: target.index,
            serial: target.serial,
            key,
        });
        if let Some(trace) = self.cache.get_mut(target.index) {
            trace.begin_chained_recording(key);
        }
        stats.side_recordings += 1;
        true
    }

    /// The scout saw the interpreter take the failure edge of `wake`'s guard:
    /// the recording of the side that guard continues begins at the
    /// instruction about to run. Returns whether it began.
    fn wake_side(&mut self, scout: &Recorder, wake: WakeCandidate) -> bool {
        let Some(target) = self.side_target else {
            return false;
        };
        let Some(trace) = self.cache.get(target.index) else {
            return false;
        };
        let (Some(profile), Some((tree, planned))) = (trace.profile(), trace.seed_plan()) else {
            return false;
        };
        // The guard of the root, or of the side queued last.
        let parent = if wake.root {
            None
        } else {
            let Some(parent) = tree.sides.len().checked_sub(1) else {
                return false;
            };
            u32::try_from(parent).ok()
        };
        let Some(root_latch) =
            tree.poll_snapshot
                .and_then(|poll| match tree.snapshots[poll.index()].phase {
                    crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => Some(pc),
                    crate::jit_ir::trace::SnapshotPhase::Before { .. } => None,
                })
        else {
            return false;
        };
        let place = native::SeedPlace {
            key: native::SideKey {
                parent,
                origin: wake.origin,
            },
            pc: wake.successor,
            last: Some(wake.guard_pc),
            header: trace.header_pc(),
            latch: root_latch,
            encloses: true,
            beyond: !wake.root,
            captured: scout.captured(),
            closure: None,
            caller: None,
            return_exit: false,
            depth: scout.depth(),
            inner: Vec::new(),
            root_registers: Vec::new(),
            root_cells: Vec::new(),
            root_environment: None,
            lazy: trace.lazy.clone(),
        };
        let Some((seed, header, latch)) = native::side_seed(&tree, &planned, &place) else {
            return false;
        };
        let chunk = Arc::clone(scout.chunk());
        let liveness = self.liveness_of(&chunk);
        let side_recorder = Recorder::new_side(chunk, scout.depth(), header, latch, profile, seed)
            .with_limits(self.record_limits)
            .with_captured(scout.captured())
            .with_liveness(liveness)
            .with_rebuilt_callee_frames(self.rebuild_callee_frames)
            .with_value_selects(self.value_selects);
        // The side of a root's guard is an ordinary side; the one behind a
        // queued side waits for it in the queue.
        let key = native::SideKey {
            parent: if wake.root {
                None
            } else {
                Some(native::PREVIOUS_SIDE)
            },
            origin: wake.origin,
        };
        self.recorder = Some(Box::new(side_recorder));
        self.side_target = Some(SideTarget {
            index: target.index,
            serial: target.serial,
            key,
        });
        if let Some(trace) = self.cache.get_mut(target.index) {
            trace.begin_chained_recording(key);
        }
        true
    }

    /// Records the sides of a stepping stone's tree ahead of its compile, in
    /// the interpreter, instead of compiling the root to find its exits out:
    /// the root is installed as a placeholder (see [`CompiledTrace::staged`])
    /// and a scout waits for the interpreter to leave its loop. The side that
    /// continues the loop's exit is recorded then, and the sides behind it,
    /// and the tree compiles once. Returns whether the root was staged.
    fn stage_root(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        closed: &ClosedTrace,
        recorder: &Recorder,
        at: &HookPoint<'_>,
    ) -> bool {
        let ir = &closed.ir;
        let header = closed.header_pc;
        let refused = |why: &str| {
            if debug_env::trace() {
                eprintln!("jit-trace: the root at pc {header} is not staged: {why}");
            }
            false
        };
        if !self.compile
            || !self.stage_roots
            || ir.counted
            || ir.function.is_some()
            || ir.loop_tail.is_some()
            || !recorder.captured().is_empty()
        {
            return refused("the policy or the trace rules it out");
        }
        let Some(latch) =
            ir.poll_snapshot
                .and_then(|poll| match ir.snapshots[poll.index()].phase {
                    crate::jit_ir::trace::SnapshotPhase::After { pc, .. } => Some(pc),
                    crate::jit_ir::trace::SnapshotPhase::Before { .. } => None,
                })
        else {
            return refused("it has no poll snapshot");
        };
        // The sides ahead of the compile are recorded for a root with one
        // loop around it: the side that runs through that loop, and the code
        // beyond it. A deeper nest takes its sides from the exits of the
        // compiled root, level by level.
        let around = nest::enclosing_loops(chunk, nest::LoopSpan { header, latch });
        if around.len() != 1 {
            return refused("it is not inside exactly one loop");
        }
        // The loop around the root has to be nearly done: the interpreter
        // runs the rest of it with the scout watching. The sides recorded
        // ahead are the sides of one pass through the nest, and a root with
        // other hot guards would compile again for the sides they take later,
        // which the stepping stone (cheap) and the natural order of sides
        // (one compile for the sides that come together) do better for a
        // long outer loop. Measured against the alternative of staging every
        // root with one loop around it, which compiles the tree after the
        // first outer side and takes the others from the exits of the
        // compiled tree (instructions:u, scale 3, tracing): `control/short_inner`
        // 34.5M to 33.4M, `algorithm/spectral_norm` 107.9M to 117.3M (its
        // roots ask for a second outer side a few dozen entries after the
        // tree compiled, and an outer side that runs through loops waits
        // for SIDE_BEYOND_ITERATIONS and the burst gate before it may be
        // compiled apart, so while the tree is young that second outer side
        // compiles the tree again; see "Sides compiled apart").
        if !around
            .first()
            .is_some_and(|outer| outer_loop_nearly_done(chunk, *outer, at))
        {
            return refused("the loop around it has many iterations left");
        }
        // A loop around the root with a trace of its own, or one on its way,
        // leaves the root's exit to that trace.
        let owned = around.first().is_some_and(|outer| {
            self.sites
                .existing_mut(chunk, outer.header)
                .is_some_and(|site| {
                    site.head.is_some()
                        || !matches!(
                            site.state,
                            SiteState::Cold { .. } | SiteState::Cooldown { .. }
                        )
                })
        });
        if owned {
            return refused("the loop around it has a trace of its own");
        }
        // The guard whose failure leaves the root's loop.
        let Some(exit) = ir.guards.iter().find(|info| {
            let snapshot = &ir.snapshots[info.snapshot.index()];
            snapshot.frames.len() == 1
                && matches!(
                    snapshot.phase,
                    crate::jit_ir::trace::SnapshotPhase::After { pc, successor }
                        if pc == latch && successor == latch + 1
                )
        }) else {
            return refused("no guard leaves its loop");
        };
        let origin = exit.origin;
        let profile = self.profile.unwrap_or_default();
        let ir = Arc::new(ir.clone());
        let Some(site) = self.sites.existing_mut(chunk, header) else {
            return false;
        };
        site.state = SiteState::Compiling;
        self.install_compiled(
            stats,
            chunk,
            header,
            Arc::clone(&ir),
            CompiledTrace::staged(&ir),
        );
        let Some(index) =
            self.sites
                .existing_mut(chunk, header)
                .and_then(|site| match site.state {
                    SiteState::Installed { index } => Some(index),
                    _ => None,
                })
        else {
            // The installation settled the site itself.
            return true;
        };
        let Some(trace) = self.cache.get_mut(index) else {
            return true;
        };
        let serial = trace.serial;
        let key = native::SideKey {
            parent: None,
            origin,
        };
        trace.begin_chained_recording(key);
        let scout = Recorder::new(
            Arc::clone(chunk),
            recorder.depth(),
            header,
            latch,
            recorder.coroutine(),
            profile,
        )
        .with_limits(self.record_limits)
        .with_captured(recorder.captured())
        .scouting(
            (latch, usize::MAX),
            vec![WakeCandidate {
                guard_pc: latch,
                successor: latch + 1,
                origin,
                beyond: true,
                root: true,
            }],
            native::EAGER_BEYOND_WAIT,
        );
        self.recorder = Some(Box::new(scout));
        self.side_target = Some(SideTarget { index, serial, key });
        stats.side_recordings += 1;
        true
    }

    /// Builds the trace's queued sides into its tree and compiles the result,
    /// unless a compile is already in flight. A side that does not fit the
    /// tree (its types disagree with the loop's, or another side's) is
    /// refused for good.
    pub(super) fn submit_queued(&mut self, stats: &mut TraceStatistics, index: u32) {
        self.submit_queued_sides(stats, index);
        // A side the tree refuses for good spends its attempts here and never
        // records again, so no abort or failed compile follows to notice that
        // a stepping stone has run out of sides.
        self.recompile_stone(index);
    }

    fn submit_queued_sides(&mut self, stats: &mut TraceStatistics, index: u32) {
        if self.submit_detached(index) {
            return;
        }
        let epoch = self.epoch;
        let policy_tier = self.tier;
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        let Some((tree, keys, refused)) = trace.take_queued_tree() else {
            return;
        };
        stats.side_rejections += refused;
        let Some((tree, planned)) = tree else {
            return;
        };
        let tier = if policy_tier == OptimizationTier::Optimized && native::is_helper_bound(&tree) {
            OptimizationTier::Baseline
        } else if trace.is_peak() {
            // A tree that earned the peak tier keeps it as it grows.
            OptimizationTier::Peak
        } else {
            policy_tier
        };
        let tree = Arc::new(tree);
        trace.remember_grown_plan(&tree, &planned);
        let request = CompileRequest {
            planned: Some(planned),
            ir: tree,
            // A tree is the code that runs: it compiles at the policy's tier
            // even when the root it grew from was a stepping stone.
            tier,
            epoch,
            target: CompileTarget::Retree {
                index,
                serial: trace.serial,
                version: trace.tree_version(),
            },
            reserved: 0,
            side: None,
            victims: Vec::new(),
        };
        trace.begin_side_compile(keys);
        self.outbox.push(request);
    }

    /// A tree compile failed or was refused: the trace keeps its code, and
    /// sides recorded meanwhile get their turn. A refusal for lack of room
    /// costs the sides nothing. A failed compile retries a batch one side at
    /// a time, and a single side that fails is done.
    fn fail_side(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        serial: u64,
        version: u32,
        failure: SideFailure<'_>,
    ) {
        if let Some(trace) = self.cache.get_mut(index)
            && trace.serial == serial
            && trace.tree_version() == version
        {
            match failure {
                SideFailure::Refused => {
                    // Asking for a refused stone recompile again right away
                    // would be refused again while the broker is full.
                    if trace.end_side_compile_refused() {
                        self.submit_queued_sides(stats, index);
                        return;
                    }
                }
                SideFailure::Failed(tree) => {
                    stats.side_rejections += 1;
                    trace.end_side_compile_failed(tree);
                }
            }
            self.submit_queued(stats, index);
        }
    }

    /// A side compiled apart did not join its tree: the broker refused it, the
    /// backend failed on it, or the tree's code could not take it. It goes
    /// back to the queue to compile with the tree (`mark`: a refusal for lack
    /// of room is not the side's fault, and may try again apart).
    #[allow(clippy::too_many_arguments)]
    fn fail_detached(
        &mut self,
        stats: &mut TraceStatistics,
        index: u32,
        serial: u64,
        version: u32,
        source: usize,
        tree: &Arc<TraceIr>,
        mark: bool,
    ) {
        if let Some(trace) = self.cache.get_mut(index)
            && trace.serial == serial
            && trace.tree_version() == version
        {
            stats.detached_refusals += 1;
            trace.end_detached_compile_failed(tree, source, mark);
            self.submit_queued(stats, index);
        }
    }

    /// Compiles the first queued side of the trace apart from its tree, if
    /// the tree can take it that way. Returns whether a compile was queued.
    fn submit_detached(&mut self, index: u32) -> bool {
        if !self.detach_sides {
            return false;
        }
        let epoch = self.epoch;
        let policy_tier = self.tier;
        let stones = self.has_stepping_stones();
        let link_writes = self.detach_link_writes;
        let join = self.join_small_trees && debug_env::joins_small_trees();
        let Some(trace) = self.cache.get_mut(index) else {
            return false;
        };
        if !trace.can_detach_now() {
            return false;
        }
        let Some(plan) = trace.plan_detached(stones, link_writes, join) else {
            return false;
        };
        // A side behind a side compiled apart (the loops of a link that
        // return into the caller) runs once per call, in code that a loop
        // of the tree does most of the work in. The baseline tier compiles
        // it for a third of the instructions (nbody, scale 3: 42.7M with
        // every side at the optimized tier). The side that returns runs
        // with the caller's whole bookkeeping, which is most of a small
        // nest's work (`control/loop_in_call`: 55.7M at the optimized tier,
        // 58.1M with the baseline).
        let behind = plan.tree.sides[plan.source]
            .parent
            .is_some_and(|parent| plan.apart.contains(&parent));
        let tier = if (native::is_helper_bound(&plan.tree) || native::takes_untyped(&plan.tree))
            && policy_tier == OptimizationTier::Optimized
        {
            OptimizationTier::Baseline
        } else if trace.is_peak() {
            // The tree earned the peak tier, and its sides get it too.
            OptimizationTier::Peak
        } else if behind && policy_tier == OptimizationTier::Optimized {
            OptimizationTier::Baseline
        } else {
            policy_tier
        };
        let Some(context) = trace.begin_detached_compile(&plan) else {
            return false;
        };
        let target = CompileTarget::Side {
            index,
            serial: trace.serial,
            version: trace.tree_version(),
            source: plan.source,
        };
        let side = crate::jit::SideJob {
            planned: plan.planned,
            apart: plan.apart,
            detached: plan.detached,
            side: plan.source,
            exit_style: plan.exit_style,
            context,
        };
        self.outbox.push(CompileRequest {
            ir: Arc::new(plan.tree),
            tier,
            epoch,
            target,
            reserved: 0,
            side: Some(side),
            planned: None,
            victims: Vec::new(),
        });
        true
    }

    /// The guest registers live at `pc` of `chunk`, from the shared liveness.
    fn registers_live_at(&mut self, chunk: &Arc<Chunk>, pc: usize) -> Vec<u16> {
        let liveness = self.liveness_of(chunk);
        let sets = liveness.get_or_init(|| observable_registers(chunk));
        let count = usize::from(chunk.register_count);
        let Some(set) = sets.get(pc) else {
            // No liveness for the pc: every register may be live.
            return (0..count).filter_map(|r| u16::try_from(r).ok()).collect();
        };
        (0..count)
            .filter(|register| {
                u8::try_from(*register)
                    .map_or(true, |register| register_set_contains(set, register))
            })
            .filter_map(|register| u16::try_from(register).ok())
            .collect()
    }

    /// Admits a closed trace for compilation: reserves its IR bytes in the
    /// cache (evicting cold traces if needed) and queues the request. When
    /// the caps leave no room the trace is dropped and its header backs off
    /// without spending its retry budget.
    fn request_compile(
        &mut self,
        stats: &mut TraceStatistics,
        chunk: &Arc<Chunk>,
        header_pc: usize,
        ir: TraceIr,
        tier: OptimizationTier,
    ) {
        let _icount = crate::jit::icount::scope("request-compile");
        let ir = Arc::new(ir);
        let bytes = ir.estimated_bytes();
        if debug_env::trace() {
            eprintln!(
                "jit-trace: compile requested for the header at pc {header_pc}{}",
                debug_env::at_line(chunk, header_pc)
            );
        }
        let head = self
            .sites
            .existing_mut(chunk, header_pc)
            .and_then(|site| site.head);
        let siblings = self.chain(head);
        if siblings.len() >= self.cache.limits().versions {
            stats.cache_rejections += 1;
            self.back_off_refused(chunk, header_pc, COMPILE_REFUSAL_COOLDOWN);
            return;
        }
        self.sweep_dead_traces();
        // Victims an earlier queued request already claimed stay out of this
        // plan, or both would count the same freed bytes.
        let mut keep = siblings.clone();
        keep.extend(
            self.outbox
                .iter()
                .flat_map(|request| request.victims.iter().map(|(index, _)| *index)),
        );
        let Some(victims) = self.cache.plan_eviction(0, bytes, false, &keep) else {
            stats.cache_rejections += 1;
            self.back_off_refused(chunk, header_pc, COMPILE_REFUSAL_COOLDOWN);
            return;
        };
        // The victims go only when the broker accepts the request. The
        // reservation may exceed the cap until then.
        let victims = victims
            .into_iter()
            .filter_map(|index| Some((index, self.cache.get(index)?.serial)))
            .collect();
        self.cache.reserve_meta_over_cap(bytes);
        if let Some(site) = self.sites.existing_mut(chunk, header_pc) {
            site.state = SiteState::Compiling;
        }
        // The recorder verified the trace when it closed it, and a trace
        // without sides plans as itself. Handing the compile that plan says
        // so: it then verifies the trace again in debug builds only, as it
        // does for a tree that was verified as it grew.
        let planned = if ir.sides.is_empty() {
            crate::jit_ir::trace::plan_tree(&ir).ok().map(Arc::new)
        } else {
            None
        };
        self.outbox.push(CompileRequest {
            ir,
            tier,
            epoch: self.epoch,
            target: CompileTarget::New {
                chunk: Arc::downgrade(chunk),
                header_pc,
            },
            reserved: bytes,
            side: None,
            planned,
            victims,
        });
    }

    /// Whether a side compiled apart from its tree is outstanding: queued for
    /// the broker or with it. For tests.
    #[cfg(test)]
    pub(crate) fn side_compile_pending(&self) -> bool {
        self.outbox
            .iter()
            .any(|request| matches!(request.target, CompileTarget::Side { .. }))
            || self
                .pending
                .values()
                .any(|pending| matches!(pending.target, CompileTarget::Side { .. }))
    }

    /// Whether requests wait for the broker or results are outstanding. The
    /// VM services compilations only when this is true.
    #[inline]
    pub(crate) fn has_compile_work(&self) -> bool {
        !self.outbox.is_empty() || !self.pending.is_empty()
    }

    /// Whether the caller waits for the result of `request` when the broker
    /// compiles in the background. A stepping stone is installed only to
    /// give its exit the outer side, and the tree compiled over it replaces
    /// it; until both land the interpreter runs the loop at full price, so
    /// the compile of the baseline root, of the tree that replaces a baseline
    /// root and of a baseline root's own recompile are waited for (a blocked
    /// thread runs no instructions). Anything else installs when it is done.
    pub(crate) fn awaits_result(&self, request: &CompileRequest) -> bool {
        if !self.has_stepping_stones() {
            return false;
        }
        match request.target {
            CompileTarget::New { .. } => request.tier == OptimizationTier::Baseline,
            CompileTarget::Retree { index, serial, .. } => self
                .cache
                .get(index)
                .is_some_and(|trace| trace.serial == serial && trace.is_baseline()),
            // The trace runs its code meanwhile, and the side lands when it
            // is done.
            CompileTarget::Side { .. } => false,
        }
    }

    /// Whether a baseline-tier root is a stepping stone (see
    /// [`Self::is_stepping_stone`]). Under a baseline policy every root is
    /// baseline code and none is a stone: there is no better tier to wait
    /// for or to recompile at.
    fn has_stepping_stones(&self) -> bool {
        self.tier != OptimizationTier::Baseline
    }

    /// Takes the requests that wait for the broker.
    pub(crate) fn take_outbox(&mut self) -> Vec<CompileRequest> {
        std::mem::take(&mut self.outbox)
    }

    /// The broker accepted `request` under `id`.
    pub(crate) fn note_submitted(
        &mut self,
        stats: &mut TraceStatistics,
        mut request: CompileRequest,
        id: CompilationId,
    ) {
        stats.compile_submissions += 1;
        self.broker_full_skips = 0;
        for (index, serial) in std::mem::take(&mut request.victims) {
            if self
                .cache
                .get(index)
                .is_some_and(|trace| trace.serial == serial)
            {
                self.evict(stats, index);
            }
        }
        let pending = PendingCompile {
            target: request.target,
            epoch: request.epoch,
            reserved: request.reserved,
            ir: request.ir,
            followers: Vec::new(),
        };
        if let Some(first) = self.pending.get_mut(&id) {
            // The broker folded an identical request into an outstanding
            // one. Keep this target: it gets its own compile afterwards.
            first.followers.push(CompileRequest {
                ir: pending.ir,
                tier: request.tier,
                epoch: pending.epoch,
                target: pending.target,
                reserved: pending.reserved,
                side: request.side,
                planned: None,
                victims: Vec::new(),
            });
        } else {
            self.pending.insert(id, pending);
        }
    }

    /// The broker refused `request`. Its trace is dropped, the header backs
    /// off, and nothing counts against the retry budget: the broker's queue
    /// or byte budget was full.
    pub(crate) fn note_submit_failed(
        &mut self,
        stats: &mut TraceStatistics,
        request: CompileRequest,
        error: SubmitError,
    ) {
        stats.compile_refusals += 1;
        self.broker_full_skips = BROKER_FULL_SKIPS;
        self.cache.release_reserved(request.reserved);
        match request.target {
            CompileTarget::New { chunk, header_pc } => {
                if let Some(chunk) = chunk.upgrade() {
                    let cooldown = if error == SubmitError::QueueFull {
                        QUEUE_FULL_COOLDOWN
                    } else {
                        COMPILE_REFUSAL_COOLDOWN
                    };
                    self.back_off_refused(&chunk, header_pc, cooldown);
                }
            }
            CompileTarget::Retree {
                index,
                serial,
                version,
            } => self.fail_side(stats, index, serial, version, SideFailure::Refused),
            CompileTarget::Side {
                index,
                serial,
                version,
                source,
                ..
            } => self.fail_detached(stats, index, serial, version, source, &request.ir, false),
        }
    }

    /// A finished compilation from the broker. Results from before a flush,
    /// for a header or trace that no longer exists, or for a site that is no
    /// longer waiting are dropped.
    pub(crate) fn complete_compile(
        &mut self,
        stats: &mut TraceStatistics,
        id: CompilationId,
        epoch: u64,
        result: Result<crate::jit::CompileOutput, TraceCompileError>,
    ) {
        use crate::jit::CompileOutput;
        let _phase = alloc_phase::enter(Phase::Installation);
        // A panic counts wherever its result goes: it is a codegen bug even
        // when a flush made the result stale.
        if let Err(TraceCompileError::Panicked(message)) = &result {
            stats.compile_panics += 1;
            if debug_env::trace() {
                eprintln!(
                    "jit-trace: compile panic {}: {message}",
                    stats.compile_panics
                );
            }
        }
        // Every completion frees room in the broker.
        self.broker_full_skips = 0;
        let Some(mut pending) = self.pending.remove(&id) else {
            stats.stale_completions += 1;
            return;
        };
        self.cache.release_reserved(pending.reserved);
        let followers = std::mem::take(&mut pending.followers);
        if epoch != self.epoch || pending.epoch != self.epoch {
            stats.stale_completions += 1;
            for follower in followers {
                self.cache.release_reserved(follower.reserved);
            }
            return;
        }
        // Followers compile separately: the finished code installs once.
        self.outbox.extend(followers);
        if let CompileTarget::Side {
            index,
            serial,
            version,
            source,
            ..
        } = pending.target
        {
            match result {
                Ok(CompileOutput::Side(side)) => {
                    self.install_side(stats, index, serial, version, source, *side, pending.ir);
                }
                Ok(CompileOutput::Tree(_)) => {
                    unreachable!("a side request makes a side")
                }
                Err(error) => {
                    if debug_env::trace() {
                        eprintln!("jit-trace: side of trace {index} not compiled: {error}");
                    }
                    self.fail_detached(stats, index, serial, version, source, &pending.ir, true);
                }
            }
            return;
        }
        let result = result.map(|output| match output {
            CompileOutput::Tree(tree) => *tree,
            CompileOutput::Side(_) => unreachable!("a tree request makes a tree"),
        });
        match pending.target {
            CompileTarget::New { chunk, header_pc } => {
                let Some(chunk) = chunk.upgrade() else {
                    stats.stale_completions += 1;
                    return;
                };
                match result {
                    Ok(compiled) => {
                        self.install_compiled(stats, &chunk, header_pc, pending.ir, compiled);
                    }
                    Err(error) => {
                        if debug_env::trace() {
                            eprintln!("jit-trace: trace at pc {header_pc} not compiled: {error}");
                        }
                        self.settle_site(
                            stats,
                            &chunk,
                            header_pc,
                            TraceAbortReason::CompileRejected,
                            None,
                        );
                    }
                }
            }
            CompileTarget::Side { .. } => unreachable!("handled above"),
            CompileTarget::Retree {
                index,
                serial,
                version,
            } => match result {
                Ok(compiled) => {
                    self.install_tree(stats, index, serial, version, compiled, pending.ir);
                }
                Err(error) => {
                    if debug_env::trace() {
                        eprintln!("jit-trace: tree at trace {index} not compiled: {error}");
                    }
                    self.fail_side(
                        stats,
                        index,
                        serial,
                        version,
                        SideFailure::Failed(&pending.ir),
                    );
                }
            },
        }
    }

    /// Drops all traces and recording state, as `jit.flush()` does. A
    /// recording in progress is counted as aborted by the flush. Compile
    /// requests the broker holds are abandoned: their results will not
    /// install, because the epoch advanced. The code of a trace that is
    /// running is freed when the run returns.
    pub(crate) fn flush(&mut self, stats: &mut TraceStatistics) {
        if self.recorder.take().is_some() {
            stats.recording_aborts[TraceAbortReason::Flushed.index()] += 1;
        }
        self.side_target = None;
        self.side_candidate = None;
        stats.stale_completions += self.pending.len() as u64;
        self.pending.clear();
        self.broker_full_skips = 0;
        self.outbox.clear();
        self.sites.clear();
        self.liveness.clear();
        self.queue.clear();
        self.bank.clear();
        drop(self.cache.clear());
        self.epoch += 1;
    }

    /// The values the root bank holds: the tables and closures the lanes of a
    /// native run name. The collector reads them as roots, so an object only
    /// a trace's lane still reaches survives a collection that runs while the
    /// trace is between its entry and the restore of its exit. The bank is
    /// empty whenever no run is in flight, so a trace never pins an object
    /// (or the entry of a weak table) across runs.
    pub(crate) fn root_values(&self) -> &[Value] {
        &self.bank
    }

    /// Tells the policy whether any hook is installed. A hook that appears
    /// where none was leaves the traces that exist uncounted: returns
    /// whether there is such work, which the VM then flushes.
    pub(crate) fn note_hooks(&mut self, installed: bool) -> bool {
        let appeared = installed && !self.hooks_installed;
        self.hooks_installed = installed;
        appeared
            && (self.cache.live() != 0
                || self.recorder.is_some()
                || !self.pending.is_empty()
                || !self.outbox.is_empty()
                || !self.queue.is_empty())
    }

    /// Sets the live count of the running thread's count hook for the next
    /// entry: `None` when it has no hook.
    pub(crate) const fn set_hook_budget(&mut self, budget: Option<i64>) {
        self.hook_budget = budget;
    }

    /// The count a run left, see [`Self::set_hook_budget`].
    pub(crate) const fn hook_budget(&self) -> Option<i64> {
        self.hook_budget
    }

    /// Whether a native run is between its entry and the restore of its
    /// exit. Its frames' registers may then hold values their pcs do not
    /// describe, so a collection must root whole windows and clear nothing.
    pub(crate) const fn run_in_flight(&self) -> bool {
        self.in_flight
    }

    /// Tells the policy which VM is entering a trace, for the helper that
    /// resumes a coroutine from native code (see
    /// [`oximond::jit_ir::trace::HelperEffects::RESUMES`]): a trace that
    /// runs another coroutine needs the VM itself, which owns this policy, so
    /// the VM lends its address for the run. Zero takes it back. The helper
    /// and nothing else uses it, and only while a run is in flight.
    pub(crate) const fn set_resume_host(&mut self, vm: usize) {
        self.resume_vm = vm;
    }

    /// Marks a native run in flight, for tests that collect inside one.
    #[cfg(test)]
    pub(crate) const fn set_run_in_flight(&mut self, in_flight: bool) {
        self.in_flight = in_flight;
    }

    /// The bank, for tests that hold a run open.
    #[cfg(test)]
    pub(crate) fn bank_mut(&mut self) -> &mut Vec<Value> {
        &mut self.bank
    }

    /// The flush epoch. Advances on every flush.
    #[cfg(test)]
    pub(crate) const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The trace cache, for tests.
    #[cfg(test)]
    pub(crate) const fn cache(&self) -> &TraceCache {
        &self.cache
    }

    /// The trace cache, for tests that pin runs or change the caps.
    #[cfg(test)]
    pub(crate) const fn cache_mut(&mut self) -> &mut TraceCache {
        &mut self.cache
    }

    /// The site table, for tests.
    #[cfg(test)]
    pub(crate) const fn sites_mut(&mut self) -> &mut policy::SiteTable {
        &mut self.sites
    }

    /// Compile requests the broker holds, for tests.
    #[cfg(test)]
    pub(crate) fn pending_compiles(&self) -> usize {
        self.pending.len()
    }

    /// Whether sides wait in the queue of the trace at `index`, for tests.
    #[cfg(test)]
    pub(crate) fn has_queued_sides(&self, index: u32) -> bool {
        self.cache
            .get(index)
            .is_some_and(|trace| trace.tree_ir() != &trace.pending_tree())
    }

    /// The pending tree and its cached plan for the trace at `index` (see
    /// `InstalledTrace::seed_plan`), for tests.
    #[cfg(test)]
    pub(crate) fn seed_plan_of(
        &self,
        index: u32,
    ) -> Option<(
        Arc<crate::jit_ir::trace::TraceIr>,
        Arc<Vec<crate::jit_ir::trace::TreeMember>>,
    )> {
        self.cache.get(index)?.seed_plan()
    }

    /// The tree of the trace at `index` as it will be once its queued sides
    /// join it, built afresh, for tests.
    #[cfg(test)]
    pub(crate) fn pending_tree_of(&self, index: u32) -> Option<crate::jit_ir::trace::TraceIr> {
        self.cache
            .get(index)
            .map(native::InstalledTrace::pending_tree)
    }

    /// Sends side `source` of the trace at `index` through a compile apart
    /// that the tree could not take: it goes back to the front of the queue,
    /// as after a refused detach. For tests.
    #[cfg(test)]
    pub(crate) fn requeue_refused_detach(&mut self, index: u32, source: usize) {
        let Some(trace) = self.cache.get_mut(index) else {
            return;
        };
        let tree = trace.tree_ir().clone();
        let side = &tree.sides[source];
        trace.begin_refused_detach_for_test(native::SideKey {
            parent: side.parent.and_then(|parent| u32::try_from(parent).ok()),
            origin: side.guard,
        });
        trace.end_detached_compile_failed(&tree, source, false);
    }

    /// The serial of the trace at `index`, which identifies it across slot
    /// reuse, for tests.
    #[cfg(test)]
    pub(crate) fn installed_serial(&self, index: u32) -> Option<u64> {
        self.cache.get(index).map(|trace| trace.serial)
    }
}

/// The library-qualified name of the native or builtin function `callable`
/// (`math.ceil`, `pcall`), as its installer gave it.
/// Whether the numeric for loop `outer` has at most
/// [`native::EAGER_BEYOND_OUTER_TRIPS`] iterations left in the frame `at`
/// stands in. A loop of another kind, or one the registers do not describe,
/// has none known.
fn outer_loop_nearly_done(chunk: &Arc<Chunk>, outer: nest::LoopSpan, at: &HookPoint<'_>) -> bool {
    if !Arc::ptr_eq(chunk, at.chunk) {
        return false;
    }
    let Some(latch) = chunk.code.get(outer.latch) else {
        return false;
    };
    if latch.opcode() != Ok(crate::bytecode::OpCode::NumericForLoop) {
        return false;
    }
    let base = usize::from(latch.a());
    let number = |register: usize| match at.registers.get(register)? {
        #[allow(clippy::cast_precision_loss)]
        Value::Integer(value) => Some(*value as f64),
        Value::Number(value) => Some(*value),
        _ => None,
    };
    let (Some(control), Some(limit), Some(step)) =
        (number(base), number(base + 1), number(base + 2))
    else {
        return false;
    };
    if debug_env::trace() {
        eprintln!("jit-trace: outer loop control {control} limit {limit} step {step}");
    }
    step > 0.0 && ((limit - control) / step) <= native::EAGER_BEYOND_OUTER_TRIPS
}

fn callable_name(at: &HookPoint<'_>, callable: &Value) -> String {
    match callable {
        Value::BuiltinFunction(builtin) => builtin.name().to_owned(),
        Value::NativeFunction(function) => at
            .surroundings
            .as_ref()
            .and_then(|around| around.runtime.native_name(*function))
            .map_or_else(|| format!("native#{}", function.index()), str::to_owned),
        _ => String::new(),
    }
}

/// How many steps of `next` the trace helpers made on this thread.
pub(crate) fn next_helper_steps() -> u64 {
    helpers::next_steps()
}

/// How many times a specialized pattern helper ran on this thread.
pub(crate) fn prepared_helper_calls() -> u64 {
    helpers::prepared_calls()
}

#[cfg(test)]
mod abort_site_tests {
    use super::recorder::{HookPoint, HookSource, Recorder};
    use super::*;
    use crate::bytecode::Instruction;

    fn hook(chunk: &Arc<Chunk>, pc: usize, depth: usize) -> HookPoint<'_> {
        HookPoint {
            chunk,
            depth,
            top_level: false,
            pc,
            registers: &[],
            open: &[],
            coroutine: None,
            hooks_active: false,
            source: HookSource::General,
            surroundings: None,
        }
    }

    /// A tree of the inner loop of a nest, stepped until it is installed:
    /// the VM, the index of the tree in the cache and its serial.
    fn tree_that_encloses() -> (crate::Lua, u32, u64) {
        let mut lua = crate::Lua::builder()
            .profile(crate::CompatibilityProfile::Lua51)
            .execution_mode(crate::ExecutionMode::Tracing)
            .build()
            .unwrap();
        let chunk = lua
            .compile(
                "=(abort)",
                "local acc = 0
                 for i = 1, 4000 do
                     for j = 1, 40 do acc = (acc + i * j) % 1000003 end
                 end
                 return acc",
            )
            .unwrap();
        let vm = lua.vm_mut();
        vm.use_synchronous_jit_compilation();
        let mut frames = vm.stack_for(&chunk).unwrap();
        for _ in 0..5_000_000 {
            let state = &vm.jit.trace;
            let found = (0..16).find_map(|index| {
                let trace = state.cache().get(index)?;
                trace.encloses.then_some((index, trace.serial))
            });
            if let Some((index, serial)) = found {
                return (lua, index, serial);
            }
            assert!(vm.step(&mut frames).unwrap().is_none(), "no tree installed");
        }
        panic!("no tree that encloses was installed");
    }

    /// Aborts a side recording of `chunk` with `LoopNotClosed` at `pc` of
    /// `frame`, and tells whether the guard `origin` of the tree is now held
    /// for the return.
    fn held_after_abort(
        lua: &mut crate::Lua,
        (index, serial): (u32, u64),
        root: &Arc<Chunk>,
        frame: &Arc<Chunk>,
        pc: usize,
        origin: u32,
    ) -> bool {
        let key = native::SideKey {
            parent: None,
            origin,
        };
        let state = &mut lua.vm_mut().jit.trace;
        state.recorder = Some(Box::new(Recorder::new(
            Arc::clone(root),
            1,
            0,
            0,
            None,
            crate::CompatibilityProfile::Lua51,
        )));
        state.side_target = Some(SideTarget { index, serial, key });
        let mut counters = TraceStatistics::default();
        state.abort_recording_at(
            &mut counters,
            TraceAbortReason::LoopNotClosed,
            Some(&hook(frame, pc, 2)),
        );
        state
            .cache()
            .get(index)
            .unwrap()
            .side_is_held_for_return(key)
    }

    /// The side of an outer loop that ran on to a `Return` waits for its
    /// tree to run past the return. Which instruction that was is read from
    /// the chunk of the frame the interpreter was running, which is not the
    /// recording's root chunk when the recording had left the root frame.
    #[test]
    fn a_side_is_held_for_the_return_of_the_frame_it_ended_in() {
        let root = Arc::new(Chunk::new(
            2,
            Vec::new(),
            vec![
                Instruction::load_constant(0, 0),
                Instruction::return_(0),
                Instruction::load_constant(1, 0),
            ],
        ));
        let callee = Arc::new(Chunk::new(
            2,
            Vec::new(),
            vec![
                Instruction::load_constant(0, 0),
                Instruction::load_constant(1, 0),
                Instruction::return_(0),
            ],
        ));
        let (mut lua, index, serial) = tree_that_encloses();
        let tree = (index, serial);
        // pc 2 returns in the callee and not in the root.
        assert!(held_after_abort(&mut lua, tree, &root, &callee, 2, 1));
        // pc 1 returns in the root and not in the callee.
        assert!(!held_after_abort(&mut lua, tree, &root, &callee, 1, 2));
    }
}
