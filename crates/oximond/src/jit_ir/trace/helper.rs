//! The helper table: library functions a trace calls directly
//! ([`TraceOp::CallHelper`](super::TraceOp::CallHelper)).
//!
//! A helper is an `extern "C"` function. The IR names it by [`HelperId`], a
//! stable index into [`HELPERS`], which holds what the recorder, the optimizer
//! and the backend need to know: the typed signature and the effect flags. The
//! table has no addresses. The crate that implements a helper installs its
//! address with the backend before the trace is compiled, so a persisted
//! trace only ever names ids.
//!
//! # Calling convention
//!
//! The backend calls `extern "C" fn([context: usize,] out: *mut i64, args...)
//! -> i64`:
//!
//! - `context` comes first when [`HelperInfo::context`] is set. It is the
//!   address of the VM's state for the run, the same word a host call gets.
//! - `out` points at [`MAX_HELPER_RESULTS`] words of the run's control block.
//!   The helper writes result `i` there as one word (integers and lanes as
//!   they are, floats as their bits, booleans as 0 or 1) and writes nothing
//!   when it misses. A helper with several results writes them all, so a call
//!   such as `string.find` or `next` returns its indices or its cursor, key
//!   and value in one go.
//! - `args` are the operands in registers: integers, lanes and addresses as
//!   `i64`, floats as `f64`, booleans widened to `i64`. A
//!   [`HelperType::Value`] parameter takes any scalar and is two `i64`
//!   registers, the entry tag of the operand's type and its bits, the words
//!   of a [`TraceInputSlot`](../../../oximond::jit/struct.TraceInputSlot.html).
//!   The backend knows the tag from the operand's type, so it is an immediate.
//! - A helper with [`HelperInfo::required`] below its parameter count takes
//!   fewer arguments at a call site. It receives the count of arguments given
//!   as an `i64` right after `out`, and the parameters the call leaves out
//!   are passed as nil (tag nil, bits 0).
//! - The return value is a status. Zero is an answer. Anything else is a miss
//!   (a wrong type, an exhausted budget, a case the helper leaves to the
//!   interpreter). The code reads it only when [`HelperEffects::MAY_EXIT`] is
//!   set, and then leaves through the call's `Before` snapshot, so the
//!   interpreter replays the instruction.
//!
//! A helper never raises a Lua error, never calls Lua or a metamethod, never
//! collects garbage and never yields. A helper that writes checks every
//! precondition before its first write: it misses with no effect, or it
//! succeeds.
//!
//! The helpers of [`HelperEffects::RESUMES`] are the exception, and the only
//! one: [`HelperId::coroutine_resume`] and [`HelperId::coroutine_wrap`] run
//! another coroutine, which runs any Lua code and may collect garbage. They
//! cannot check their precondition first, since what they check is what the
//! thread gives. A miss of theirs comes *after* the thread ran, and the VM
//! remembers what the call gave so that the interpreter's replay of the
//! instruction the guard leaves before takes it from there and does not run
//! the thread again. The effect flags say what the thread could have done to
//! the heap (everything), and a trace holds nothing across such a call that
//! the call does not read again ([`TraceIr::resume_conflict`](super::TraceIr::resume_conflict)).

use super::ir::TraceType;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Most arguments a helper takes.
pub const MAX_HELPER_ARGS: usize = 6;

/// Most results a helper gives.
pub const MAX_HELPER_RESULTS: usize = 4;

/// A stable index into [`HELPERS`]. Persisted traces name helpers by number,
/// so a change to the numbers also changes the profile format version (see
/// `MAGIC` in the VM's profile module). The numbers are dense: the helpers
/// only tests use have ids far above them.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct HelperId(pub u16);

impl HelperId {
    /// `string.len` and `#s` of a string lane the trace cannot read inline:
    /// a string of the trace's roots.
    pub const STRING_LEN: Self = Self(0);
    /// `string.len` and `#s` of a string borrowed from a table slot.
    pub const STRING_LEN_REF: Self = Self(1);
    /// `string.char` of one to five integral numbers.
    pub const STRING_CHAR: Self = Self(2);
    /// `tostring` of a number or a string.
    pub const TOSTRING: Self = Self(3);
    /// The modern `string.format`, for a call the runtime's formatter answers
    /// without `__tostring`.
    pub const STRING_FORMAT: Self = Self(4);
    /// `LuaJIT`'s `string.format`, through the native's own callback.
    pub const STRING_FORMAT_LUAJIT: Self = Self(5);
    /// `string.format` registered as a native of the other profiles, through
    /// the native's own callback.
    pub const STRING_FORMAT_NATIVE: Self = Self(6);
    /// The concatenation `..` of two to five strings and numbers.
    pub const CONCAT: Self = Self(7);
    /// `string.find` that finds a match: the start index.
    pub const STRING_FIND: Self = Self(8);
    /// `string.find` that finds none. It has no result and misses when there
    /// is a match, so the interpreter replays the call.
    pub const STRING_FIND_ABSENT: Self = Self(9);
    /// `table.concat` of a list of strings.
    pub const TABLE_CONCAT: Self = Self(10);
    /// `getmetatable` of a table, a string or another value that has a
    /// metatable without a `__metatable` field: the metatable's lane.
    pub const GETMETATABLE: Self = Self(11);
    /// `getmetatable` of a value that has no metatable. It has no result and
    /// misses when there is one, so the interpreter replays the call.
    pub const GETMETATABLE_ABSENT: Self = Self(12);
    /// `setmetatable (table, metatable)` where the table's current metatable
    /// is not protected and the new one (a table or nil) has no `__gc` and no
    /// `__mode`. It gives no result: the call's result is its first argument.
    pub const SETMETATABLE: Self = Self(13);
    /// Checks the metatable of a virtual table: that it asks for no
    /// finalizer and no weak mode (`__gc`, `__mode`) and lacks the
    /// metamethods the second argument, a mask, names (1 `__index`, 2
    /// `__newindex`, 4 `__len`). Misses otherwise, so the interpreter makes
    /// the `setmetatable` call.
    pub const PLAIN_METATABLE: Self = Self(14);
    /// Whether two strings have the same bytes (`rawequal` of two strings,
    /// which never asks `__eq`). Its one result is a boolean. A trace holds
    /// strings by lane, and two lanes may name equal strings.
    pub const STRING_EQUAL: Self = Self(15);
    /// [`HelperId::CONCAT`] for the text of an error value: misses when the
    /// result is no UTF-8, because the interpreter turns the bytes that are
    /// none into replacement characters when it raises.
    pub const CONCAT_ERROR_TEXT: Self = Self(16);
    /// `table.insert (t, value)` and `table.insert (t, position, value)` of a
    /// dense list (see `Runtime::table_trace_insert`), the elements after
    /// `position` moving up one place. Gives no result. Misses, having
    /// changed nothing, for any table or call the interpreter might treat
    /// differently: a hash part, a hole, a metatable with a metamethod the
    /// profile's `table.insert` honours, a position out of range, nil.
    pub const TABLE_INSERT: Self = Self(17);
    /// `table.remove (t)` and `table.remove (t, position)` of a dense list
    /// whose element there is an integer. Its one result is that element.
    /// Misses, changing nothing, where [`Self::TABLE_INSERT`] would, and for
    /// another kind of element.
    pub const TABLE_REMOVE_INTEGER: Self = Self(18);
    /// [`Self::TABLE_REMOVE_INTEGER`] for a float element.
    pub const TABLE_REMOVE_FLOAT: Self = Self(19);
    /// [`Self::TABLE_REMOVE_INTEGER`] for a boolean element.
    pub const TABLE_REMOVE_BOOLEAN: Self = Self(20);
    /// [`Self::TABLE_REMOVE_INTEGER`] for a table element, given as a lane.
    pub const TABLE_REMOVE_TABLE: Self = Self(21);
    /// `table.remove (t)` and `table.remove (t, position)` where the position
    /// is outside `1..=#t`, so the call gives nil and changes nothing. It has
    /// no result and misses when the position is inside.
    pub const TABLE_REMOVE_NOTHING: Self = Self(22);
    /// [`Self::TABLE_REMOVE_INTEGER`] for a string element, which the result
    /// holds as a new lane of the run's bank.
    pub const TABLE_REMOVE_STRING: Self = Self(23);
    /// `string.sub (s, i [, j])` in general: any string and any indices the
    /// interpreter's `string.sub` takes without an error. Its one result is
    /// the new string.
    pub const STRING_SUB: Self = Self(24);
    /// `string.rep (s, n [, sep])` where the result fits the run's room for
    /// strings. Misses for a count or a size the interpreter might raise an
    /// error for.
    pub const STRING_REP: Self = Self(25);
    /// `string.upper (s)`, in the C locale.
    pub const STRING_UPPER: Self = Self(26);
    /// `string.lower (s)`, in the C locale.
    pub const STRING_LOWER: Self = Self(27);
    /// `string.reverse (s)`.
    pub const STRING_REVERSE: Self = Self(28);
    /// `math.random` of a profile whose draw is a float: the argument-free
    /// form, and every form before Lua 5.3 and in `LuaJIT`. Advances the
    /// generator the interpreter's call advances, by the same code. Misses,
    /// before the generator moves, for any arguments the interpreter might
    /// raise an error for.
    pub const MATH_RANDOM_FLOAT: Self = Self(29);
    /// `math.random (m)` and `math.random (m, n)` of Lua 5.3 and later, whose
    /// draw is an integer. Otherwise [`Self::MATH_RANDOM_FLOAT`].
    pub const MATH_RANDOM_INTEGER: Self = Self(30);
    /// `math.modf` of a float whose integral part is an integer of the
    /// profile (Lua 5.3 and later): the integral part as an integer, and the
    /// fraction. Misses for a float whose integral part is a float.
    pub const MATH_MODF_INTEGER: Self = Self(31);
    /// `math.modf` of a float whose integral part is a float: every float
    /// before Lua 5.3, and in later profiles a NaN, an infinity or a float
    /// beyond the integers. Gives the integral part and the fraction.
    pub const MATH_MODF_FLOAT: Self = Self(32);
    /// `tonumber (s [, base])` that gives an integer. Misses when the
    /// interpreter's call gives anything else, an error included.
    pub const TONUMBER_INTEGER: Self = Self(33);
    /// [`Self::TONUMBER_INTEGER`] for a float.
    pub const TONUMBER_FLOAT: Self = Self(34);
    /// [`Self::TONUMBER_INTEGER`] for a call that gives nil. It has no result.
    pub const TONUMBER_NIL: Self = Self(35);
    /// `string.byte (s [, i [, j]])` of one to four bytes. Its first
    /// argument is the number of bytes the recorded call gave, and the
    /// helper misses when the call gives another number, so the results stay
    /// the same ones. The results are the bytes, as integers.
    pub const STRING_BYTE: Self = Self(36);
    /// The check before a method call on a string (`s:upper ()`): the string
    /// metatable's `__index` is a table that holds, under the key (a string),
    /// the function the second argument names (its pinned identity). The third
    /// is the receiver, passed so the trace guards its type. Changes
    /// nothing; misses when the metatable, its `__index` or the field is
    /// another one, so the interpreter makes the lookup.
    pub const STRING_METHOD: Self = Self(37);

    /// The first id of [`Self::string_find_pattern`], the family of
    /// `string.find` calls that find a match: the start, the end and then the
    /// captures the recorded call gave (at most [`MAX_FIND_CAPTURES`]).
    pub const STRING_FIND_PATTERN_BASE: u16 = 38;
    /// The first id of [`Self::string_match`], the family of `string.match`
    /// calls that find a match: the captures, or the whole match for a
    /// pattern with none (one to [`MAX_HELPER_RESULTS`] results).
    pub const STRING_MATCH_BASE: u16 = 45;
    /// A `string.match` that finds none. It has no result and misses when
    /// there is a match, so the interpreter replays the call.
    pub const STRING_MATCH_ABSENT: Self = Self(75);
    /// The first id of [`Self::gmatch_next`], the family of steps of a
    /// `string.gmatch` iterator that give a match, as [`Self::string_match`]
    /// does. Each commits the iterator's cursor.
    pub const GMATCH_NEXT_BASE: u16 = 76;
    /// A step of a `string.gmatch` iterator that gives none: the iteration is
    /// over. Commits the iterator's cursor, and misses when there is a match.
    pub const GMATCH_END: Self = Self(106);
    /// `string.gmatch (s, p [, init])`: makes the iterator, a new lane of the
    /// run's bank.
    pub const GMATCH_NEW: Self = Self(107);
    /// `string.gsub (s, p, r [, n])` with a string or a table for `r`: the new
    /// string and the number of replacements. Misses for every call the
    /// interpreter might raise an error for, and for a table with `__index`.
    pub const STRING_GSUB: Self = Self(108);
    /// A string operand of arithmetic, converted to the integer the
    /// interpreter's arithmetic converts it to. The first argument is the
    /// conversion (see the VM's `string_number`: a profile's own, or for Lua
    /// 5.4 and later the string metatable's arithmetic metamethod of an
    /// operator), the second is the string. Misses when the string does not
    /// convert, converts to a float, or the metamethod is not the string
    /// library's own, so the interpreter makes the operation. Changes
    /// nothing.
    pub const STRING_NUMBER_INTEGER: Self = Self(176);
    /// [`Self::STRING_NUMBER_INTEGER`] for a conversion that gives a float.
    pub const STRING_NUMBER_FLOAT: Self = Self(177);
    /// `utf8.char` of up to five numbers, of the profile's own limit for a
    /// code point. Misses for every call the interpreter raises an error for.
    pub const UTF8_CHAR: Self = Self(178);
    /// The first id of [`Self::utf8_codepoint`], the family of
    /// `utf8.codepoint (s [, i [, j [, lax]]])` calls that gave `n`
    /// code points (`0..=`[`MAX_UTF8_CODEPOINTS`]). It misses when the call
    /// gives another number of them, or raises an error.
    pub const UTF8_CODEPOINT_BASE: u16 = 179;
    /// `utf8.len (s [, i [, j [, lax]]])` that counts the characters: its one
    /// result. Misses when the string is invalid in the range, so the
    /// interpreter gives the failure.
    pub const UTF8_LEN: Self = Self(184);
    /// `utf8.len` that finds an invalid sequence: nil and the position of
    /// the invalid byte. The one result is the position. Misses when the
    /// string is valid.
    pub const UTF8_LEN_FAIL: Self = Self(185);
    /// `utf8.offset (s, n [, i])` that gives a position.
    pub const UTF8_OFFSET: Self = Self(186);
    /// `utf8.offset` of Lua 5.5, which gives the start and the end of the
    /// character.
    pub const UTF8_OFFSET_END: Self = Self(187);
    /// `utf8.offset` that finds no such character: nil. No result; misses
    /// when there is one.
    pub const UTF8_OFFSET_NIL: Self = Self(188);
    /// A step of the iterator `utf8.codes` gave: `(mode, s, i)` gives the
    /// position and the code point of the next character. `mode` is 0 for the
    /// strict step, 1 for the lax one and 2 for Lua 5.3's. Misses at the end
    /// of the string and for an invalid sequence.
    pub const UTF8_NEXT: Self = Self(189);
    /// The step that ends the iteration: no result, misses when there is
    /// another character or an invalid sequence.
    pub const UTF8_NEXT_END: Self = Self(190);
    /// The check `utf8.codes (s)` makes before it gives its iterator: misses
    /// when the string starts with a continuation byte, which the call
    /// raises an error for.
    pub const UTF8_CODES_CHECK: Self = Self(191);
    /// `string.pack (format, values...)` of up to five values.
    pub const STRING_PACK: Self = Self(192);
    /// `string.packsize (format)`.
    pub const STRING_PACKSIZE: Self = Self(193);
    /// The first id of [`Self::string_unpack`], the family of
    /// `string.unpack` calls that gave up to [`MAX_UNPACK_VALUES`] values and
    /// the next position.
    pub const STRING_UNPACK_BASE: u16 = 194;
    /// The first id of [`Self::next`], the family of `next` calls that gave
    /// an entry, by the types of the key and the value.
    pub const NEXT_BASE: u16 = 234;
    /// A `next` that gives no entry: the traversal is over. Its parameters
    /// are the table and the key, and it misses when an entry follows the
    /// key (and for every call the interpreter raises an error for).
    pub const NEXT_END: Self = Self(264);
    /// The check `pairs (t)` makes on the profiles whose `pairs` honours
    /// `__pairs`: misses when the table's metatable has a handler, which the
    /// interpreter calls. Changes nothing.
    pub const PAIRS_PLAIN: Self = Self(265);
    /// The first id of [`Self::next_box`], the steps of a `next` loop whose
    /// control value the trace keeps in a lane of the run's bank (see
    /// [`Self::NEXT_BOX_END`]).
    pub const NEXT_BOX_BASE: u16 = 266;
    /// The end of a traversal that keeps its control value in a lane. Its
    /// parameters are the table and that lane. It misses when an entry
    /// follows the lane's key (and for every call the interpreter raises an
    /// error for), and changes nothing.
    pub const NEXT_BOX_END: Self = Self(308);
    /// Stores a scalar in the control lane of a traversal: the start of the
    /// loop (`pairs (t)` gives nil), or any other write the loop's control
    /// register gets outside a step. It cannot miss.
    pub const BOX_SET: Self = Self(309);
    /// Checks a call through `__call`: that the callee (the first argument, a
    /// value) is a table whose metatable's raw `__call` is the pinned closure
    /// of the identity in the second. Any table of any metatable passes when
    /// that holds, so one trace serves every object of a class. Misses
    /// otherwise, so the interpreter makes the call. The id after
    /// [`Self::BOX_SET`].
    pub const CALL_HANDLER: Self = Self(310);
    /// `table[key] = value` for a string key the trace holds at run time (a
    /// lane string or a borrow out of an array slot) and a value of any
    /// scalar type, nil included, on a table the trace holds by lane or
    /// binding: the interpreter's raw store. It adds, overwrites or deletes
    /// the key, so it may move hash storage, but no array part
    /// ([`HelperEffects::REWRITES_HASH`]). Misses, having changed nothing,
    /// when the table is read-only, when the key is absent and the table's
    /// metatable has `__newindex`, which the interpreter then runs, for a key
    /// that may name a metamethod, and when a binding of the run reads the
    /// field under this key as a constant.
    pub const SET_STRING: Self = Self(311);
    /// `table[key] = value` for a string key the table holds with a number or
    /// boolean of the type of `value`: the value is overwritten where it is.
    /// Misses, having changed nothing, for any other store (an absent key, a
    /// value of another type, a read-only table), so the entries, their
    /// addresses and the table's layout stay as they are.
    pub const SET_PRESENT: Self = Self(312);
    /// `table[key] = nil` for a string key: a key a hashed table holds
    /// becomes a deleted key the traversals skip, in the place it had, and an
    /// absent key stays absent. Misses, having changed nothing, for a key of
    /// any other table (a small or shaped one, which shift or drop the entry),
    /// a read-only table, an absent key a `__newindex` takes, and a key whose
    /// field a binding of the run reads as a constant.
    pub const DELETE_STRING: Self = Self(313);
    /// Checks a read that runs a function `__index`: the first argument (a
    /// value) is a table, the second a key, the third the identity of the
    /// pinned closure. Walks the way the interpreter does: while the table
    /// lacks the key, its metatable's raw `__index` is either a table, which
    /// the walk goes on in, or a Lua closure, which must be the pinned one.
    /// Gives the table the walk ended in, the first argument of the closure
    /// (the receiver itself when no hop was made). Misses, having changed
    /// nothing, for a key some table of the walk holds, a walk that ends
    /// anywhere else or past the interpreter's hop limit, and a table with no
    /// lane to spare, so the interpreter makes the read. Any table of any
    /// metatable passes when that holds, so one trace serves every object of
    /// a class.
    pub const INDEX_HANDLER: Self = Self(314);
    /// [`Self::INDEX_HANDLER`] for a write that runs a function `__newindex`
    /// (the interpreter walks `__newindex` for an absent key the same way).
    /// Gives the table whose handler runs, the first argument of the closure.
    pub const NEWINDEX_HANDLER: Self = Self(315);
    /// Checks `#t` that runs a function `__len`: the first argument (a value)
    /// is a table whose metatable's raw `__len` is the pinned closure of the
    /// identity in the second. Misses otherwise.
    pub const LEN_HANDLER: Self = Self(316);
    /// Stores the third argument (a value) under the second (a string key) of
    /// the first (a table) as a write that runs no function does: the walk of
    /// [`Self::NEWINDEX_HANDLER`] ends in a table that holds the key or whose
    /// metatable has no `__newindex`, and the value goes there (nil deletes).
    /// With a fourth argument other than zero the table itself takes the
    /// value, as `rawset` does.
    /// Moves hash storage only: the array parts of tables and the strings
    /// borrowed from them stay where they were. Misses, having changed
    /// nothing, for a walk that ends in a function or past the hop limit and
    /// for a read-only table.
    pub const TABLE_SET_STRING: Self = Self(317);
    /// Checks an operator metamethod that is a Lua closure: the first and
    /// second arguments (values) are the operands, the third an integer that
    /// names the event (the low byte, an index into the VM's operator
    /// metamethod names) and the rule that picks the handler (the byte above:
    /// 0 the left operand's handler unless it defers to the right one's, as
    /// arithmetic does; 1 the left one's, else the right one's; 2 the handler
    /// both operands share; 3 that, for operands of one type), the fourth the
    /// identity of the pinned closure that rule must pick, or -1 when it must
    /// pick none. The rules are the interpreter's, and the profile decides
    /// which one an event takes. Misses, having changed nothing, for any
    /// other handler, so the interpreter makes the operation. Any operand of
    /// any metatable passes when the handler is the pinned one, so one trace
    /// serves every object of a class.
    pub const OPERATOR_HANDLER: Self = Self(318);
    /// `table[key] = value` for a string key the table holds as a deleted
    /// key (see [`Self::DELETE_STRING`]) and a number or boolean `value`, on
    /// a table without a metatable: the entry takes the value where it
    /// stands, so no key is added and no entry moves. Misses, having changed
    /// nothing, for any other store (an absent or live key, a small or shaped
    /// table, a read-only table, a metatable, a key that may name a
    /// metamethod, a key whose field a binding of the run reads as a
    /// constant). Like a delete it turns a nil a bound field or a lane's row
    /// may hold into a value, so a trace holds no field binding or lane
    /// access beside it ([`HelperEffects::DELETES_KEYS`]).
    pub const SET_REVIVE: Self = Self(319);
    /// Makes a table and gives its lane: the first argument is the length its
    /// array part is sized for, the others (at most [`MAX_NEW_TABLE_KEYS`])
    /// the string keys of a constructor's shape, which give the table the
    /// storage the interpreter's constructor would. The table is empty. The
    /// run's bank roots it, and the loop-back release drops the lane
    /// ([`TraceOp::ReleaseStrings`](super::TraceOp::ReleaseStrings)) once
    /// nothing in the trace can name it. Misses, having made nothing, when
    /// the run has no lane to spare or the runtime has no table left.
    ///
    /// A table made with a shape also leaves the address of its first value
    /// slot in the second result word (zero for a table without one, or a
    /// host that has none). The slot of a key is as many values further as
    /// the key has places before it among the shape's sorted keys, and holds
    /// nil until the table is filled. The emitter fills a field the trace
    /// stores under a root string key straight into its slot where it can
    /// ([`Self::TABLE_INIT`] is the call it replaces).
    ///
    /// A call that names no key and gives [`TABLE_NEW_PREFILL`] in its
    /// argument count makes the array part its length long (up to
    /// [`MAX_PREFILLED_ARRAY`]), with nil slots that count as present, and
    /// leaves the address of the first slot in the second result word. The
    /// emitter sets the flag only when it fills every slot itself, before
    /// anything can read the table, so the table is a dense list again.
    pub const TABLE_NEW: Self = Self(320);
    /// Stores the third argument under the second (an integer, a float or a
    /// string key) of the first, as the interpreter's raw set does. The
    /// first argument is a table [`Self::TABLE_NEW`] made that no operation
    /// other than a call of this one names, directly or through a table
    /// that holds it, so the call moves no storage another operation holds.
    /// A fill may follow the store that made the table part of a cycle. The
    /// emitter fills a shape's slot in place only while it knows the slots
    /// have not moved (`MadeTable::moved` in the emitter tracks that).
    /// Misses, having changed nothing, for a value or key it cannot store.
    pub const TABLE_INIT: Self = Self(321);
    /// Whether two Lua closures run the same prototype (they share a chunk):
    /// the first argument is the live callee, the second a pinned closure.
    /// Its one result is a boolean. It reads nothing a trace can change, so
    /// it may be hoisted. This is a call site's guard when the callee varies
    /// between closures of one prototype ([`InlinedFrame::dynamic`](super::InlinedFrame::dynamic)).
    pub const CLOSURE_SAME_PROTO: Self = Self(322);
    /// Upvalue `index` (the second argument) of the closure in the first, as
    /// an integer. Reads the cell through the live closure, whether the cell
    /// is open (aliasing a stack slot) or closed. Misses, having changed
    /// nothing, when the cell holds another type, so the interpreter runs the
    /// instruction. Ordered against [`Self::CELL_SET`] like an effect.
    pub const CELL_GET_INTEGER: Self = Self(323);
    /// [`Self::CELL_GET_INTEGER`] for a float.
    pub const CELL_GET_FLOAT: Self = Self(324);
    /// [`Self::CELL_GET_INTEGER`] for a boolean.
    pub const CELL_GET_BOOLEAN: Self = Self(325);
    /// [`Self::CELL_GET_INTEGER`] for a table: the result is its lane, which
    /// the run's bank roots until the run ends. A tree that writes no table
    /// into a cell and resumes no coroutine reads the lane out of the
    /// closure's row, which names the lane of each cell's table when the row
    /// is filled, and checks that the cell still holds that table.
    pub const CELL_GET_TABLE: Self = Self(326);
    /// Stores the third argument (a value) into upvalue `index` (the second)
    /// of the closure in the first, as `SetUpvalue` does: through the cell,
    /// open or closed, with the collector's write barrier for a string or a
    /// table. Misses, having changed nothing, only for a closure with no such
    /// cell.
    pub const CELL_SET: Self = Self(327);
    /// Whether two table lanes (the arguments) hold the same table. A
    /// function trace's run gives a table a new lane at every read, so two
    /// lanes of one table differ and lane numbers cannot tell tables apart.
    /// Its one result is a boolean. It reads only the run's bank, which holds
    /// a table for the run.
    pub const TABLE_SAME: Self = Self(328);

    /// The first id of [`Self::coroutine_resume`], the family of
    /// `coroutine.resume (thread, args...)` calls that run the thread to its
    /// next yield or its end and give the values it handed back, followed by
    /// the family of [`Self::coroutine_wrap`].
    pub const COROUTINE_RESUME_BASE: u16 = 329;

    /// `string.sub (s, i, j)` of a string the entry binds and two floats the
    /// trace checked to lie in `-2^31 <= x < 2^31`, in a profile without
    /// integers. It is [`Self::STRING_SUB`] under another id: the optimizer
    /// drops a call whose result only snapshots read (the exits make the
    /// substring from a `Slice` entry), which it may do because the checked
    /// indices make that string the one every profile's `string.sub`
    /// gives.
    pub const STRING_SLICE: Self = Self(Self::COROUTINE_RESUME_BASE + 2 * RESUME_SHAPES as u16);

    /// Reads key `key` (the second argument, a number or a string) of the
    /// table in the first (a value) as `rawget` does, for a table the trace
    /// knows only by lane, and gives the integer it holds. With a third
    /// argument other than zero a key the table lacks is looked for in the
    /// chain of tables its metatable's `__index` names (the way a class table
    /// answers for its objects), up to the interpreter's hop limit. Misses,
    /// having changed nothing, unless the key is present in the table the
    /// read ends at with a non-nil value of the result's type, so the
    /// interpreter runs the read (a nil, `__index` as a function, another
    /// type, an error). It reads only: it allocates nothing, collects
    /// nothing and runs no Lua. The following ids are its siblings by result
    /// type.
    pub const TABLE_GET_INTEGER: Self = Self(Self::STRING_SLICE.0 + 1);
    /// [`Self::TABLE_GET_INTEGER`] for a float.
    pub const TABLE_GET_FLOAT: Self = Self(Self::STRING_SLICE.0 + 2);
    /// [`Self::TABLE_GET_INTEGER`] for a boolean.
    pub const TABLE_GET_BOOLEAN: Self = Self(Self::STRING_SLICE.0 + 3);
    /// [`Self::TABLE_GET_INTEGER`] for a table: the result is its lane, which
    /// the run's bank roots until the run ends. Misses too when the run has
    /// no lane to spare.
    pub const TABLE_GET_TABLE: Self = Self(Self::STRING_SLICE.0 + 4);
    /// [`Self::TABLE_GET_TABLE`] for a Lua closure or native function.
    pub const TABLE_GET_CLOSURE: Self = Self(Self::STRING_SLICE.0 + 5);
    /// [`Self::TABLE_GET_INTEGER`] for a string: the result is a new string
    /// lane of the run's bank, which holds the very string. Misses too when
    /// the run has no room for another string.
    pub const TABLE_GET_STRING: Self = Self(Self::STRING_SLICE.0 + 6);
    /// `#t` for a table the trace knows only by lane (the first argument, a
    /// value), as an integer: the table's border. With a second argument
    /// other than zero the profile ignores `__len` (Lua 5.1), and the border
    /// is the answer whatever the metatable holds. Otherwise it misses,
    /// having changed nothing, for a table whose metatable has a `__len`
    /// (the interpreter runs it).
    pub const TABLE_LENGTH: Self = Self(Self::STRING_SLICE.0 + 7);
    /// Stores the third argument (a value: a number, a boolean, nil, a
    /// string or a table) under the second (a number key) of the first (a
    /// table met by lane), as [`Self::TABLE_SET_STRING`] does for a string
    /// key: the walk of `__newindex` through tables ends in a table that
    /// holds the key or has no handler, and the VM's raw store puts the value
    /// there with its write barrier and accounting (nil deletes). With a
    /// fourth argument other than zero the table itself takes the value, as
    /// `rawset` does. It may grow or rehash the array and hash parts of any
    /// table, so a trace holds no address of a field or row beside it and
    /// reads array words again after it. Misses, having changed nothing, for a
    /// walk that ends in a function or past the hop limit, a read-only table
    /// and a NaN key, so the interpreter makes the store.
    pub const TABLE_SET_INDEX: Self = Self(Self::STRING_SLICE.0 + 8);
    /// [`Self::TABLE_GET_INTEGER`] for a userdata (a Luanti object, say): the
    /// result is a new lane of the run's bank, which holds the very
    /// userdata. Misses too when the run has no room for another lane.
    pub const TABLE_GET_USERDATA: Self = Self(Self::STRING_SLICE.0 + 9);
    /// The step of an `ipairs` loop that ends it, for a table the trace
    /// knows only by lane (the first argument, a value): key `key` (the
    /// second, a number) is absent from the table, which has no metatable.
    /// It has no result. Misses, having changed nothing, when the key holds
    /// a value or the table has a metatable, so the interpreter runs the
    /// step (another trip, or `__index`).
    pub const TABLE_ABSENT_INDEX: Self = Self(Self::STRING_SLICE.0 + 10);
    /// [`Self::CELL_GET_INTEGER`] for a function (a Lua closure or a native)
    /// in a frame that runs whichever closure of a prototype its call had:
    /// the result is the function's lane of the run's bank, which a call
    /// guard compares with the lane of the callee it pinned. A closure's lane
    /// is the one the bank has for it, or a new one; a native's is the lane
    /// the trace pinned it in, and the read misses for one the trace did not
    /// pin, or for any other value.
    pub const CELL_GET_CALLABLE: Self = Self(Self::STRING_SLICE.0 + 11);
    /// [`Self::CELL_GET_INTEGER`] for a string, in a frame that runs
    /// whichever closure of a prototype its call had: the result is a lane
    /// of the run's bank that holds the cell's string (the `id` of
    /// Mineclonia's concentric ring `chunk_test`, structures.lua:404, a
    /// different string in each closure). Misses for another value, or when
    /// the run has no room for another lane.
    pub const CELL_GET_STRING: Self = Self(Self::STRING_SLICE.0 + 12);
    /// Whether two Lua closures have the same environment table: the first
    /// argument is the live callee of a frame that runs whichever closure of
    /// a prototype its call had, the second the closure the recording
    /// pinned for it. Its one result is a boolean. A global that such a
    /// frame reads (`assert` in Mineclonia's concentric ring `chunk_test`,
    /// structures.lua:405) is then the pinned closure's global, behind a
    /// guard on this.
    pub const CLOSURE_SAME_ENV: Self = Self(Self::STRING_SLICE.0 + 13);
    /// The string a borrow names (the argument, the address of an array
    /// slot), as a lane of the run's bank. The recording owns a borrow with
    /// it before a call that may move the slot, so the string outlives the
    /// call. Misses when the run has no room for another lane.
    pub const STRING_OWN: Self = Self(Self::STRING_SLICE.0 + 14);
    /// The first id of [`Self::vm_call`], the family of host C function
    /// calls a trace makes without leaving.
    pub const VM_CALL_BASE: u16 = Self::STRING_SLICE.0 + 15;

    /// The last id.
    const LAST: Self = Self::vm_call(MAX_RESUME_VALUES, pow6(MAX_RESUME_VALUES) - 1);

    /// The call of a host C function (one registered through the C API) whose
    /// results are `values` values (`0..=`[`MAX_RESUME_VALUES`]) of the
    /// kinds `kinds` says: its `i`th base-6 digit is 0 for an integer, 1 for
    /// a float, 2 for a boolean, 3 for nil (the result has no word), 4 for a
    /// table (a lane of the run's bank) and 5 for a string (a lane too). The call
    /// takes no argument: it names the instruction by its guard. The helper
    /// writes the state of the guard's `Before` snapshot back to the
    /// interpreter's frame, makes the call exactly as the interpreter's
    /// `Call` does, and gives the results when they are of the recorded
    /// kinds ([`HelperEffects::RESUMES`]: the function may run any Lua
    /// code). It misses *before* the call when it cannot make it (a hook, a
    /// coroutine, another callee), and *after* it when the call went another
    /// way (an error, other results): the VM then keeps what the call gave
    /// and the interpreter's replay of the call takes it from there.
    #[must_use]
    pub const fn vm_call(values: usize, kinds: usize) -> Self {
        assert!(values <= MAX_RESUME_VALUES && kinds < pow6(values));
        let mut before = 0;
        let mut count = 0;
        while count < values {
            before += pow6(count);
            count += 1;
        }
        #[allow(clippy::cast_possible_truncation, reason = "a few hundred helpers")]
        Self(Self::VM_CALL_BASE + (before + kinds) as u16)
    }

    /// Whether this is a [`Self::vm_call`] helper.
    #[must_use]
    pub const fn is_vm_call(self) -> bool {
        self.0 >= Self::VM_CALL_BASE && self.0 <= Self::LAST.0
    }

    /// Whether this reads or writes a closure's cell through the live
    /// closure (`CELL_GET_*`, [`Self::CELL_SET`]): a cell, never a table.
    #[must_use]
    pub const fn is_cell_access(self) -> bool {
        matches!(
            self,
            Self::CELL_GET_INTEGER
                | Self::CELL_GET_FLOAT
                | Self::CELL_GET_BOOLEAN
                | Self::CELL_GET_TABLE
                | Self::CELL_GET_CALLABLE
                | Self::CELL_GET_STRING
                | Self::CELL_SET
        )
    }

    /// The first id of [`Self::string_find_prefix`], the family of
    /// `string.find` calls that find a match and want only its first
    /// results.
    pub const STRING_FIND_PREFIX_BASE: u16 = 109;
    /// The first id of [`Self::string_match_prefix`].
    pub const STRING_MATCH_PREFIX_BASE: u16 = 116;
    /// The first id of [`Self::gmatch_next_prefix`].
    pub const GMATCH_NEXT_PREFIX_BASE: u16 = 146;

    /// The `utf8.codepoint` that gave `count` code points
    /// (`0..=`[`MAX_UTF8_CODEPOINTS`]).
    #[must_use]
    pub const fn utf8_codepoint(count: usize) -> Self {
        assert!(count <= MAX_UTF8_CODEPOINTS);
        #[allow(clippy::cast_possible_truncation, reason = "a handful")]
        Self(Self::UTF8_CODEPOINT_BASE + count as u16)
    }

    /// The call of `coroutine.resume` whose results are `values` values
    /// (`0..=`[`MAX_RESUME_VALUES`]), of which the `i`th is an integer, a float
    /// or a boolean as the `i`th base-3 digit of `kinds` says (0, 1, 2). The
    /// thread is the first argument, a value; the others are what the call
    /// passes it (at most [`MAX_RESUME_ARGS`]). The helper runs the thread
    /// through the VM's own switch ([`HelperEffects::RESUMES`]) and succeeds
    /// when the thread yielded or returned exactly values of those kinds
    /// (missing ones are nil, which no kind accepts). `coroutine.resume`'s
    /// leading `true` is not a result: the call succeeds only when it is
    /// there. Anything else (an error, a dead thread, another kind, a hook
    /// installed meanwhile) misses *after* the thread ran, and the VM
    /// remembers what the call gave, so the interpreter's replay of the call
    /// takes the results from that memory instead of running the thread
    /// again.
    #[must_use]
    pub const fn coroutine_resume(values: usize, kinds: usize) -> Self {
        assert!(values <= MAX_RESUME_VALUES && kinds < pow3(values));
        let mut before = 0;
        let mut count = 0;
        while count < values {
            before += pow3(count);
            count += 1;
        }
        #[allow(clippy::cast_possible_truncation, reason = "a few dozen helpers")]
        Self(Self::COROUTINE_RESUME_BASE + (before + kinds) as u16)
    }

    /// The call of a function `coroutine.wrap` made whose results are `values`
    /// values (`0..=`[`MAX_RESUME_VALUES`]) of the kinds `kinds` says, as
    /// [`Self::coroutine_resume`] has them. The function is the first argument,
    /// a callable lane: the helper checks that it is a wrapper and runs its
    /// thread, and the call has no leading `true`. A call that fails raises
    /// the error, which the helper leaves to the interpreter's replay of the
    /// call, like every other answer it cannot take.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "a few dozen helpers")]
    pub const fn coroutine_wrap(values: usize, kinds: usize) -> Self {
        Self(Self::coroutine_resume(values, kinds).0 + RESUME_SHAPES as u16)
    }

    /// Whether this is a [`Self::coroutine_wrap`] helper.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "a few dozen helpers")]
    pub const fn is_coroutine_wrap(self) -> bool {
        self.0 >= Self::COROUTINE_RESUME_BASE + RESUME_SHAPES as u16
            && self.0 <= Self::STRING_SLICE.0
    }

    /// The `string.unpack` that gave `values` values, of which the `i`th is
    /// an integer, a float or a string as the `i`th base-3 digit of `kinds`
    /// says (0, 1, 2), then the next position.
    #[must_use]
    pub const fn string_unpack(values: usize, kinds: usize) -> Self {
        assert!(values <= MAX_UNPACK_VALUES && kinds < pow3(values));
        let mut before = 0;
        let mut count = 0;
        while count < values {
            before += pow3(count);
            count += 1;
        }
        #[allow(clippy::cast_possible_truncation, reason = "a few dozen helpers")]
        Self(Self::STRING_UNPACK_BASE + (before + kinds) as u16)
    }

    /// Whether this is one of the `utf8` or `string.unpack` helpers whose
    /// miss is the call going another way than the recording (an invalid
    /// string, another number of results, the end of an iteration). A side
    /// trace records the call as it now ends.
    #[must_use]
    pub const fn is_text_outcome(self) -> bool {
        (self.0 >= Self::UTF8_CODEPOINT_BASE && self.0 <= Self::UTF8_NEXT_END.0)
            || (self.0 >= Self::STRING_UNPACK_BASE && self.0 < Self::NEXT_BASE)
    }

    /// The `next` that gave an entry whose key has the type `key` and whose
    /// value has the type `value` ([`NextKind`]); `None` for a call that
    /// wants the key only, which takes no notice of the value.
    #[must_use]
    pub const fn next(key: NextKind, value: Option<NextKind>) -> Self {
        let value = match value {
            Some(kind) => kind as u16,
            None => NEXT_NO_VALUE,
        };
        Self(Self::NEXT_BASE + (key as u16) * NEXT_ROW + value)
    }

    /// The step of a traversal that keeps its control value in a lane of the
    /// run's bank, giving an entry whose key has the type `key` and whose
    /// value is handled as `value` says ([`BoxedValue`]); `None` for a key
    /// the loop does not use, which the helper does not produce (the key of a
    /// loop whose body never reads it is the control lane itself, so it is
    /// not made a lane or a number). It commits the entry's key to the
    /// control lane. The backend takes the step without calling the helper
    /// where the trace says it can ([`super::BoxedSteps`]), with the same
    /// results and the same lane afterwards.
    #[must_use]
    pub const fn next_box(key: Option<NextKind>, value: BoxedValue) -> Self {
        let key = match key {
            Some(kind) => kind as u16,
            None => NEXT_NO_VALUE,
        };
        let value = match value {
            BoxedValue::Kind(kind) => kind as u16,
            BoxedValue::Skipped => NEXT_NO_VALUE,
            BoxedValue::Lane => NEXT_VALUE_LANE,
        };
        Self(Self::NEXT_BOX_BASE + key * NEXT_BOX_ROW + value)
    }

    /// The kinds of the key and the value of the lane-keeping step this is
    /// ([`Self::next_box`] reversed), or `None` for any other helper.
    #[must_use]
    pub const fn next_box_shape(self) -> Option<(Option<NextKind>, BoxedValue)> {
        if self.0 < Self::NEXT_BOX_BASE || self.0 >= Self::NEXT_BOX_END.0 {
            return None;
        }
        let row = self.0 - Self::NEXT_BOX_BASE;
        let (key, value) = ((row / NEXT_BOX_ROW) as usize, (row % NEXT_BOX_ROW) as usize);
        Some((
            if key < NEXT_KINDS {
                Some(NextKind::ALL[key])
            } else {
                None
            },
            if value < NEXT_KINDS {
                BoxedValue::Kind(NextKind::ALL[value])
            } else if value == NEXT_VALUE_LANE as usize {
                BoxedValue::Lane
            } else {
                BoxedValue::Skipped
            },
        ))
    }

    /// Whether this steps a traversal with `next` or ends it, whose miss is
    /// the traversal going another way than the recording (the end of the
    /// table, an entry of another type). A side trace records the step as it
    /// now ends.
    #[must_use]
    pub const fn is_next_outcome(self) -> bool {
        (self.0 >= Self::NEXT_BASE && self.0 <= Self::NEXT_END.0)
            || (self.0 >= Self::NEXT_BOX_BASE && self.0 <= Self::NEXT_BOX_END.0)
    }

    /// The `string.find` that finds a match and gave `captures` captures
    /// (`0..=`[`MAX_FIND_CAPTURES`]), of which the ones in the bits of `kinds`
    /// are strings and the others positions.
    #[must_use]
    pub const fn string_find_pattern(captures: usize, kinds: usize) -> Self {
        Self(Self::STRING_FIND_PATTERN_BASE + shape_offset(captures, 0, kinds))
    }

    /// The `string.match` that gave `results` results (`1..=4`), of which the
    /// ones in the bits of `kinds` are strings and the others positions.
    #[must_use]
    pub const fn string_match(results: usize, kinds: usize) -> Self {
        Self(Self::STRING_MATCH_BASE + shape_offset(results, 1, kinds))
    }

    /// The step of a `string.gmatch` iterator that gave `results` results,
    /// shaped as [`Self::string_match`].
    #[must_use]
    pub const fn gmatch_next(results: usize, kinds: usize) -> Self {
        Self(Self::GMATCH_NEXT_BASE + shape_offset(results, 1, kinds))
    }

    /// [`Self::string_find_pattern`] for a call that wants the first
    /// `2 + captures` results only: the match may have more captures, which
    /// the call drops, so a pattern of another capture count does not miss.
    /// The others must be the shape's.
    #[must_use]
    pub const fn string_find_prefix(captures: usize, kinds: usize) -> Self {
        Self(Self::STRING_FIND_PREFIX_BASE + shape_offset(captures, 0, kinds))
    }

    /// [`Self::string_match`] for a call that wants the first `results`
    /// results only (see [`Self::string_find_prefix`]).
    #[must_use]
    pub const fn string_match_prefix(results: usize, kinds: usize) -> Self {
        Self(Self::STRING_MATCH_PREFIX_BASE + shape_offset(results, 1, kinds))
    }

    /// [`Self::gmatch_next`] for a loop that wants the first `results`
    /// results only (see [`Self::string_find_prefix`]).
    #[must_use]
    pub const fn gmatch_next_prefix(results: usize, kinds: usize) -> Self {
        Self(Self::GMATCH_NEXT_PREFIX_BASE + shape_offset(results, 1, kinds))
    }

    /// Whether this is a step of a `string.gmatch` iterator: one that gives a
    /// match or the one that ends the iteration.
    #[must_use]
    pub const fn is_gmatch_step(self) -> bool {
        self.0 == Self::GMATCH_END.0
            || (self.0 >= Self::GMATCH_NEXT_BASE && self.0 <= Self::gmatch_next(4, 15).0)
            || (self.0 >= Self::GMATCH_NEXT_PREFIX_BASE
                && self.0 <= Self::gmatch_next_prefix(4, 15).0)
    }

    /// Whether this is a `string.find`, `string.match`, `string.gmatch`,
    /// `utf8` or `string.unpack` helper, whose miss is the call going another way than the
    /// recording (a match where there was none, the other shape of results,
    /// or the end of an iteration). A side trace records the call as it now
    /// ends.
    #[must_use]
    pub const fn is_search_outcome(self) -> bool {
        self.0 == Self::STRING_FIND.0
            || self.0 == Self::STRING_FIND_ABSENT.0
            || (self.0 >= Self::STRING_FIND_PATTERN_BASE && self.0 <= Self::GMATCH_END.0)
            || (self.0 >= Self::STRING_FIND_PREFIX_BASE && self.0 < Self::STRING_NUMBER_INTEGER.0)
            || self.is_text_outcome()
            || self.is_next_outcome()
            || self.is_metamethod_check()
    }

    /// Whether this is a store that keeps the table's layout and misses on
    /// a key that went another way than the recording: an overwrite that
    /// finds the key deleted, or a revival that finds it live. The miss
    /// changes nothing, so a side trace records the store as the interpreter
    /// now finds it, and a loop that fills a table it half emptied runs both
    /// stores natively.
    #[must_use]
    pub const fn is_store_outcome(self) -> bool {
        self.0 == Self::SET_PRESENT.0 || self.0 == Self::SET_REVIVE.0
    }

    /// Whether this helper's miss is a branch of the guest code that a side
    /// trace continues: a search that went another way
    /// ([`Self::is_search_outcome`]), a store that found its key another way
    /// ([`Self::is_store_outcome`]), or the end of an unrolled `ipairs` loop
    /// that found the list longer ([`Self::TABLE_ABSENT_INDEX`]).
    #[must_use]
    pub const fn miss_takes_side(self) -> bool {
        self.is_search_outcome() || self.is_store_outcome() || self.0 == Self::TABLE_ABSENT_INDEX.0
    }

    /// Whether this is a `string.find` helper of any shape.
    #[must_use]
    pub const fn is_find_call(self) -> bool {
        self.0 == Self::STRING_FIND.0
            || self.0 == Self::STRING_FIND_ABSENT.0
            || (self.0 >= Self::STRING_FIND_PATTERN_BASE && self.0 < Self::STRING_MATCH_BASE)
            || (self.0 >= Self::STRING_FIND_PREFIX_BASE && self.0 < Self::STRING_MATCH_PREFIX_BASE)
    }

    /// Whether this is a call of `string.find`, `string.match`,
    /// `string.gmatch` (the iterator or a step of it) or `string.gsub` that
    /// searches a string with a pattern.
    #[must_use]
    pub const fn is_pattern_call(self) -> bool {
        (self.0 >= Self::STRING_FIND_PATTERN_BASE && self.0 <= Self::STRING_GSUB.0)
            || (self.0 >= Self::STRING_FIND_PREFIX_BASE && self.0 < Self::STRING_NUMBER_INTEGER.0)
    }

    /// Whether this checks which metamethod, if any, an access runs
    /// ([`Self::INDEX_HANDLER`], [`Self::NEWINDEX_HANDLER`],
    /// [`Self::LEN_HANDLER`]) or stores under a string key
    /// ([`Self::TABLE_SET_STRING`]). Its miss is the access going another way
    /// than the recording: a key the table holds, a handler of another
    /// class, a handler that is no Lua closure.
    #[must_use]
    pub const fn is_metamethod_check(self) -> bool {
        self.0 >= Self::INDEX_HANDLER.0 && self.0 <= Self::TABLE_SET_STRING.0
    }

    /// Whether this makes or steps a `string.gmatch` iterator, a closure
    /// that holds a lane of the run's bank.
    #[must_use]
    pub const fn uses_iterator(self) -> bool {
        (self.0 >= Self::GMATCH_NEXT_BASE && self.0 <= Self::GMATCH_NEW.0)
            || (self.0 >= Self::GMATCH_NEXT_PREFIX_BASE && self.0 < Self::STRING_NUMBER_INTEGER.0)
    }

    /// The first id of the helpers only tests use, far above the ones
    /// [`HELPERS`] numbers.
    #[cfg(test)]
    pub const TEST_BASE: u16 = 0xFF00;
    /// Quotient and remainder of two integers, floored, giving two results.
    /// It exists to exercise the multi-result convention in tests; the
    /// recorder never emits it.
    #[cfg(test)]
    pub const TEST_DIVMOD: Self = Self(Self::TEST_BASE);
    /// Stores its argument in the run's scratch word and gives it back, so a
    /// test can tell a helper that writes the heap from one that does not.
    /// The recorder never emits it.
    #[cfg(test)]
    pub const TEST_WRITE: Self = Self(Self::TEST_BASE + 1);
    /// Gives its argument back and is flagged as reading the heap, so a test
    /// can tell a helper that reads from one that writes. The recorder never
    /// emits it.
    #[cfg(test)]
    pub const TEST_READ: Self = Self(Self::TEST_BASE + 2);
    /// Gives its argument plus one back and is flagged as advancing private
    /// state, like a random draw. The recorder never emits it.
    #[cfg(test)]
    pub const TEST_DRAW: Self = Self(Self::TEST_BASE + 3);
    /// Swaps the array words of a test table for the spare pair after them
    /// (see `jit::trace::helpers::TEST_TABLE_WORDS`) and is flagged as
    /// rewriting arrays, so a test can move an array under a trace. Its
    /// argument is the table's address. The recorder never emits it.
    #[cfg(test)]
    pub const TEST_RESIZE: Self = Self(Self::TEST_BASE + 4);

    /// The table entry, or `None` for an id no release defines.
    #[must_use]
    pub fn info(self) -> Option<&'static HelperInfo> {
        if (self.0 as usize) < HELPER_COUNT {
            return Some(&HELPERS[self.0 as usize]);
        }
        #[cfg(test)]
        if self.0 >= Self::TEST_BASE && ((self.0 - Self::TEST_BASE) as usize) < TEST_HELPERS.len() {
            return Some(&TEST_HELPERS[(self.0 - Self::TEST_BASE) as usize]);
        }
        None
    }

    /// The effect flags of the helper. An unknown id gets the worst case, a
    /// write to the heap, so nothing is assumed about a call nobody defines.
    #[must_use]
    pub fn effects(self) -> HelperEffects {
        match self.info() {
            Some(info) => info.effects,
            None => HelperEffects::WRITES_HEAP,
        }
    }

    /// The id of the helper called `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let production = HELPERS.iter().position(|info| info.name == name);
        if let Some(index) = production {
            return u16::try_from(index).ok().map(Self);
        }
        #[cfg(test)]
        if let Some(index) = TEST_HELPERS.iter().position(|info| info.name == name) {
            return u16::try_from(index)
                .ok()
                .map(|index| Self(Self::TEST_BASE + index));
        }
        None
    }
}

/// The types a key or a value of `next` has in a helper's result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NextKind {
    Integer = 0,
    Float = 1,
    Boolean = 2,
    String = 3,
    Table = 4,
}

/// How many [`NextKind`]s there are.
pub const NEXT_KINDS: usize = 5;
/// The value slot of a `next` helper that wants the key only.
const NEXT_NO_VALUE: u16 = 5;
/// The helpers of one key kind: a value kind each, and the key-only one.
pub const NEXT_ROW: u16 = 6;
/// The value slot of a lane-keeping `next` helper that stores the value in
/// the lane after the control lane.
const NEXT_VALUE_LANE: u16 = 6;
/// The helpers of one key kind in the lane-keeping family: a value kind each,
/// the key-only one, and the one that keeps the value in a lane.
pub const NEXT_BOX_ROW: u16 = 7;
/// Most values a `coroutine.resume` helper gives besides the leading boolean.
pub const MAX_RESUME_VALUES: usize = 3;

/// Most arguments a `coroutine.resume` helper passes to the thread.
pub const MAX_RESUME_ARGS: usize = MAX_HELPER_ARGS - 2;

/// How many shapes of result [`HelperId::coroutine_resume`] and
/// [`HelperId::coroutine_wrap`] each have: every count of values up to [`MAX_RESUME_VALUES`] with every
/// combination of three kinds.
pub const RESUME_SHAPES: usize = 1 + 3 + 9 + 27;

/// How many helpers there are: one more than the last id.
pub const HELPER_COUNT: usize = HelperId::LAST.0 as usize + 1;

/// Most string keys of a shape a [`HelperId::TABLE_NEW`] call names.
pub const MAX_NEW_TABLE_KEYS: usize = MAX_HELPER_ARGS - 1;

/// A bit of the argument count native code hands [`HelperId::TABLE_NEW`]:
/// the call names no key and the code fills every one of its first
/// [`MAX_PREFILLED_ARRAY`] array slots itself, so the helper makes the array
/// part that long, with nil slots, and returns the address of the first.
pub const TABLE_NEW_PREFILL: i64 = 1 << 8;

/// Most array slots a [`TABLE_NEW_PREFILL`] call makes.
pub const MAX_PREFILLED_ARRAY: usize = 8;

/// What a step of a lane-keeping traversal does with the entry's value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxedValue {
    /// The loop has no variable for it: the step does not look at it.
    Skipped,
    /// The loop has a variable the body never reads: the step stores the
    /// value in the lane after the control lane, which an exit restores the
    /// variable from, and gives no result.
    Lane,
    /// The loop reads the variable: the step gives the value as a result.
    Kind(NextKind),
}

impl NextKind {
    /// The kind of a result of type `ty`, if `next` gives one.
    #[must_use]
    pub const fn of(ty: TraceType) -> Option<Self> {
        match ty {
            TraceType::Integer => Some(Self::Integer),
            TraceType::Float => Some(Self::Float),
            TraceType::Boolean => Some(Self::Boolean),
            TraceType::String => Some(Self::String),
            TraceType::Table => Some(Self::Table),
            _ => None,
        }
    }

    /// The representation of a result of this kind.
    #[must_use]
    pub const fn helper_type(self) -> HelperType {
        match self {
            Self::Integer => HelperType::Integer,
            Self::Float => HelperType::Float,
            Self::Boolean => HelperType::Boolean,
            Self::String => HelperType::String,
            Self::Table => HelperType::Table,
        }
    }

    /// Every kind, in the order of the helper ids.
    pub const ALL: [Self; NEXT_KINDS] = [
        Self::Integer,
        Self::Float,
        Self::Boolean,
        Self::String,
        Self::Table,
    ];
}

/// Most code points a `utf8.codepoint` helper gives.
pub const MAX_UTF8_CODEPOINTS: usize = MAX_HELPER_RESULTS;

/// Most values a `string.unpack` helper gives besides the next position.
pub const MAX_UNPACK_VALUES: usize = MAX_HELPER_RESULTS - 1;

/// Most values a `string.pack` helper takes besides the format.
pub const MAX_PACK_VALUES: usize = MAX_HELPER_ARGS - 1;

/// Most captures the results of a `string.find` helper hold after the start
/// and the end of the match.
pub const MAX_FIND_CAPTURES: usize = MAX_HELPER_RESULTS - 2;

/// Where the helper of `count` results with the kinds `kinds` stands in its
/// family, which lists the counts from `first` up, each with every
/// combination of kinds.
#[allow(
    clippy::cast_possible_truncation,
    reason = "a family has a few dozen helpers"
)]
const fn shape_offset(count: usize, first: usize, kinds: usize) -> u16 {
    assert!(count >= first && kinds < (1 << count));
    ((1 << count) - (1 << first) + kinds) as u16
}

/// Results of the shaped helpers, one row per set of kinds (bit `i` set: the
/// `i`th capture is a string, otherwise a position). `lead` results of
/// positions come first.
const fn shape_rows(lead: usize) -> [[HelperType; MAX_HELPER_RESULTS]; 16] {
    let mut rows = [[HelperType::Integer; MAX_HELPER_RESULTS]; 16];
    let mut kinds = 0;
    while kinds < 16 {
        let mut capture = 0;
        while lead + capture < MAX_HELPER_RESULTS {
            if kinds & (1 << capture) != 0 {
                rows[kinds][lead + capture] = HelperType::String;
            }
            capture += 1;
        }
        kinds += 1;
    }
    rows
}

/// Results of the `string.unpack` helpers, one row per set of kinds (the
/// `i`th base-3 digit: 0 an integer, 1 a float, 2 a string). The places past
/// the values are integers: the position comes right after the last value.
const fn unpack_rows() -> [[HelperType; MAX_HELPER_RESULTS]; 27] {
    let mut rows = [[HelperType::Integer; MAX_HELPER_RESULTS]; 27];
    let mut kinds = 0;
    while kinds < 27 {
        let mut digits = kinds;
        let mut position = 0;
        while position < MAX_UNPACK_VALUES {
            rows[kinds][position] = match digits % 3 {
                0 => HelperType::Integer,
                1 => HelperType::Float,
                _ => HelperType::String,
            };
            digits /= 3;
            position += 1;
        }
        kinds += 1;
    }
    rows
}

/// Results of the `coroutine.resume` helpers, one row per set of kinds (the
/// `i`th base-3 digit: 0 an integer, 1 a float, 2 a boolean).
const fn resume_rows() -> [[HelperType; MAX_HELPER_RESULTS]; 27] {
    let mut rows = [[HelperType::Integer; MAX_HELPER_RESULTS]; 27];
    let mut kinds = 0;
    while kinds < 27 {
        let mut digits = kinds;
        let mut position = 0;
        while position < MAX_RESUME_VALUES {
            rows[kinds][position] = match digits % 3 {
                0 => HelperType::Integer,
                1 => HelperType::Float,
                _ => HelperType::Boolean,
            };
            digits /= 3;
            position += 1;
        }
        kinds += 1;
    }
    rows
}

/// Results of the [`HelperId::vm_call`] helpers, one row per set of kinds
/// (the `i`th base-6 digit: 0 an integer, 1 a float, 2 a boolean, 3 nil, 4 a
/// table, 5 a string). Nil has no word to carry; its place reads as an
/// integer the recorder never uses.
const fn vm_call_rows() -> [[HelperType; MAX_HELPER_RESULTS]; 216] {
    let mut rows = [[HelperType::Integer; MAX_HELPER_RESULTS]; 216];
    let mut kinds = 0;
    while kinds < 216 {
        let mut digits = kinds;
        let mut position = 0;
        while position < MAX_RESUME_VALUES {
            rows[kinds][position] = match digits % 6 {
                1 => HelperType::Float,
                2 => HelperType::Boolean,
                4 => HelperType::Table,
                5 => HelperType::String,
                _ => HelperType::Integer,
            };
            digits /= 6;
            position += 1;
        }
        kinds += 1;
    }
    rows
}

static VM_CALL_ROWS: [[HelperType; MAX_HELPER_RESULTS]; 216] = vm_call_rows();
static UNPACK_ROWS: [[HelperType; MAX_HELPER_RESULTS]; 27] = unpack_rows();
static RESUME_ROWS: [[HelperType; MAX_HELPER_RESULTS]; 27] = resume_rows();
static MATCH_ROWS: [[HelperType; MAX_HELPER_RESULTS]; 16] = shape_rows(0);
static FIND_ROWS: [[HelperType; MAX_HELPER_RESULTS]; 16] = shape_rows(2);

/// How a helper argument or result is represented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HelperType {
    /// An integer, as `i64`.
    Integer,
    /// A float, as `f64`.
    Float,
    /// A boolean, widened to `i64` (0 or 1).
    Boolean,
    /// A string lane of the run's bank or roots, as `i64`.
    String,
    /// The address of a table slot holding a string, as `i64`.
    StringRef,
    /// A table lane of the run's bank, as `i64`.
    Table,
    /// A callable the bank holds (a closure, a native or a `string.gmatch`
    /// iterator), as its lane in `i64`.
    Closure,
    /// A userdata the bank holds, as its lane in `i64`. Results only.
    Userdata,
    /// Any scalar the trace holds (an integer, a float, a boolean, nil, a
    /// string, a borrowed string or a table), as an entry tag and its bits,
    /// two `i64`. Parameters only.
    Value,
}

impl HelperType {
    /// The trace type a value of this representation has. A
    /// [`Self::Value`] has no one type and answers nil: use
    /// [`Self::accepts`] for a parameter.
    #[must_use]
    pub const fn trace_type(self) -> TraceType {
        match self {
            Self::Integer => TraceType::Integer,
            Self::Float => TraceType::Float,
            Self::Boolean => TraceType::Boolean,
            Self::String => TraceType::String,
            Self::StringRef => TraceType::StringRef,
            Self::Table => TraceType::Table,
            Self::Closure => TraceType::Closure,
            Self::Userdata => TraceType::Userdata,
            Self::Value => TraceType::Nil,
        }
    }

    /// Whether an operand of type `ty` can be passed for this parameter.
    #[must_use]
    pub fn accepts(self, ty: TraceType) -> bool {
        match self {
            Self::Value => matches!(
                ty,
                TraceType::Integer
                    | TraceType::Float
                    | TraceType::Boolean
                    | TraceType::Nil
                    | TraceType::String
                    | TraceType::StringRef
                    | TraceType::Table
                    | TraceType::Closure
                    | TraceType::Thread
            ),
            _ => self.trace_type() == ty,
        }
    }
}

/// What a helper does to the heap, as flags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HelperEffects(u16);

impl HelperEffects {
    /// Reads and writes nothing but its arguments. Excludes the two heap
    /// flags.
    pub const PURE: Self = Self(1);
    /// Reads table storage or strings.
    pub const READS_HEAP: Self = Self(2);
    /// Writes table storage. A cached table read does not survive the call.
    pub const WRITES_HEAP: Self = Self(4);
    /// Makes strings (lanes of the run's bank) or other heap objects.
    pub const ALLOCATES: Self = Self(8);
    /// May miss, and then the code leaves through the call's guard.
    pub const MAY_EXIT: Self = Self(16);
    /// With [`Self::WRITES_HEAP`]: the writes move no table storage. The
    /// array and hash parts and the addresses of their slots stay where they
    /// were, so bound words, rows and borrowed strings survive the call
    /// (`setmetatable` changes a header word and nothing else).
    pub const KEEPS_STORAGE: Self = Self(32);
    /// With [`Self::WRITES_HEAP`]: the call changes a table's metatable. A
    /// trace that holds a fact an entry check established about a metatable
    /// (a table without one, one without `__len`, a field found through
    /// `__index`) cannot also hold such a call: the fact would go stale in the
    /// middle of a run.
    pub const WRITES_METATABLE: Self = Self(64);
    /// With [`Self::WRITES_HEAP`]: the call moves the array parts of tables
    /// and nothing else. Hash parts, field slots and the rows of tables met
    /// by lane stay where they were, so a trace may hold those beside such a
    /// call (unlike a general table write, see [`Self::moves_storage`]).
    /// The array part and the border of a table are another matter: the
    /// call takes the address of the run's binding slots right after `out`,
    /// and rewrites the words of every array and length binding of the table
    /// it moved, so native code reads them again afterwards.
    pub const REWRITES_ARRAYS: Self = Self(128);
    /// With [`Self::WRITES_HEAP`] and [`Self::KEEPS_STORAGE`]: the call
    /// deletes a key and leaves the entry where it was, as a nil the
    /// traversals skip, or fills such a nil in place
    /// ([`HelperId::SET_REVIVE`]). Bound field slots and the rows of tables met by lane
    /// stay in place but a value they were typed for may now be nil, so a
    /// trace holds no field binding or lane access beside such a call (see
    /// `TraceIr::deletes_keys`). Traversal snapshots stay valid.
    pub const DELETES_KEYS: Self = Self(512);
    /// With [`Self::WRITES_HEAP`]: the call moves the hash parts of tables
    /// and nothing else (a store under a string key). The array parts, the
    /// slots in them and so the strings borrowed from them stay where they
    /// were; field slots, the rows of tables met by lane and anything else
    /// in a hash part may move, as for any write that does not keep storage.
    pub const REWRITES_HASH: Self = Self(1024);
    /// Advances state of the run that no table holds (`math.random`'s
    /// generator). The call commits an effect, so it stays in order against
    /// every other such call and a snapshot after it resumes after it, and it
    /// is never merged, dropped or hoisted. Without [`Self::WRITES_HEAP`] it
    /// leaves every cached table read and every table binding valid.
    pub const PRIVATE_STATE: Self = Self(256);
    /// Makes a table, or fills one just made ([`HelperId::TABLE_NEW`] and
    /// [`HelperId::TABLE_INIT`]). Every call stands for a new object, so it
    /// is never merged or hoisted out of the loop, and the preheader holds
    /// none. Unlike a store it commits nothing a guest can see: the table is
    /// named by no register until the trace stores it, so a guard that
    /// leaves after the call and replays the instruction at its `Before`
    /// snapshot only abandons it.
    pub const FRESH_TABLE: Self = Self(2048);
    /// With [`Self::WRITES_HEAP`]: the call runs another coroutine to its
    /// next switch, or makes a host C function call ([`HelperId::vm_call`]),
    /// which runs any Lua code. Every table, cell and global
    /// may be written, metatables included, and any hook may be installed.
    /// The facts a binding takes from a metatable (no `__len`, a field found
    /// through `__index`) are checked again by the call itself, so it does
    /// not carry [`Self::WRITES_METATABLE`].
    /// The call moves storage like any write that does not promise to keep
    /// it, and takes the address of the run's binding slots right after
    /// `out`, like a helper that [`Self::REWRITES_ARRAYS`], because it
    /// rewrites the words of every array, length and header binding of the
    /// run itself. A trace that holds anything a coroutine could invalidate
    /// beside such a call is refused ([`TraceIr::resume_conflict`](super::TraceIr::resume_conflict)).
    pub const RESUMES: Self = Self(4096);
    /// With [`Self::WRITES_HEAP`]: the call takes the address of the run's
    /// binding slots right after `out`, as a helper that
    /// [`Self::REWRITES_ARRAYS`] does, and rewrites the words of every array
    /// and length binding of the one table it stored into, but it makes no
    /// promise about hash parts (`TABLE_SET_INDEX` may also move those).
    /// The table it stores into can be met by lane while a binding names the
    /// same table (the `t` a `__newindex` receives), so the words are
    /// rewritten for whichever table the store reached.
    pub const REWRITES_WORDS: Self = Self(8192);

    /// The flags of `self` and `other` together.
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every flag of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether a call commits an effect in body order: it writes the heap or
    /// advances private state.
    #[must_use]
    pub const fn commits_effect(self) -> bool {
        self.contains(Self::WRITES_HEAP) || self.contains(Self::PRIVATE_STATE)
    }

    /// Whether the helper may grow, shrink or replace the storage of any
    /// table: it writes the heap and does not promise to keep storage.
    #[must_use]
    pub const fn moves_storage(self) -> bool {
        self.contains(Self::WRITES_HEAP) && !self.contains(Self::KEEPS_STORAGE)
    }

    /// Whether the call deletes a key in place ([`Self::DELETES_KEYS`]).
    #[must_use]
    pub const fn deletes_keys(self) -> bool {
        self.contains(Self::WRITES_HEAP) && self.contains(Self::DELETES_KEYS)
    }

    /// Whether the call takes the address of the run's binding slots right
    /// after `out`: it rewrites binding words ([`Self::REWRITES_ARRAYS`],
    /// [`Self::REWRITES_WORDS`]) or runs a coroutine ([`Self::RESUMES`]).
    #[must_use]
    pub const fn takes_words(self) -> bool {
        self.contains(Self::REWRITES_ARRAYS)
            || self.contains(Self::REWRITES_WORDS)
            || self.contains(Self::RESUMES)
    }

    /// Whether the call runs another coroutine or a host function
    /// ([`Self::RESUMES`]).
    #[must_use]
    pub const fn resumes(self) -> bool {
        self.contains(Self::RESUMES)
    }

    /// Whether the call may end the lifetime of the strings borrowed from
    /// array slots: it moves storage and does not promise to move hash parts
    /// only ([`Self::REWRITES_HASH`]).
    #[must_use]
    pub const fn ends_borrows(self) -> bool {
        self.moves_storage() && !self.contains(Self::REWRITES_HASH)
    }

    /// Whether the helper may move hash storage: it moves storage and does
    /// not promise to move array parts only ([`Self::REWRITES_ARRAYS`]).
    #[must_use]
    pub const fn moves_hash_storage(self) -> bool {
        self.moves_storage() && !self.contains(Self::REWRITES_ARRAYS)
    }
}

/// One entry of [`HELPERS`].
#[derive(Clone, Copy, Debug)]
pub struct HelperInfo {
    /// The library-qualified name, for diagnostics and lookups by name.
    pub name: &'static str,
    pub params: &'static [HelperType],
    /// How many leading parameters a call must pass. Fewer than
    /// `params.len()` makes the helper variadic: the trailing parameters are
    /// all [`HelperType::Value`], and the helper is told how many were given.
    pub required: usize,
    pub results: &'static [HelperType],
    pub effects: HelperEffects,
    /// Whether the helper takes the run's context address first.
    pub context: bool,
}

/// The helpers, indexed by [`HelperId`].
///
/// Built on first use, not in the image: a static table of slices and names
/// is 26 KB of relocated data, which the dynamic loader rewrites on every
/// start (7 page faults and about 1,200 relocations) before any trace is
/// recorded.
pub static HELPERS: LazyLock<[HelperInfo; HELPER_COUNT]> = LazyLock::new(assemble);

/// Helpers of the families that have an entry for every shape of result
/// ([`HelperId::string_find_pattern`], [`HelperId::string_match`] and
/// [`HelperId::gmatch_next`], and their prefix twins) share a name.
#[allow(
    clippy::large_stack_arrays,
    reason = "const evaluation builds the helper table once, at compile time"
)]
fn assemble() -> [HelperInfo; HELPER_COUNT] {
    let mut table = [BASE_HELPERS[0]; HELPER_COUNT];
    let mut index = 0;
    while index < BASE_HELPERS.len() {
        table[index] = BASE_HELPERS[index];
        index += 1;
    }
    let mut prefixed = 0;
    while prefixed < 2 {
        let mut captures = 0;
        while captures <= MAX_FIND_CAPTURES {
            let mut kinds = 0;
            while kinds < (1 << captures) {
                let at = if prefixed == 0 {
                    HelperId::string_find_pattern(captures, kinds)
                } else {
                    HelperId::string_find_prefix(captures, kinds)
                }
                .0 as usize;
                table[at] = HelperInfo {
                    name: "string.find_pattern",
                    params: &[
                        HelperType::Value,
                        HelperType::Value,
                        HelperType::Value,
                        HelperType::Value,
                    ],
                    required: 2,
                    results: FIND_ROWS[kinds].split_at(2 + captures).0,
                    effects: HelperEffects::PURE
                        .with(HelperEffects::ALLOCATES)
                        .with(HelperEffects::MAY_EXIT),
                    context: true,
                };
                kinds += 1;
            }
            captures += 1;
        }
        let mut results = 1;
        while results <= MAX_HELPER_RESULTS {
            let mut kinds = 0;
            while kinds < (1 << results) {
                let at = if prefixed == 0 {
                    HelperId::string_match(results, kinds)
                } else {
                    HelperId::string_match_prefix(results, kinds)
                }
                .0 as usize;
                table[at] = HelperInfo {
                    name: "string.match",
                    params: &[HelperType::Value, HelperType::Value, HelperType::Value],
                    required: 2,
                    results: MATCH_ROWS[kinds].split_at(results).0,
                    effects: HelperEffects::PURE
                        .with(HelperEffects::ALLOCATES)
                        .with(HelperEffects::MAY_EXIT),
                    context: true,
                };
                let at = if prefixed == 0 {
                    HelperId::gmatch_next(results, kinds)
                } else {
                    HelperId::gmatch_next_prefix(results, kinds)
                }
                .0 as usize;
                table[at] = HelperInfo {
                    name: "gmatch.next",
                    params: &[HelperType::Closure],
                    required: 1,
                    results: MATCH_ROWS[kinds].split_at(results).0,
                    effects: HelperEffects::PRIVATE_STATE
                        .with(HelperEffects::ALLOCATES)
                        .with(HelperEffects::MAY_EXIT),
                    context: true,
                };
                kinds += 1;
            }
            results += 1;
        }
        prefixed += 1;
    }
    table[HelperId::STRING_MATCH_ABSENT.0 as usize] = HelperInfo {
        name: "string.match_absent",
        params: &[HelperType::Value, HelperType::Value, HelperType::Value],
        required: 2,
        results: &[],
        effects: HelperEffects::PURE.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::SET_STRING.0 as usize] = HelperInfo {
        name: "table.set_string",
        params: &[HelperType::Table, HelperType::Value, HelperType::Value],
        required: 3,
        results: &[],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_HASH)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::SET_PRESENT.0 as usize] = HelperInfo {
        name: "table.set_present",
        params: &[HelperType::Table, HelperType::Value, HelperType::Value],
        required: 3,
        results: &[],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::KEEPS_STORAGE)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::DELETE_STRING.0 as usize] = HelperInfo {
        name: "table.delete_string",
        params: &[HelperType::Table, HelperType::Value],
        required: 2,
        results: &[],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::KEEPS_STORAGE)
            .with(HelperEffects::DELETES_KEYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::CALL_HANDLER.0 as usize] = HelperInfo {
        name: "call.handler",
        params: &[HelperType::Value, HelperType::Integer],
        required: 2,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::INDEX_HANDLER.0 as usize] = HelperInfo {
        name: "index.handler",
        params: &[HelperType::Value, HelperType::Value, HelperType::Integer],
        required: 3,
        results: &[HelperType::Table],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::NEWINDEX_HANDLER.0 as usize] = HelperInfo {
        name: "newindex.handler",
        params: &[HelperType::Value, HelperType::Value, HelperType::Integer],
        required: 3,
        results: &[HelperType::Table],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::LEN_HANDLER.0 as usize] = HelperInfo {
        name: "len.handler",
        params: &[HelperType::Value, HelperType::Integer],
        required: 2,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::SET_REVIVE.0 as usize] = HelperInfo {
        name: "table.set_revive",
        params: &[HelperType::Table, HelperType::Value, HelperType::Value],
        required: 3,
        results: &[],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::KEEPS_STORAGE)
            .with(HelperEffects::DELETES_KEYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::TABLE_NEW.0 as usize] = HelperInfo {
        name: "table.new",
        params: &[
            HelperType::Integer,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 1,
        results: &[HelperType::Table],
        effects: HelperEffects::ALLOCATES
            .with(HelperEffects::MAY_EXIT)
            .with(HelperEffects::FRESH_TABLE),
        context: true,
    };
    table[HelperId::TABLE_INIT.0 as usize] = HelperInfo {
        name: "table.init",
        params: &[HelperType::Table, HelperType::Value, HelperType::Value],
        required: 3,
        results: &[],
        effects: HelperEffects::ALLOCATES
            .with(HelperEffects::MAY_EXIT)
            .with(HelperEffects::FRESH_TABLE),
        context: true,
    };
    table[HelperId::STRING_OWN.0 as usize] = HelperInfo {
        name: "string.own",
        params: &[HelperType::StringRef],
        required: 1,
        results: &[HelperType::String],
        // It makes a lane and reads the slot, and changes nothing a guest can
        // see: it does not end the borrows it is there to end.
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::CLOSURE_SAME_ENV.0 as usize] = HelperInfo {
        name: "closure.same_env",
        params: &[HelperType::Closure, HelperType::Closure],
        required: 2,
        results: &[HelperType::Boolean],
        // `setfenv` changes a closure's environment: the check keeps its
        // place among the heap's writes.
        effects: HelperEffects::READS_HEAP,
        context: true,
    };
    table[HelperId::CLOSURE_SAME_PROTO.0 as usize] = HelperInfo {
        name: "closure.same_proto",
        params: &[HelperType::Closure, HelperType::Closure],
        required: 2,
        results: &[HelperType::Boolean],
        effects: HelperEffects::PURE,
        context: true,
    };
    let table_reads = [
        (
            HelperId::TABLE_GET_INTEGER,
            "table.get_integer",
            HelperType::Integer,
        ),
        (
            HelperId::TABLE_GET_FLOAT,
            "table.get_float",
            HelperType::Float,
        ),
        (
            HelperId::TABLE_GET_BOOLEAN,
            "table.get_boolean",
            HelperType::Boolean,
        ),
        (
            HelperId::TABLE_GET_TABLE,
            "table.get_table",
            HelperType::Table,
        ),
        (
            HelperId::TABLE_GET_CLOSURE,
            "table.get_closure",
            HelperType::Closure,
        ),
        (
            HelperId::TABLE_GET_STRING,
            "table.get_string",
            HelperType::String,
        ),
        (
            HelperId::TABLE_GET_USERDATA,
            "table.get_userdata",
            HelperType::Userdata,
        ),
    ];
    let mut get = 0;
    while get < table_reads.len() {
        let (id, name, result) = table_reads[get];
        table[id.0 as usize] = HelperInfo {
            name,
            params: &[HelperType::Value, HelperType::Value, HelperType::Integer],
            required: 3,
            results: match result {
                HelperType::Integer => &[HelperType::Integer],
                HelperType::Float => &[HelperType::Float],
                HelperType::Boolean => &[HelperType::Boolean],
                HelperType::Table => &[HelperType::Table],
                HelperType::Closure => &[HelperType::Closure],
                HelperType::Userdata => &[HelperType::Userdata],
                _ => &[HelperType::String],
            },
            effects: if matches!(result, HelperType::String) {
                HelperEffects::READS_HEAP
                    .with(HelperEffects::ALLOCATES)
                    .with(HelperEffects::MAY_EXIT)
            } else {
                HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT)
            },
            context: true,
        };
        get += 1;
    }
    table[HelperId::TABLE_ABSENT_INDEX.0 as usize] = HelperInfo {
        name: "table.absent_index",
        params: &[HelperType::Value, HelperType::Value],
        required: 2,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::TABLE_SET_INDEX.0 as usize] = HelperInfo {
        name: "table.set_index",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Integer,
        ],
        required: 4,
        results: &[],
        effects: HelperEffects::WRITES_HEAP
            .with(HelperEffects::MAY_EXIT)
            .with(HelperEffects::REWRITES_WORDS),
        context: true,
    };
    table[HelperId::TABLE_LENGTH.0 as usize] = HelperInfo {
        name: "table.length",
        params: &[HelperType::Value, HelperType::Integer],
        required: 2,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    let cell_reads = [
        (
            HelperId::CELL_GET_INTEGER,
            "cell.get_integer",
            HelperType::Integer,
        ),
        (
            HelperId::CELL_GET_FLOAT,
            "cell.get_float",
            HelperType::Float,
        ),
        (
            HelperId::CELL_GET_BOOLEAN,
            "cell.get_boolean",
            HelperType::Boolean,
        ),
        (
            HelperId::CELL_GET_TABLE,
            "cell.get_table",
            HelperType::Table,
        ),
        (
            HelperId::CELL_GET_CALLABLE,
            "cell.get_callable",
            HelperType::Closure,
        ),
        (
            HelperId::CELL_GET_STRING,
            "cell.get_string",
            HelperType::String,
        ),
    ];
    let mut read = 0;
    while read < cell_reads.len() {
        let (id, name, result) = cell_reads[read];
        table[id.0 as usize] = HelperInfo {
            name,
            params: &[HelperType::Closure, HelperType::Integer],
            required: 2,
            results: match result {
                HelperType::Integer => &[HelperType::Integer],
                HelperType::Float => &[HelperType::Float],
                HelperType::Boolean => &[HelperType::Boolean],
                HelperType::Closure => &[HelperType::Closure],
                HelperType::String => &[HelperType::String],
                _ => &[HelperType::Table],
            },
            // A cell is private state of the run's closures: reads and
            // writes of it keep their order, and none is merged or hoisted.
            // A string read takes a lane of the bank, as a table read of a
            // string does.
            effects: if matches!(result, HelperType::String) {
                HelperEffects::PRIVATE_STATE
                    .with(HelperEffects::ALLOCATES)
                    .with(HelperEffects::MAY_EXIT)
            } else {
                HelperEffects::PRIVATE_STATE.with(HelperEffects::MAY_EXIT)
            },
            context: true,
        };
        read += 1;
    }
    table[HelperId::CELL_SET.0 as usize] = HelperInfo {
        name: "cell.set",
        params: &[HelperType::Closure, HelperType::Integer, HelperType::Value],
        required: 3,
        results: &[],
        effects: HelperEffects::PRIVATE_STATE.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::TABLE_SAME.0 as usize] = HelperInfo {
        name: "table.same",
        params: &[HelperType::Table, HelperType::Table],
        required: 2,
        results: &[HelperType::Boolean],
        effects: HelperEffects::PURE,
        context: true,
    };
    table[HelperId::OPERATOR_HANDLER.0 as usize] = HelperInfo {
        name: "operator.handler",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Integer,
            HelperType::Integer,
        ],
        required: 4,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::TABLE_SET_STRING.0 as usize] = HelperInfo {
        name: "table.set_string_walk",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Integer,
        ],
        required: 4,
        results: &[],
        effects: HelperEffects::WRITES_HEAP
            .with(HelperEffects::REWRITES_HASH)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::GMATCH_END.0 as usize] = HelperInfo {
        name: "gmatch.end",
        params: &[HelperType::Closure],
        required: 1,
        results: &[],
        effects: HelperEffects::PRIVATE_STATE.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::GMATCH_NEW.0 as usize] = HelperInfo {
        name: "string.gmatch",
        params: &[HelperType::Value, HelperType::Value, HelperType::Value],
        required: 2,
        results: &[HelperType::Closure],
        effects: HelperEffects::PRIVATE_STATE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::STRING_GSUB.0 as usize] = HelperInfo {
        name: "string.gsub",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 3,
        results: &[HelperType::String, HelperType::Integer],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::STRING_NUMBER_INTEGER.0 as usize] = HelperInfo {
        name: "string.number_integer",
        params: &[HelperType::Integer, HelperType::Value],
        required: 2,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    table[HelperId::STRING_NUMBER_FLOAT.0 as usize] = HelperInfo {
        name: "string.number_float",
        params: &[HelperType::Integer, HelperType::Value],
        required: 2,
        results: &[HelperType::Float],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    };
    assemble_text(&mut table);
    assemble_resume(&mut table);
    assemble_vm_call(&mut table);
    table[HelperId::STRING_SLICE.0 as usize] = table[HelperId::STRING_SUB.0 as usize];
    assemble_next(&mut table);
    assemble_next_box(&mut table);
    table
}

/// The result rows of the `next` helpers: the key, then the value unless the
/// call wants the key only.
static NEXT_ROWS: [[HelperType; 2]; NEXT_KINDS * (NEXT_KINDS + 1)] = next_rows();

const fn next_rows() -> [[HelperType; 2]; NEXT_KINDS * (NEXT_KINDS + 1)] {
    let mut rows = [[HelperType::Integer; 2]; NEXT_KINDS * (NEXT_KINDS + 1)];
    let mut key = 0;
    while key < NEXT_KINDS {
        let mut value = 0;
        while value <= NEXT_KINDS {
            rows[key * (NEXT_KINDS + 1) + value][0] = NextKind::ALL[key].helper_type();
            if value < NEXT_KINDS {
                rows[key * (NEXT_KINDS + 1) + value][1] = NextKind::ALL[value].helper_type();
            }
            value += 1;
        }
        key += 1;
    }
    rows
}

/// The `next` and `pairs` helpers. They read table storage and publish
/// strings and tables as lanes of the run's bank.
const fn assemble_next(table: &mut [HelperInfo; HELPER_COUNT]) {
    const READS: HelperEffects = HelperEffects::READS_HEAP
        .with(HelperEffects::ALLOCATES)
        .with(HelperEffects::MAY_EXIT);
    let mut key = 0;
    while key < NEXT_KINDS {
        let mut value = 0;
        while value <= NEXT_KINDS {
            let row = key * (NEXT_KINDS + 1) + value;
            table[HelperId::NEXT_BASE as usize + row] = HelperInfo {
                name: "table.next",
                params: &[HelperType::Table, HelperType::Value],
                required: 2,
                results: if value == NEXT_KINDS {
                    NEXT_ROWS[row].split_at(1).0
                } else {
                    &NEXT_ROWS[row]
                },
                effects: READS,
                context: true,
            };
            value += 1;
        }
        key += 1;
    }
    table[HelperId::NEXT_END.0 as usize] = plain(
        "table.next_end",
        &[HelperType::Table, HelperType::Value],
        2,
        &[],
        HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
    );
    table[HelperId::PAIRS_PLAIN.0 as usize] = plain(
        "pairs.plain",
        &[HelperType::Table],
        1,
        &[],
        HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
    );
}

/// The result rows of the lane-keeping `next` helpers: the key if the loop
/// wants it, then the value if it wants that.
static NEXT_BOX_ROWS: [[HelperType; 2]; NEXT_ROW as usize * NEXT_BOX_ROW as usize] =
    next_box_rows();

const fn next_box_rows() -> [[HelperType; 2]; NEXT_ROW as usize * NEXT_BOX_ROW as usize] {
    let mut rows = [[HelperType::Integer; 2]; NEXT_ROW as usize * NEXT_BOX_ROW as usize];
    let mut key = 0;
    while key < NEXT_ROW as usize {
        let mut value = 0;
        while value < NEXT_BOX_ROW as usize {
            let row = key * NEXT_BOX_ROW as usize + value;
            let mut at = 0;
            if key < NEXT_KINDS {
                rows[row][at] = NextKind::ALL[key].helper_type();
                at += 1;
            }
            if value < NEXT_KINDS {
                rows[row][at] = NextKind::ALL[value].helper_type();
            }
            value += 1;
        }
        key += 1;
    }
    rows
}

/// The `next` helpers of a traversal whose control value is a lane of the
/// run's bank. They read table storage, replace the lane's value and publish
/// strings and tables as lanes of the bank.
const fn assemble_next_box(table: &mut [HelperInfo; HELPER_COUNT]) {
    const STEPS: HelperEffects = HelperEffects::READS_HEAP
        .with(HelperEffects::PRIVATE_STATE)
        .with(HelperEffects::ALLOCATES)
        .with(HelperEffects::MAY_EXIT);
    let mut key = 0;
    while key < NEXT_ROW as usize {
        let mut value = 0;
        while value < NEXT_BOX_ROW as usize {
            let row = key * NEXT_BOX_ROW as usize + value;
            let wanted = (key < NEXT_KINDS) as usize + (value < NEXT_KINDS) as usize;
            table[HelperId::NEXT_BOX_BASE as usize + row] = HelperInfo {
                name: "table.next_box",
                params: &[HelperType::Table, HelperType::String],
                required: 2,
                results: NEXT_BOX_ROWS[row].split_at(wanted).0,
                effects: STEPS,
                context: true,
            };
            value += 1;
        }
        key += 1;
    }
    table[HelperId::NEXT_BOX_END.0 as usize] = plain(
        "table.next_box_end",
        &[HelperType::Table, HelperType::String],
        2,
        &[],
        HelperEffects::READS_HEAP
            .with(HelperEffects::PRIVATE_STATE)
            .with(HelperEffects::MAY_EXIT),
    );
    table[HelperId::BOX_SET.0 as usize] = plain(
        "table.box_set",
        &[HelperType::String, HelperType::Value],
        2,
        &[],
        HelperEffects::PRIVATE_STATE,
    );
}

/// Six to the power `exponent` (a few at most).
const fn pow6(exponent: usize) -> usize {
    let mut power = 1;
    let mut count = 0;
    while count < exponent {
        power *= 6;
        count += 1;
    }
    power
}

/// Three to the power `exponent` (a few at most).
const fn pow3(exponent: usize) -> usize {
    let mut power = 1;
    let mut count = 0;
    while count < exponent {
        power *= 3;
        count += 1;
    }
    power
}

/// An entry of a helper that takes the run's context.
const fn plain(
    name: &'static str,
    params: &'static [HelperType],
    required: usize,
    results: &'static [HelperType],
    effects: HelperEffects,
) -> HelperInfo {
    HelperInfo {
        name,
        params,
        required,
        results,
        effects,
        context: true,
    }
}

/// The `utf8` and `string.pack` helpers. They read nothing but their
/// arguments (strings are immutable), and the ones that make a string
/// allocate it.
const fn assemble_text(table: &mut [HelperInfo; HELPER_COUNT]) {
    const PURE_EXIT: HelperEffects = HelperEffects::PURE.with(HelperEffects::MAY_EXIT);
    const MAKES_STRING: HelperEffects = PURE_EXIT.with(HelperEffects::ALLOCATES);
    const FIVE: &[HelperType] = &[HelperType::Value; 5];
    const FOUR: &[HelperType] = &[HelperType::Value; 4];
    const THREE: &[HelperType] = &[HelperType::Value; 3];
    const SIX: &[HelperType] = &[HelperType::Value; MAX_HELPER_ARGS];
    const INTEGERS: [HelperType; MAX_HELPER_RESULTS] = [HelperType::Integer; MAX_HELPER_RESULTS];
    table[HelperId::UTF8_CHAR.0 as usize] =
        plain("utf8.char", FIVE, 0, &[HelperType::String], MAKES_STRING);
    let mut count = 0;
    while count <= MAX_UTF8_CODEPOINTS {
        table[HelperId::utf8_codepoint(count).0 as usize] = plain(
            "utf8.codepoint",
            FOUR,
            1,
            INTEGERS.split_at(count).0,
            PURE_EXIT,
        );
        count += 1;
    }
    table[HelperId::UTF8_LEN.0 as usize] =
        plain("utf8.len", FOUR, 1, &[HelperType::Integer], PURE_EXIT);
    table[HelperId::UTF8_LEN_FAIL.0 as usize] =
        plain("utf8.len_fail", FOUR, 1, &[HelperType::Integer], PURE_EXIT);
    table[HelperId::UTF8_OFFSET.0 as usize] =
        plain("utf8.offset", THREE, 2, &[HelperType::Integer], PURE_EXIT);
    table[HelperId::UTF8_OFFSET_END.0 as usize] = plain(
        "utf8.offset_end",
        THREE,
        2,
        &[HelperType::Integer, HelperType::Integer],
        PURE_EXIT,
    );
    table[HelperId::UTF8_OFFSET_NIL.0 as usize] =
        plain("utf8.offset_nil", THREE, 2, &[], PURE_EXIT);
    table[HelperId::UTF8_NEXT.0 as usize] = plain(
        "utf8.next",
        &[HelperType::Integer, HelperType::Value, HelperType::Value],
        3,
        &[HelperType::Integer, HelperType::Integer],
        PURE_EXIT,
    );
    table[HelperId::UTF8_NEXT_END.0 as usize] = plain(
        "utf8.next_end",
        &[HelperType::Integer, HelperType::Value, HelperType::Value],
        3,
        &[],
        PURE_EXIT,
    );
    table[HelperId::UTF8_CODES_CHECK.0 as usize] =
        plain("utf8.codes_check", &[HelperType::Value], 1, &[], PURE_EXIT);
    table[HelperId::STRING_PACK.0 as usize] =
        plain("string.pack", SIX, 1, &[HelperType::String], MAKES_STRING);
    table[HelperId::STRING_PACKSIZE.0 as usize] = plain(
        "string.packsize",
        &[HelperType::Value],
        1,
        &[HelperType::Integer],
        PURE_EXIT,
    );
    let mut values = 0;
    while values <= MAX_UNPACK_VALUES {
        let mut kinds = 0;
        while kinds < pow3(values) {
            let mut digits = kinds;
            let mut strings = false;
            while digits > 0 {
                strings |= digits % 3 == 2;
                digits /= 3;
            }
            table[HelperId::string_unpack(values, kinds).0 as usize] = plain(
                "string.unpack",
                THREE,
                2,
                UNPACK_ROWS[kinds].split_at(values + 1).0,
                if strings { MAKES_STRING } else { PURE_EXIT },
            );
            kinds += 1;
        }
        values += 1;
    }
}

/// The `coroutine.resume` helpers: the thread and what it is passed, in, and
/// the values it hands back, out.
const fn assemble_resume(table: &mut [HelperInfo; HELPER_COUNT]) {
    const ARGS: &[HelperType] = &[HelperType::Value; MAX_RESUME_ARGS + 1];
    const EFFECTS: HelperEffects = HelperEffects::READS_HEAP
        .with(HelperEffects::WRITES_HEAP)
        .with(HelperEffects::RESUMES)
        .with(HelperEffects::MAY_EXIT);
    const WRAPPED: &[HelperType] = &[
        HelperType::Closure,
        HelperType::Value,
        HelperType::Value,
        HelperType::Value,
        HelperType::Value,
    ];
    let mut values = 0;
    while values <= MAX_RESUME_VALUES {
        let mut kinds = 0;
        while kinds < pow3(values) {
            table[HelperId::coroutine_resume(values, kinds).0 as usize] = plain(
                "coroutine.resume",
                ARGS,
                1,
                RESUME_ROWS[kinds].split_at(values).0,
                EFFECTS,
            );
            table[HelperId::coroutine_wrap(values, kinds).0 as usize] = plain(
                "coroutine.wrap",
                WRAPPED,
                1,
                RESUME_ROWS[kinds].split_at(values).0,
                EFFECTS,
            );
            kinds += 1;
        }
        values += 1;
    }
}

/// The `host.call` helpers: no argument, the values the function gave out.
const fn assemble_vm_call(table: &mut [HelperInfo; HELPER_COUNT]) {
    const EFFECTS: HelperEffects = HelperEffects::READS_HEAP
        .with(HelperEffects::WRITES_HEAP)
        .with(HelperEffects::RESUMES)
        .with(HelperEffects::MAY_EXIT);
    let mut values = 0;
    while values <= MAX_RESUME_VALUES {
        let mut kinds = 0;
        while kinds < pow6(values) {
            table[HelperId::vm_call(values, kinds).0 as usize] = plain(
                "host.call",
                &[],
                0,
                VM_CALL_ROWS[kinds].split_at(values).0,
                EFFECTS,
            );
            kinds += 1;
        }
        values += 1;
    }
}

const BASE_HELPERS: [HelperInfo; 38] = [
    HelperInfo {
        name: "string.len",
        params: &[HelperType::String],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.len_ref",
        params: &[HelperType::StringRef],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.char",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "tostring",
        params: &[HelperType::Value],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.format",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.format_luajit",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.format_native",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "concat",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 2,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.find",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 2,
        results: &[HelperType::Integer],
        effects: HelperEffects::PURE.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.find_absent",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 2,
        results: &[],
        effects: HelperEffects::PURE.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.concat",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "getmetatable",
        params: &[HelperType::Value],
        required: 1,
        results: &[HelperType::Table],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "getmetatable_absent",
        params: &[HelperType::Value],
        required: 1,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "setmetatable",
        params: &[HelperType::Table, HelperType::Value],
        required: 2,
        results: &[],
        effects: HelperEffects::WRITES_HEAP
            .with(HelperEffects::KEEPS_STORAGE)
            .with(HelperEffects::WRITES_METATABLE)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "plain_metatable",
        params: &[HelperType::Table, HelperType::Integer],
        required: 2,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.equal",
        params: &[HelperType::Value, HelperType::Value],
        required: 2,
        results: &[HelperType::Boolean],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "concat_error_text",
        params: &[
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 2,
        results: &[HelperType::String],
        effects: HelperEffects::PURE
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.insert",
        params: &[HelperType::Table, HelperType::Value, HelperType::Value],
        required: 2,
        results: &[],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.remove_integer",
        params: &[HelperType::Table, HelperType::Value],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.remove_float",
        params: &[HelperType::Table, HelperType::Value],
        required: 1,
        results: &[HelperType::Float],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.remove_boolean",
        params: &[HelperType::Table, HelperType::Value],
        required: 1,
        results: &[HelperType::Boolean],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.remove_table",
        params: &[HelperType::Table, HelperType::Value],
        required: 1,
        results: &[HelperType::Table],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.remove_nothing",
        params: &[HelperType::Table, HelperType::Value],
        required: 1,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "table.remove_string",
        params: &[HelperType::Table, HelperType::Value],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS)
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.sub",
        params: &[HelperType::Value, HelperType::Value, HelperType::Value],
        required: 2,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.rep",
        params: &[HelperType::Value, HelperType::Value, HelperType::Value],
        required: 2,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.upper",
        params: &[HelperType::Value],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.lower",
        params: &[HelperType::Value],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.reverse",
        params: &[HelperType::Value],
        required: 1,
        results: &[HelperType::String],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::ALLOCATES)
            .with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "math.random_float",
        params: &[HelperType::Value, HelperType::Value],
        required: 0,
        results: &[HelperType::Float],
        effects: HelperEffects::PRIVATE_STATE.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "math.random_integer",
        params: &[HelperType::Value, HelperType::Value],
        required: 0,
        results: &[HelperType::Integer],
        effects: HelperEffects::PRIVATE_STATE.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "math.modf_integer",
        params: &[HelperType::Float],
        required: 1,
        results: &[HelperType::Integer, HelperType::Float],
        effects: HelperEffects::PURE.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "math.modf_float",
        params: &[HelperType::Float],
        required: 1,
        results: &[HelperType::Float, HelperType::Float],
        effects: HelperEffects::PURE.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "tonumber_integer",
        params: &[HelperType::Value, HelperType::Value],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "tonumber_float",
        params: &[HelperType::Value, HelperType::Value],
        required: 1,
        results: &[HelperType::Float],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "tonumber_nil",
        params: &[HelperType::Value, HelperType::Value],
        required: 1,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.byte",
        params: &[
            HelperType::Integer,
            HelperType::Value,
            HelperType::Value,
            HelperType::Value,
        ],
        required: 2,
        results: &[
            HelperType::Integer,
            HelperType::Integer,
            HelperType::Integer,
            HelperType::Integer,
        ],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
    HelperInfo {
        name: "string.method",
        params: &[HelperType::String, HelperType::Integer, HelperType::Value],
        required: 3,
        results: &[],
        effects: HelperEffects::READS_HEAP.with(HelperEffects::MAY_EXIT),
        context: true,
    },
];

/// The helpers only tests use, indexed by [`HelperId`] less
/// [`HelperId::TEST_BASE`]. They are not in [`HELPERS`]: a release build
/// has no such ids.
#[cfg(test)]
pub const TEST_HELPERS: [HelperInfo; 5] = [
    HelperInfo {
        name: "test.divmod",
        params: &[HelperType::Integer, HelperType::Integer],
        required: 2,
        results: &[HelperType::Integer, HelperType::Integer],
        effects: HelperEffects::PURE.with(HelperEffects::MAY_EXIT),
        context: false,
    },
    HelperInfo {
        name: "test.write",
        params: &[HelperType::Integer],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::WRITES_HEAP,
        context: false,
    },
    HelperInfo {
        name: "test.read",
        params: &[HelperType::Integer],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP,
        context: false,
    },
    HelperInfo {
        name: "test.draw",
        params: &[HelperType::Integer],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::PRIVATE_STATE,
        context: false,
    },
    HelperInfo {
        name: "test.resize",
        params: &[HelperType::Integer],
        required: 1,
        results: &[HelperType::Integer],
        effects: HelperEffects::READS_HEAP
            .with(HelperEffects::WRITES_HEAP)
            .with(HelperEffects::REWRITES_ARRAYS),
        context: false,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_well_formed() {
        let mut all: Vec<(u16, &HelperInfo)> = HELPERS
            .iter()
            .enumerate()
            .map(|(position, info)| (u16::try_from(position).unwrap(), info))
            .collect();
        #[cfg(test)]
        all.extend(TEST_HELPERS.iter().enumerate().map(|(position, info)| {
            (HelperId::TEST_BASE + u16::try_from(position).unwrap(), info)
        }));
        for (id, info) in all {
            assert!(info.required <= info.params.len(), "{}", info.name);
            assert!(info.params.len() <= MAX_HELPER_ARGS, "{}", info.name);
            assert!(info.results.len() <= MAX_HELPER_RESULTS, "{}", info.name);
            // Only values can be left out of a call.
            assert!(
                info.params[info.required..]
                    .iter()
                    .all(|param| *param == HelperType::Value),
                "{}",
                info.name
            );
            // Names look up to an id of their own name, and are unique but
            // for the families that list a helper for every shape of result.
            let found = HelperId::from_name(info.name).expect(info.name);
            assert_eq!(found.info().map(|known| known.name), Some(info.name));
            if !SHAPED_NAMES.contains(&info.name) {
                assert_eq!(found, HelperId(id), "{}", info.name);
            }
            assert_eq!(HelperId(id).info().map(|known| known.name), Some(info.name));
            // A result is never a tagged value, and a pure helper reads and
            // writes no heap.
            assert!(info.results.iter().all(|r| *r != HelperType::Value));
            if info.effects.contains(HelperEffects::PURE) {
                assert!(!info.effects.contains(HelperEffects::READS_HEAP));
                assert!(!info.effects.contains(HelperEffects::WRITES_HEAP));
            }
        }
    }

    /// The names that stand for a family of helpers.
    const SHAPED_NAMES: [&str; 11] = [
        "coroutine.resume",
        "coroutine.wrap",
        "host.call",
        "table.next",
        "table.next_box",
        "string.find_pattern",
        "string.match",
        "gmatch.next",
        "utf8.codepoint",
        "string.unpack",
        // `string.sub` under two ids: `STRING_SUB`, and `STRING_SLICE`, which
        // the optimizer may drop.
        "string.sub",
    ];

    #[test]
    fn the_shaped_families_list_every_shape_in_place() {
        for captures in 0..=MAX_FIND_CAPTURES {
            for kinds in 0..(1 << captures) {
                let info = HelperId::string_find_pattern(captures, kinds)
                    .info()
                    .unwrap();
                assert_eq!(info.name, "string.find_pattern");
                assert_eq!(info.results.len(), 2 + captures);
                for (position, result) in info.results.iter().enumerate() {
                    let string = position >= 2 && kinds & (1 << (position - 2)) != 0;
                    assert_eq!(
                        *result,
                        if string {
                            HelperType::String
                        } else {
                            HelperType::Integer
                        }
                    );
                }
            }
        }
        for results in 1..=MAX_HELPER_RESULTS {
            for kinds in 0..(1 << results) {
                for (id, name) in [
                    (HelperId::string_match(results, kinds), "string.match"),
                    (HelperId::gmatch_next(results, kinds), "gmatch.next"),
                ] {
                    let info = id.info().unwrap();
                    assert_eq!(info.name, name);
                    assert_eq!(info.results.len(), results);
                    for (position, result) in info.results.iter().enumerate() {
                        let string = kinds & (1 << position) != 0;
                        assert_eq!(
                            *result,
                            if string {
                                HelperType::String
                            } else {
                                HelperType::Integer
                            }
                        );
                    }
                }
            }
        }
        assert_eq!(
            HelperId::gmatch_next(MAX_HELPER_RESULTS, (1 << MAX_HELPER_RESULTS) - 1).0 + 1,
            HelperId::GMATCH_END.0
        );
    }

    #[test]
    fn the_text_families_list_every_shape_in_place() {
        for count in 0..=MAX_UTF8_CODEPOINTS {
            let info = HelperId::utf8_codepoint(count).info().unwrap();
            assert_eq!(info.name, "utf8.codepoint");
            assert_eq!(info.results.len(), count);
            assert!(info.results.iter().all(|r| *r == HelperType::Integer));
        }
        let mut next = HelperId::STRING_UNPACK_BASE;
        for values in 0..=MAX_UNPACK_VALUES {
            for kinds in 0..pow3(values) {
                let id = HelperId::string_unpack(values, kinds);
                assert_eq!(id.0, next);
                next += 1;
                let info = id.info().unwrap();
                assert_eq!(info.name, "string.unpack");
                assert_eq!(info.results.len(), values + 1);
                assert_eq!(info.results[values], HelperType::Integer);
                let mut digits = kinds;
                for result in &info.results[..values] {
                    assert_eq!(
                        *result,
                        [HelperType::Integer, HelperType::Float, HelperType::String][digits % 3]
                    );
                    digits /= 3;
                }
            }
        }
        assert_eq!(next, HelperId::NEXT_BASE);
        assert!(!HelperId::CALL_HANDLER.is_text_outcome());
        assert!(HelperId::UTF8_LEN.is_text_outcome());
        assert!(HelperId::string_unpack(3, 26).is_text_outcome());
        assert!(!HelperId::UTF8_CHAR.is_text_outcome());
        assert!(!HelperId::STRING_PACK.is_text_outcome());
    }

    #[test]
    fn the_next_family_lists_every_shape_in_place() {
        let mut expected = HelperId::NEXT_BASE;
        for key in NextKind::ALL {
            for value in NextKind::ALL.map(Some).into_iter().chain([None]) {
                let id = HelperId::next(key, value);
                assert_eq!(id.0, expected);
                expected += 1;
                let info = id.info().unwrap();
                assert_eq!(info.name, "table.next");
                assert_eq!(info.params, &[HelperType::Table, HelperType::Value]);
                let wanted: Vec<HelperType> = [Some(key), value]
                    .into_iter()
                    .flatten()
                    .map(NextKind::helper_type)
                    .collect();
                assert_eq!(info.results, &wanted[..]);
                assert!(id.is_search_outcome());
                assert!(id.effects().contains(HelperEffects::READS_HEAP));
                assert!(!id.effects().contains(HelperEffects::WRITES_HEAP));
            }
        }
        assert_eq!(expected, HelperId::NEXT_END.0);
        assert!(HelperId::NEXT_END.is_search_outcome());
        assert!(!HelperId::PAIRS_PLAIN.is_search_outcome());
        assert_eq!(HelperId::PAIRS_PLAIN.0 + 1, HelperId::NEXT_BOX_BASE);
        assert!(!HelperId::string_unpack(3, 26).is_next_outcome());
        assert!(!HelperId::NEXT_END.is_text_outcome());
    }

    #[test]
    fn the_lane_keeping_next_family_lists_every_shape_in_place() {
        let mut expected = HelperId::NEXT_BOX_BASE;
        let kinds = NextKind::ALL.map(Some).into_iter().chain([None]);
        for key in kinds.clone() {
            let values = NextKind::ALL
                .map(BoxedValue::Kind)
                .into_iter()
                .chain([BoxedValue::Skipped, BoxedValue::Lane]);
            for value in values {
                let id = HelperId::next_box(key, value);
                assert_eq!(id.0, expected);
                expected += 1;
                let info = id.info().unwrap();
                assert_eq!(info.name, "table.next_box");
                assert_eq!(info.params, &[HelperType::Table, HelperType::String]);
                let value = match value {
                    BoxedValue::Kind(kind) => Some(kind),
                    BoxedValue::Skipped | BoxedValue::Lane => None,
                };
                let wanted: Vec<HelperType> = [key, value]
                    .into_iter()
                    .flatten()
                    .map(NextKind::helper_type)
                    .collect();
                assert_eq!(info.results, &wanted[..]);
                assert!(id.is_search_outcome());
                let effects = id.effects();
                assert!(effects.contains(HelperEffects::READS_HEAP));
                assert!(effects.contains(HelperEffects::PRIVATE_STATE));
                assert!(effects.contains(HelperEffects::MAY_EXIT));
                assert!(!effects.contains(HelperEffects::WRITES_HEAP));
            }
        }
        assert_eq!(expected, HelperId::NEXT_BOX_END.0);
        assert!(HelperId::NEXT_BOX_END.is_search_outcome());
        // Storing the control value cannot miss and is no outcome of a step.
        assert!(!HelperId::BOX_SET.is_search_outcome());
        assert!(
            !HelperId::BOX_SET
                .effects()
                .contains(HelperEffects::MAY_EXIT)
        );
        assert!(
            HelperId::BOX_SET
                .effects()
                .contains(HelperEffects::PRIVATE_STATE)
        );
        assert_eq!(HelperId::BOX_SET.0 + 1, HelperId::CALL_HANDLER.0);
        assert_eq!(HelperId::CALL_HANDLER.0 + 1, HelperId::SET_STRING.0);
        assert_eq!(HelperId::SET_STRING.0 + 1, HelperId::SET_PRESENT.0);
        assert_eq!(HelperId::SET_PRESENT.0 + 1, HelperId::DELETE_STRING.0);
        assert_eq!(HelperId::DELETE_STRING.0 + 1, HelperId::INDEX_HANDLER.0);
        assert_eq!(HelperId::INDEX_HANDLER.0 + 1, HelperId::NEWINDEX_HANDLER.0);
        assert_eq!(HelperId::NEWINDEX_HANDLER.0 + 1, HelperId::LEN_HANDLER.0);
        assert_eq!(HelperId::LEN_HANDLER.0 + 1, HelperId::TABLE_SET_STRING.0);
        assert_eq!(
            HelperId::TABLE_SET_STRING.0 + 1,
            HelperId::OPERATOR_HANDLER.0
        );
        assert_eq!(HelperId::OPERATOR_HANDLER.0 + 1, HelperId::SET_REVIVE.0);
        assert_eq!(HelperId::SET_REVIVE.0 + 1, HelperId::TABLE_NEW.0);
        assert_eq!(HelperId::TABLE_NEW.0 + 1, HelperId::TABLE_INIT.0);
        assert_eq!(HelperId::TABLE_INIT.0 + 1, HelperId::CLOSURE_SAME_PROTO.0);
        assert_eq!(
            HelperId::CLOSURE_SAME_PROTO.0 + 1,
            HelperId::CELL_GET_INTEGER.0
        );
        assert_eq!(HelperId::CELL_GET_INTEGER.0 + 1, HelperId::CELL_GET_FLOAT.0);
        assert_eq!(HelperId::CELL_GET_FLOAT.0 + 1, HelperId::CELL_GET_BOOLEAN.0);
        assert_eq!(HelperId::CELL_GET_BOOLEAN.0 + 1, HelperId::CELL_GET_TABLE.0);
        assert_eq!(HelperId::CELL_GET_TABLE.0 + 1, HelperId::CELL_SET.0);
        assert_eq!(HelperId::CELL_SET.0 + 1, HelperId::TABLE_SAME.0);
        assert_eq!(HelperId::TABLE_SAME.0 + 1, HelperId::COROUTINE_RESUME_BASE);
        assert_eq!(
            HelperId::COROUTINE_RESUME_BASE + 2 * u16::try_from(RESUME_SHAPES).unwrap(),
            HelperId::STRING_SLICE.0
        );
        assert_eq!(HelperId::TABLE_GET_INTEGER.0, HelperId::STRING_SLICE.0 + 1);
        assert_eq!(
            HelperId::TABLE_GET_USERDATA.0 + 1,
            HelperId::TABLE_ABSENT_INDEX.0
        );
        assert_eq!(
            HelperId::TABLE_ABSENT_INDEX.0 + 1,
            HelperId::CELL_GET_CALLABLE.0
        );
        assert_eq!(
            HelperId::CELL_GET_CALLABLE.0 + 1,
            HelperId::CELL_GET_STRING.0
        );
        assert_eq!(
            HelperId::CELL_GET_STRING.0 + 1,
            HelperId::CLOSURE_SAME_ENV.0
        );
        assert_eq!(HelperId::CLOSURE_SAME_ENV.0 + 1, HelperId::STRING_OWN.0);
        assert_eq!(HelperId::STRING_OWN.0 + 1, HelperId::VM_CALL_BASE);
        assert_eq!(
            HelperId::VM_CALL_BASE + u16::try_from(1 + 6 + 36 + 216).unwrap(),
            u16::try_from(HELPERS.len()).unwrap()
        );
        // The end of an unrolled `ipairs` loop is a guard with no result
        // whose miss a side continues, as a search's is.
        let absent = HelperId::TABLE_ABSENT_INDEX.info().unwrap();
        assert_eq!(absent.results, []);
        assert!(HelperId::TABLE_ABSENT_INDEX.miss_takes_side());
        assert!(HelperId::STRING_MATCH_ABSENT.miss_takes_side());
        assert!(HelperId::SET_PRESENT.miss_takes_side());
        assert!(!HelperId::TABLE_GET_INTEGER.miss_takes_side());
        // The same call as `STRING_SUB`, under another id.
        let (slice, sub) = (HelperId::STRING_SLICE, HelperId::STRING_SUB);
        assert_eq!(slice.effects(), sub.effects());
        assert_eq!(
            slice
                .info()
                .map(|info| (info.name, info.params, info.results)),
            sub.info()
                .map(|info| (info.name, info.params, info.results))
        );
        // The prototype check is pure and may be hoisted. A cell access is
        // ordered like an effect, writes no table and moves no storage.
        let same = HelperId::CLOSURE_SAME_PROTO.effects();
        assert!(!same.commits_effect() && !same.contains(HelperEffects::MAY_EXIT));
        let table_same = HelperId::TABLE_SAME.effects();
        assert!(!table_same.commits_effect() && !table_same.contains(HelperEffects::MAY_EXIT));
        for helper in [
            HelperId::CELL_GET_INTEGER,
            HelperId::CELL_GET_FLOAT,
            HelperId::CELL_GET_BOOLEAN,
            HelperId::CELL_GET_TABLE,
            HelperId::CELL_GET_CALLABLE,
            HelperId::CELL_GET_STRING,
            HelperId::CELL_SET,
        ] {
            let effects = helper.effects();
            assert!(effects.commits_effect() && effects.contains(HelperEffects::MAY_EXIT));
            assert!(!effects.moves_storage() && !effects.contains(HelperEffects::WRITES_HEAP));
        }
        // Making and filling a table commit no effect and move no storage,
        // and neither may be hoisted or merged.
        for helper in [HelperId::TABLE_NEW, HelperId::TABLE_INIT] {
            let effects = helper.effects();
            assert!(!effects.commits_effect() && !effects.moves_storage());
            assert!(effects.contains(HelperEffects::FRESH_TABLE));
        }
        // An overwrite in place moves no storage, and a delete in place only
        // empties an entry: neither moves hash storage, and only the delete
        // forbids the field bindings a nil would make stale.
        let overwrite = HelperId::SET_PRESENT.effects();
        assert!(overwrite.commits_effect());
        assert!(!overwrite.moves_storage() && !overwrite.deletes_keys());
        let delete = HelperId::DELETE_STRING.effects();
        assert!(delete.commits_effect());
        assert!(!delete.moves_storage() && delete.deletes_keys());
        assert!(!HelperId::SET_STRING.effects().deletes_keys());
        // A revive fills a deleted entry in place: it keeps hash storage like
        // a delete, and forbids the same field bindings and lane accesses.
        let revive = HelperId::SET_REVIVE.effects();
        assert!(revive.commits_effect());
        assert!(!revive.moves_storage() && revive.deletes_keys());
        // A store of a string key may add or delete the key, so it moves
        // hash storage but leaves the array parts where they are.
        let effects = HelperId::SET_STRING.effects();
        assert!(effects.commits_effect());
        assert!(effects.moves_hash_storage());
        assert!(!effects.contains(HelperEffects::REWRITES_ARRAYS));
    }

    #[test]
    fn the_production_ids_are_dense_and_have_no_test_entries() {
        for (position, info) in HELPERS.iter().enumerate() {
            assert!(!info.name.starts_with("test."), "{}", info.name);
            let found = HelperId::from_name(info.name).unwrap();
            assert_eq!(found.info().map(|known| known.name), Some(info.name));
            if !SHAPED_NAMES.contains(&info.name) {
                assert_eq!(found, HelperId(u16::try_from(position).unwrap()));
            }
        }
        assert!(
            HelperId(u16::try_from(HELPERS.len()).unwrap())
                .info()
                .is_none()
        );
        assert!(HelperId(HelperId::TEST_BASE - 1).info().is_none());
        for (id, name) in [
            (HelperId::STRING_LEN, "string.len"),
            (HelperId::STRING_LEN_REF, "string.len_ref"),
            (HelperId::STRING_CHAR, "string.char"),
            (HelperId::TOSTRING, "tostring"),
            (HelperId::STRING_FORMAT, "string.format"),
            (HelperId::STRING_FORMAT_LUAJIT, "string.format_luajit"),
            (HelperId::CONCAT, "concat"),
            (HelperId::STRING_FIND, "string.find"),
            (HelperId::STRING_FIND_ABSENT, "string.find_absent"),
            (HelperId::TABLE_CONCAT, "table.concat"),
            (HelperId::STRING_FORMAT_NATIVE, "string.format_native"),
            (HelperId::GETMETATABLE, "getmetatable"),
            (HelperId::GETMETATABLE_ABSENT, "getmetatable_absent"),
            (HelperId::SETMETATABLE, "setmetatable"),
            (HelperId::PLAIN_METATABLE, "plain_metatable"),
            (HelperId::STRING_EQUAL, "string.equal"),
            (HelperId::CONCAT_ERROR_TEXT, "concat_error_text"),
            (HelperId::TABLE_INSERT, "table.insert"),
            (HelperId::TABLE_REMOVE_INTEGER, "table.remove_integer"),
            (HelperId::TABLE_REMOVE_FLOAT, "table.remove_float"),
            (HelperId::TABLE_REMOVE_BOOLEAN, "table.remove_boolean"),
            (HelperId::TABLE_REMOVE_TABLE, "table.remove_table"),
            (HelperId::TABLE_REMOVE_NOTHING, "table.remove_nothing"),
            (HelperId::TABLE_REMOVE_STRING, "table.remove_string"),
            (HelperId::STRING_SUB, "string.sub"),
            (HelperId::STRING_REP, "string.rep"),
            (HelperId::STRING_UPPER, "string.upper"),
            (HelperId::STRING_LOWER, "string.lower"),
            (HelperId::STRING_REVERSE, "string.reverse"),
            (HelperId::MATH_RANDOM_FLOAT, "math.random_float"),
            (HelperId::MATH_RANDOM_INTEGER, "math.random_integer"),
            (HelperId::MATH_MODF_INTEGER, "math.modf_integer"),
            (HelperId::MATH_MODF_FLOAT, "math.modf_float"),
            (HelperId::TONUMBER_INTEGER, "tonumber_integer"),
            (HelperId::TONUMBER_FLOAT, "tonumber_float"),
            (HelperId::TONUMBER_NIL, "tonumber_nil"),
            (HelperId::STRING_BYTE, "string.byte"),
            (HelperId::STRING_METHOD, "string.method"),
            (HelperId::CALL_HANDLER, "call.handler"),
            (HelperId::SET_STRING, "table.set_string"),
            (HelperId::SET_PRESENT, "table.set_present"),
            (HelperId::DELETE_STRING, "table.delete_string"),
            (HelperId::INDEX_HANDLER, "index.handler"),
            (HelperId::NEWINDEX_HANDLER, "newindex.handler"),
            (HelperId::LEN_HANDLER, "len.handler"),
            (HelperId::TABLE_SET_STRING, "table.set_string_walk"),
            (HelperId::OPERATOR_HANDLER, "operator.handler"),
            (HelperId::SET_REVIVE, "table.set_revive"),
            (HelperId::TABLE_NEW, "table.new"),
            (HelperId::TABLE_INIT, "table.init"),
            (HelperId::CLOSURE_SAME_PROTO, "closure.same_proto"),
            (HelperId::CELL_GET_INTEGER, "cell.get_integer"),
            (HelperId::CELL_GET_FLOAT, "cell.get_float"),
            (HelperId::CELL_GET_BOOLEAN, "cell.get_boolean"),
            (HelperId::CELL_GET_TABLE, "cell.get_table"),
            (HelperId::TABLE_GET_INTEGER, "table.get_integer"),
            (HelperId::TABLE_GET_FLOAT, "table.get_float"),
            (HelperId::TABLE_GET_BOOLEAN, "table.get_boolean"),
            (HelperId::TABLE_GET_TABLE, "table.get_table"),
            (HelperId::TABLE_GET_CLOSURE, "table.get_closure"),
            (HelperId::TABLE_GET_STRING, "table.get_string"),
            (HelperId::TABLE_LENGTH, "table.length"),
            (HelperId::TABLE_SET_INDEX, "table.set_index"),
            (HelperId::TABLE_GET_USERDATA, "table.get_userdata"),
            (HelperId::TABLE_ABSENT_INDEX, "table.absent_index"),
            (HelperId::CELL_GET_CALLABLE, "cell.get_callable"),
            (HelperId::CELL_GET_STRING, "cell.get_string"),
            (HelperId::CLOSURE_SAME_ENV, "closure.same_env"),
            (HelperId::CELL_SET, "cell.set"),
            (HelperId::TABLE_SAME, "table.same"),
            (HelperId::STRING_MATCH_ABSENT, "string.match_absent"),
            (HelperId::GMATCH_END, "gmatch.end"),
            (HelperId::GMATCH_NEW, "string.gmatch"),
            (HelperId::STRING_GSUB, "string.gsub"),
            (HelperId::STRING_NUMBER_INTEGER, "string.number_integer"),
            (HelperId::STRING_NUMBER_FLOAT, "string.number_float"),
            (HelperId::UTF8_CHAR, "utf8.char"),
            (HelperId::UTF8_LEN, "utf8.len"),
            (HelperId::UTF8_LEN_FAIL, "utf8.len_fail"),
            (HelperId::UTF8_OFFSET, "utf8.offset"),
            (HelperId::UTF8_OFFSET_END, "utf8.offset_end"),
            (HelperId::UTF8_OFFSET_NIL, "utf8.offset_nil"),
            (HelperId::UTF8_NEXT, "utf8.next"),
            (HelperId::UTF8_NEXT_END, "utf8.next_end"),
            (HelperId::UTF8_CODES_CHECK, "utf8.codes_check"),
            (HelperId::STRING_PACK, "string.pack"),
            (HelperId::STRING_PACKSIZE, "string.packsize"),
            (HelperId::NEXT_END, "table.next_end"),
            (HelperId::PAIRS_PLAIN, "pairs.plain"),
            (HelperId::NEXT_BOX_END, "table.next_box_end"),
            (HelperId::BOX_SET, "table.box_set"),
        ] {
            assert_eq!(id.info().map(|info| info.name), Some(name));
        }
    }

    #[test]
    fn pattern_calls_are_the_search_and_rewrite_helpers_of_the_string_library() {
        for id in [
            HelperId::string_find_pattern(0, 0),
            HelperId::string_find_pattern(3, 5),
            HelperId::string_match(1, 1),
            HelperId::string_match(4, 15),
            HelperId::STRING_MATCH_ABSENT,
            HelperId::gmatch_next(2, 3),
            HelperId::GMATCH_END,
            HelperId::GMATCH_NEW,
            HelperId::STRING_GSUB,
            HelperId::string_find_prefix(2, 1),
            HelperId::string_match_prefix(3, 2),
            HelperId::gmatch_next_prefix(2, 1),
        ] {
            assert!(id.is_pattern_call(), "{id:?}");
        }
        for id in [
            HelperId::STRING_LEN,
            HelperId::TOSTRING,
            HelperId::CONCAT,
            HelperId::STRING_FIND,
            HelperId::STRING_SUB,
            HelperId::STRING_NUMBER_INTEGER,
            HelperId::UTF8_LEN,
            HelperId::TABLE_INSERT,
        ] {
            assert!(!id.is_pattern_call(), "{id:?}");
        }
    }

    #[test]
    fn a_value_parameter_takes_any_scalar_and_others_one_type() {
        for ty in [
            TraceType::Integer,
            TraceType::Float,
            TraceType::Boolean,
            TraceType::Nil,
            TraceType::String,
            TraceType::StringRef,
            TraceType::Table,
        ] {
            assert!(HelperType::Value.accepts(ty));
        }
        assert!(HelperType::Value.accepts(TraceType::Closure));
        assert!(HelperType::Closure.accepts(TraceType::Closure));
        assert!(HelperType::Integer.accepts(TraceType::Integer));
        assert!(!HelperType::Integer.accepts(TraceType::Float));
    }
}
