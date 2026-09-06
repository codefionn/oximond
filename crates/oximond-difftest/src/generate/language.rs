//! Language constructs and collection behavior that ordinary expressions miss.

use crate::program::Node;
use std::fmt::Write as _;

impl super::Generator {
    pub(super) fn language_probe(&mut self) -> Node {
        let tag = self.tag("Y");
        let mut source = format!("do\n  local report = out\n  local tag = {tag}\n");

        if self.dialect.lua55 {
            let global = self.fresh("declared");
            self.feature("lua55_global_declarations");
            let _ = write!(
                source,
                "  do\n    local {global} = 9\n    do global {global} = 42; report(tag, {global}) end\n    report(tag, {global}, _G.{global})\n  end\n"
            );
            let constant = self.fresh("readonly");
            let _ = writeln!(
                source,
                "  do global<const> {constant} = 17; report(tag, {constant}) end"
            );
            // `load` tests the compiler path without making the generated
            // program invalid. Error wording differs across implementations.
            let _ = writeln!(
                source,
                "  report(tag, load('global {constant}<const>; {constant} = 1; {constant} = 2') == nil)"
            );
            self.feature("lua55_named_varargs");
            source.push_str(
                "  local function named(head, ... extras)\n    return extras.n, extras[1], extras[2], head, ...\n  end\n  report(tag, named(9, 3, nil, 4))\n  report(tag, named(5))\n",
            );
        }

        if self.dialect.close_and_const {
            self.feature("const_capture");
            source.push_str(
                "  local function capture()\n    local value <const> = 23\n    return function() return value end\n  end\n  report(tag, capture()())\n",
            );
            self.feature("close_control_flow");
            source.push_str(
                "  local closed = {}\n  local function guard(label)\n    return setmetatable({}, {__close = function(_, err) closed[#closed + 1] = label .. ':' .. type(err) end})\n  end\n  local function on_return()\n    local resource <close> = guard('return')\n    return 7\n  end\n  local returned = on_return()\n  report(tag, returned, table.concat(closed, ','))\n  for i = 1, 2 do\n    local resource <close> = guard('break')\n    break\n  end\n  report(tag, table.concat(closed, ','))\n  do\n    local resource <close> = guard('goto')\n    goto after_close\n  end\n  ::after_close::\n  report(tag, table.concat(closed, ','))\n  local ok, err = pcall(function()\n    local resource <close> = guard('error')\n    error('close probe')\n  end)\n  report(tag, ok, type(err), table.concat(closed, ','))\n",
            );
        }

        if self.dialect.goto {
            self.feature("goto_backward");
            source.push_str(
                "  local jumps = 0\n  ::again::\n  jumps = jumps + 1\n  if jumps < 4 then goto again end\n  report(tag, jumps)\n",
            );
        }

        self.feature("weak_tables_reclaimed");
        self.feature("gc");
        // The short-lived keys and values leave their creating function's
        // stack before collection. That avoids depending on dead stack slots
        // being cleared while the current function is still running.
        source.push_str(
            "  local function weak_entries(mode)\n    local weak = setmetatable({}, {__mode = mode})\n    if mode == 'k' then\n      weak[{}] = 11\n    else\n      weak.anchor = {}\n    end\n    return weak\n  end\n  local keys = weak_entries('k')\n  local values = weak_entries('v')\n  collectgarbage('collect')\n  collectgarbage('collect')\n  local key_count, value_count = 0, 0\n  for _ in pairs(keys) do key_count = key_count + 1 end\n  for _ in pairs(values) do value_count = value_count + 1 end\n  report(tag, key_count, value_count)\n",
        );
        source.push_str("end");
        self.leaf(source)
    }
}
