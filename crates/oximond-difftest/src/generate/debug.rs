//! Atomic debug-library probes. Hook installation and cleanup stay in the
//! same leaf so reduction cannot leave a hook attached to later statements.

use crate::program::Node;

const SOURCE: &str = "@difftest-debug";

impl super::Generator {
    pub(super) fn debug_probe(&mut self) -> Node {
        self.feature("debug_probes");
        let tag = self.tag("D");
        let choice = self.rng.index(8);
        let text = match choice {
            0 => self.debug_generic_lines(&tag),
            1 => self.debug_loop_lines(&tag),
            2 => self.debug_calls(&tag),
            3 => self.debug_info(&tag),
            4 => self.debug_locals(&tag),
            5 => self.debug_upvalues(&tag),
            6 => self.debug_thread(&tag),
            _ => self.debug_count(&tag),
        };
        self.leaf(format!("do\n{text}\nend"))
    }

    fn debug_loader(&self) -> &'static str {
        if self.dialect.env_local {
            "load"
        } else {
            "loadstring"
        }
    }

    fn debug_generic_lines(&mut self, tag: &str) -> String {
        self.feature("debug_generic_for_hooks");
        let iterator = if self.rng.chance(1, 2) {
            "pairs"
        } else {
            "ipairs"
        };
        let count = self.rng.range(0, 3);
        let values = (1..=count)
            .map(|value| (value * self.rng.range(1, 4)).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "local values = {{{values}}}\nlocal total = 0\nfor key, value in {iterator}(values) do\n  total = total + key + value\nend\nreturn total"
        );
        // The header, body, and end are separate source lines. In particular,
        // a zero-trip loop can reveal an extra body line event in Lua 5.1.
        self.debug_line_chunk(tag, &source, "3, 4, 5")
    }

    fn debug_loop_lines(&mut self, tag: &str) -> String {
        let count = self.rng.range(0, 3);
        let (source, lines) = match self.rng.index(3) {
            0 => {
                self.feature("debug_numeric_for_hooks");
                if self.rng.chance(1, 2) {
                    self.feature("debug_break_hooks");
                    let limit = count + 1;
                    (
                        format!(
                            "local total = 0\nfor index = 1, {limit} do\n  if index > {count} then break end\n  total = total + index\nend\nreturn total"
                        ),
                        "2, 3, 5",
                    )
                } else {
                    (
                        format!(
                            "local total = 0\nfor index = 1, {count} do\n  total = total + index\nend\nreturn total"
                        ),
                        "2, 3, 4",
                    )
                }
            }
            1 => {
                self.feature("debug_while_hooks");
                (
                    format!(
                        "local index, total = 0, 0\nwhile index < {count} do\n  index = index + 1\n  total = total + index\nend\nreturn total"
                    ),
                    "2, 3, 5",
                )
            }
            _ => {
                self.feature("debug_repeat_hooks");
                (
                    format!(
                        "local index, total = 0, 0\nrepeat\n  index = index + 1\n  total = total + index\nuntil index >= {count}\nreturn total"
                    ),
                    "2, 3, 5",
                )
            }
        };
        self.debug_line_chunk(tag, &source, lines)
    }

    fn debug_line_chunk(&mut self, tag: &str, source: &str, lines: &str) -> String {
        self.feature("debug_line_hooks");
        let loader = self.debug_loader();
        format!(
            r#"local target = assert({loader}([=[{source}]=], "{SOURCE}"))
local trace = {{}}
local function hook(event, line)
  local info = debug.getinfo(2, "S")
  if info and info.source == "{SOURCE}" then
    trace[#trace + 1] = line
  end
end
debug.sethook(hook, "l")
local ok, result = pcall(target)
debug.sethook()
out({tag}, "lines", {lines}, ok, ok and result or type(result), table.concat(trace, ","), debug.gethook() == nil)"#
        )
    }

    fn debug_calls(&mut self, tag: &str) -> String {
        self.feature("debug_call_hooks");
        let tail = self.rng.chance(1, 2);
        let body = if tail {
            self.feature("debug_tail_hooks");
            "  return child(value)"
        } else {
            "  local result = child(value)\n  return result + 1"
        };
        let value = self.rng.range(-9, 9);
        let loader = self.debug_loader();
        format!(
            r#"local target = assert({loader}([=[local function child(value)
  return value + 1
end
local function parent(value)
{body}
end
local result = parent({value})
return result]=], "{SOURCE}"))
local trace = {{}}
local function hook(event)
  local info = debug.getinfo(2, "S")
  if info and info.source == "{SOURCE}" then
    trace[#trace + 1] = event
  end
end
debug.sethook(hook, "cr")
local ok, result = pcall(target)
debug.sethook()
out({tag}, "calls", {tail}, ok, ok and result or type(result), table.concat(trace, ","), debug.gethook() == nil)"#
        )
    }

    fn debug_info(&mut self, tag: &str) -> String {
        self.feature("debug_getinfo");
        let value = self.rng.range(-9, 9);
        let loader = self.debug_loader();
        format!(
            r##"local maker = assert({loader}([=[local function subject(value, ...)
  local identity = debug.getinfo(1, "f").func == subject
  return identity, value, select("#", ...)
end
return subject]=], "{SOURCE}"))
local subject = maker()
local info = debug.getinfo(subject, "Suf")
local identity, value, count = subject({value}, false, "extra")
out({tag}, "info", info.what, info.source == "{SOURCE}", info.linedefined, info.lastlinedefined, info.func == subject, info.nups, info.nparams, info.isvararg, identity, value, count)"##
        )
    }

    fn debug_locals(&mut self, tag: &str) -> String {
        self.feature("debug_named_locals");
        let before = self.rng.range(-9, 9);
        let after = self.rng.range(10, 30);
        format!(
            r#"local function subject()
  local probe_local = {before}
  local found, observed, changed = false, nil, false
  for index = 1, 32 do
    local name, value = debug.getlocal(1, index)
    if name == "probe_local" then
      found, observed = true, value
      changed = debug.setlocal(1, index, {after}) == name
      break
    end
  end
  return found, observed, changed, probe_local
end
local found, before, changed, after = subject()
out({tag}, "locals", found, before, changed, after)"#
        )
    }

    fn debug_upvalues(&mut self, tag: &str) -> String {
        self.feature("debug_named_upvalues");
        let before = self.rng.range(-9, 9);
        let after = self.rng.range(10, 30);
        let neighbor = self.rng.range(1, 4);
        format!(
            r#"local probe_upvalue, neighbor = {before}, {neighbor}
local function subject() return probe_upvalue + neighbor end
local found, observed, changed = false, nil, false
for index = 1, 16 do
  local name, value = debug.getupvalue(subject, index)
  if name == "probe_upvalue" then
    found, observed = true, value
    changed = debug.setupvalue(subject, index, {after}) == name
    break
  end
end
out({tag}, "upvalues", found, observed, changed, probe_upvalue, subject())"#
        )
    }

    fn debug_thread(&mut self, tag: &str) -> String {
        self.feature("debug_thread_locals");
        self.feature("debug_thread_hooks");
        let before = self.rng.range(-9, 9);
        let after = self.rng.range(10, 30);
        let increment = self.rng.range(1, 4);
        let loader = self.debug_loader();
        format!(
            r#"local maker = assert({loader}([=[return function()
  local probe_thread = {before}
  coroutine.yield(probe_thread)
  probe_thread = probe_thread + {increment}
  return probe_thread
end]=], "{SOURCE}"))
local subject = maker()
local thread = coroutine.create(subject)
local trace = {{}}
local function hook(event, line)
  local info = debug.getinfo(2, "S")
  if info and info.source == "{SOURCE}" then trace[#trace + 1] = line end
end
debug.sethook(thread, hook, "l")
local installed, mask, count = debug.gethook(thread)
local first_ok, yielded = coroutine.resume(thread)
local found, observed, changed = false, nil, false
if first_ok and coroutine.status(thread) == "suspended" then
  for level = 0, 3 do
    local info = debug.getinfo(thread, level, "f")
    if info and info.func == subject then
      for index = 1, 32 do
        local name, value = debug.getlocal(thread, level, index)
        if name == "probe_thread" then
          found, observed = true, value
          changed = debug.setlocal(thread, level, index, {after}) == name
          break
        end
      end
    end
  end
end
local second_ok, result = coroutine.resume(thread)
debug.sethook(thread)
out({tag}, "thread", installed == hook, mask, count, first_ok, first_ok and yielded or type(yielded), found, observed, changed, second_ok, second_ok and result or type(result), coroutine.status(thread), table.concat(trace, ","), debug.gethook(thread) == nil, debug.gethook() == nil)"#
        )
    }

    fn debug_count(&mut self, tag: &str) -> String {
        self.feature("debug_count_hooks");
        let interval = self.rng.range(1, 5);
        let count = self.rng.range(8, 16);
        let loader = self.debug_loader();
        format!(
            r#"local target = assert({loader}([=[local total = 0
for index = 1, {count} do
  total = total + index
end
return total]=], "{SOURCE}"))
local fired = false
local function hook(event)
  local info = debug.getinfo(2, "S")
  if event == "count" and info and info.source == "{SOURCE}" then fired = true end
end
debug.sethook(hook, "", {interval})
local installed, mask, count = debug.gethook()
local ok, result = pcall(target)
debug.sethook()
out({tag}, "count", installed == hook, mask, count, fired, ok, ok and result or type(result), debug.gethook() == nil)"#
        )
    }
}
