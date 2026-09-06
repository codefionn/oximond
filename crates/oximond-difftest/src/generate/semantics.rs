//! Atomic probes for dynamic values, metatables, and control flow.

use crate::program::Node;

const TYPES: [&str; 5] = [
    r##"  local marker = {code = @VALUE@}
  local function f(...) return ... end
  local thread = coroutine.create(function() return false end)
  local function mixed(...) return select("#", ...), ... end
  out(@TAG@, mixed(nil, false, true, @VALUE@, @FLOAT@, "typed", marker, f, thread, userdata, nil))
  local values = {false, true, @VALUE@, @FLOAT@, "typed", marker, f, thread, userdata}
  for i = 1, 9 do out(@TAG@, i, type(values[i]), values[i] == values[i]) end
  out(@TAG@, values[6] == marker, values[7] == f, values[8] == thread, values[9] == userdata)"##,
    r#"  local marker, f = {}, function() return "function" end
  local thread = coroutine.create(f)
  local function mixed() return false, nil, marker, f, thread, userdata, @FLOAT@, nil end
  local a, b, c, d, e, g, h, j, k = mixed()
  out(@TAG@, a, b, c == marker, d == f, e == thread, g == userdata, h, j, k)
  local values = {"prefix", mixed()}
  for i = 1, 9 do out(@TAG@, i, type(values[i])) end
  a, b, c = "changed", false, nil
  out(@TAG@, a, b, c)
  a, b, c = b, c, a
  out(@TAG@, a, b, c)
  out(@TAG@, (mixed()), select(3, mixed()))"#,
    r#"  local marker, f = {}, function() end
  local thread = coroutine.create(f)
  local keys = {false, true, @VALUE@, "key", marker, f, thread, userdata}
  local map = {}
  for i = 1, 8 do map[keys[i]] = i end
  for i = 1, 8 do out(@TAG@, type(keys[i]), map[keys[i]], rawget(map, keys[i])) end
  out(@TAG@, map[@VALUE@ + 0.0], map[{}], map["absent"])
  map[marker], map[false] = nil, "changed"
  out(@TAG@, map[marker], map[false], map[true], map[userdata])
  local ok, err = pcall(function() map[nil] = @VALUE@ end)
  out(@TAG@, ok, type(err))"#,
    r#"  local f = function() end
  local values = {false, true, 0, @VALUE@, @FLOAT@, "", "typed", {}, f, coroutine.create(f), userdata}
  for i = 1, 11 do
    local value = values[i]
    local branch
    if value then branch = "truthy" else branch = "falsey" end
    out(@TAG@, i, type(value), not value, branch, value and "yes" or "no")
  end
  local missing
  out(@TAG@, type(missing), not missing, missing or false, false or @VALUE@, 0 and "zero")"#,
    r#"  local texts = {"", "  @VALUE@ ", "@FLOAT@", "false", "nil", "typed"}
  for i = 1, 6 do
    local number = tonumber(texts[i])
    out(@TAG@, texts[i], number, type(number), texts[i] == number)
  end
  out(@TAG@, tostring(nil), tostring(false), tostring(true), type(userdata), userdata == userdata)
  local marker = {code = @VALUE@}
  local function classify(value) return type(value), value == marker, value == false, value == nil end
  out(@TAG@, classify(nil))
  out(@TAG@, classify(false))
  out(@TAG@, classify(marker))
  out(@TAG@, classify(userdata))"#,
];

const METATABLES: [&str; 6] = [
    r#"  local fallback = {present = false, number = @VALUE@}
  local writes = {}
  local proxy = setmetatable({own = "own"}, {__index = fallback, __newindex = writes})
  out(@TAG@, proxy.own, proxy.present, proxy.number, proxy.absent)
  proxy.changed, proxy.present = @FLOAT@, true
  out(@TAG@, proxy.changed, writes.changed, proxy.present, writes.present, rawget(proxy, "changed"))
  rawset(proxy, "changed", false)
  proxy.changed = "direct"
  out(@TAG@, proxy.changed, writes.changed)
  local count = 0
  setmetatable(proxy, {__index = function(self, key)
    count = count + 1
    return key, nil, "discarded"
  end, __newindex = function(self, key, value)
    count = count + 1
    rawset(self, key, value)
  end})
  local missing = proxy.missing
  proxy.fresh = false
  out(@TAG@, missing, proxy.fresh, count, rawget(proxy, "missing"))"#,
    r##"  local marker = {}
  local callable = setmetatable({value = @VALUE@}, {__call = function(self, ...)
    return self.value, select("#", ...), ...
  end})
  local thread = coroutine.create(function() end)
  out(@TAG@, callable(nil, false, "argument", marker, thread, userdata))
  out(@TAG@, callable())
  local a, count, first, last = callable(false, marker)
  out(@TAG@, a, count, first, last)
  local mt = getmetatable(callable)
  mt.__call = function(self, value) return nil, value, self == callable, nil end
  out(@TAG@, callable(marker))
  out(@TAG@, pcall(callable, false))"##,
    r#"  local function value(v) if type(v) == "table" then return v.n end return v end
  local mt = {
    __add = function(a, b) return value(a) + value(b), "discarded" end,
    __sub = function(a, b) return value(a) - value(b) end,
    __mul = function(a, b) return value(a) * value(b) end,
    __div = function(a, b) return value(a) / value(b) end,
    __mod = function(a, b) return value(a) % value(b) end,
    __pow = function(a, b) return value(a) ^ value(b) end,
    __unm = function(a) return -a.n end,
    __concat = function(a, b) return type(a) .. ":" .. type(b) end
  }
  local a, b = setmetatable({n = @VALUE@}, mt), setmetatable({n = 2}, mt)
  out(@TAG@, a + b, a + 2, 2 + a, a - b, 2 - a, a * b)
  out(@TAG@, a / b, 2 / a, a % b, a ^ b, -a)
  out(@TAG@, a .. "suffix", "prefix" .. a, a .. b)"#,
    r#"  local calls = 0
  local mt = {__eq = function(a, b) calls = calls + 1 return a.n == b.n end,
    __lt = function(a, b) calls = calls + 1 return a.n < b.n end}
  local a, b, c = setmetatable({n = @VALUE@}, mt), setmetatable({n = @VALUE@}, mt), setmetatable({n = @VALUE@ + 1}, mt)
  local equal = a == b
  local different = a == c
  local identical = a == a
  out(@TAG@, equal, different, identical, rawequal(a, b), rawequal(a, a))
  local less = a < c
  local greater = c > a
  local less_ok, less_equal = pcall(function() return a <= b end)
  local greater_ok, greater_equal = pcall(function() return c >= a end)
  if not less_ok then less_equal = type(less_equal) end
  if not greater_ok then greater_equal = type(greater_equal) end
  out(@TAG@, less, greater, less_ok, less_equal, greater_ok, greater_equal, calls)
  mt.__le = function(left, right) return left.n <= right.n end
  out(@TAG@, a <= b, c <= a, a >= b)
  mt.__metatable = "protected"
  out(@TAG@, getmetatable(a), getmetatable(b))
  local ok, err = pcall(setmetatable, a, {})
  out(@TAG@, ok, type(err), rawget(a, "n"))
  @EXTRA@"#,
    r#"  local key = "__oximond_difftest_semantics_global"
  local saved = rawget(_G, key)
  _G[key] = false
  out(@TAG@, _G[key], rawget(_G, key))
  rawset(_G, key, @VALUE@)
  out(@TAG@, _G[key], rawget(_G, key))
  rawset(_G, key, saved)
  out(@TAG@, rawequal(rawget(_G, key), saved))
  local env = {value = false}
  @ENVIRONMENT@
  out(@TAG@, first, second, env.value)"#,
    r#"  local marker = {code = @VALUE@}
  local errors = {"semantic error", @VALUE@, false, marker}
  for i = 1, 4 do
    local payload = errors[i]
    local ok, err = pcall(function() error(payload, 0) end)
    out(@TAG@, ok, type(err), err == payload, err == marker)
    local handled, result = xpcall(function() error(payload, 0) end,
      function(e) return {kind = type(e), same = e == payload, marker = e == marker} end)
    out(@TAG@, handled, type(result), result.kind, result.same, result.marker)
  end
  local ok, err = pcall(function() error(nil, 0) end)
  out(@TAG@, ok, type(err), err == nil)
  local handled, result = xpcall(function() error(nil, 0) end, function(e) return type(e) end)
  out(@TAG@, handled, result)
  local attempts = 0
  local recovered, recovery = xpcall(function() error("first", 0) end, function(e)
    attempts = attempts + 1
    if attempts == 1 then error(marker, 0) end
    return e == marker and "recovered" or "other"
  end)
  out(@TAG@, recovered, type(recovery), recovery == "recovered", attempts)"#,
];

const CONTROLS: [&str; 5] = [
    r"  local total, i = 0, 0
  while i < 5 do
    i = i + 1
    for j = 1, 5 do
      if j == 3 then break end
      total = total + i * j
    end
    if i == 3 then break end
  end
  local visits = 0
  repeat
    visits = visits + 1
    local k = 0
    repeat
      k = k + 1
      if k == 2 then break end
      total = total + @VALUE@
    until k == 4
    if visits == 2 then break end
  until visits == 5
  local reverse = 0
  for j = 5, -1, -2 do reverse = reverse + j if j == 1 then break end end
  out(@TAG@, total, i, visits, reverse)",
    r#"  local state = {index = 0}
  local function iterator(s, previous)
    s.index = s.index + 1
    if s.index == 1 then return false, "false control", nil end
    if s.index == 2 then return @VALUE@, nil, false end
    if s.index == 3 then return "last", true, @FLOAT@ end
    return nil, "discarded"
  end
  local count = 0
  for key, value, extra in iterator, state, nil do
    count = count + 1
    out(@TAG@, key, value, extra)
  end
  out(@TAG@, count, state.index)
  state.index = 0
  for key, value in iterator, state, nil do
    out(@TAG@, key, value)
    if key == @VALUE@ then break end
  end
  out(@TAG@, state.index)"#,
    r#"  local values = {false, true, 0, "", "typed", {}, function() end, userdata}
  local count = 0
  local function touched(value) count = count + 1 return value end
  for i = 1, 9 do
    local value = values[i]
    local branch
    if value == nil then branch = "nil"
    elseif value == false then branch = "false"
    elseif type(value) == "number" then branch = "number"
    else branch = "other" end
    local a = value and touched(@VALUE@)
    local b = value or touched(@FLOAT@)
    local c = value and false or "fallback"
    out(@TAG@, type(value), branch, a, type(b), c, count)
  end"#,
    r#"  local co = coroutine.create(function()
    local total = 0
    for i = 1, 4 do
      local j = 0
      while j < 4 do
        j = j + 1
        if j == 3 then break end
        local value
        if i % 2 == 0 then value = false elseif j == 2 then value = "branch" else value = @VALUE@ end
        local sent = coroutine.yield(i, j, value, nil)
        total = total + (sent or 0)
      end
      if i == 3 then break end
    end
    repeat
      total = total + 1
      coroutine.yield(nil, false, total)
      break
    until false
    return total, nil, "done"
  end)
  for step = 1, 9 do
    local ok, a, b, c, d = coroutine.resume(co, @VALUE@)
    if ok then out(@TAG@, step, ok, a, b, c, d)
    else out(@TAG@, step, ok, type(a)) end
    if not ok or coroutine.status(co) == "dead" then break end
  end
  out(@TAG@, coroutine.status(co))"#,
    r"  local closures = {}
  for i = 1, 4 do
    local captured = i * @VALUE@
    local j = 0
    repeat
      j = j + 1
      local inner = j
      closures[#closures + 1] = function() return captured, inner end
      if j == 2 then break end
    until j == 4
    if i == 2 then break end
  end
  for i = 1, #closures do out(@TAG@, closures[i]()) end
  @GOTO@",
];

impl super::Generator {
    pub(super) fn type_probe(&mut self) -> Node {
        self.feature("runtime_types");
        let variant = self.rng.index(TYPES.len());
        self.feature(
            [
                "types_mixed_arguments",
                "types_mixed_results",
                "types_identity_keys",
                "types_truthiness",
                "types_scalar_conversion",
            ][variant],
        );
        self.semantic_leaf(TYPES[variant], "Y")
    }

    pub(super) fn metatable_probe(&mut self) -> Node {
        self.feature("runtime_metatables");
        let variant = self.rng.index(METATABLES.len());
        self.feature(
            [
                "metatable_index_newindex",
                "metatable_call_results",
                "metatable_arithmetic_concat",
                "metatable_order_equality_raw",
                "metatable_globals_environment",
                "metatable_protected_errors",
            ][variant],
        );
        let mut text = METATABLES[variant].to_owned();
        if variant == 3 {
            let mut extra = String::new();
            if self.dialect.mixed_eq_metamethod {
                self.feature("metatable_mixed_equality");
                extra.push_str("local other = setmetatable({n = @VALUE@}, {__eq = function() return false end})\n  local left = a == other\n  local right = other == a\n  out(@TAG@, left, right)\n  ");
            }
            if self.dialect.len_metamethod {
                self.feature("metatable_length");
                extra.push_str("mt.__len = function(self) return self.n end\n  out(@TAG@, #a, #b)");
            }
            text = text.replace("@EXTRA@", &extra);
        }
        if variant == 4 {
            let environment = if self.dialect.setfenv {
                self.feature("metatable_setfenv");
                "local function mutate() local old = value value = @FLOAT@ return old, value end\n  setfenv(mutate, env)\n  local first, second = mutate()"
            } else {
                self.feature("metatable_local_env");
                "local first, second\n  do\n    local _ENV = env\n    first = value\n    value = @FLOAT@\n    second = value\n  end"
            };
            text = text.replace("@ENVIRONMENT@", environment);
        }
        self.semantic_leaf(&text, "M")
    }

    pub(super) fn control_probe(&mut self) -> Node {
        self.feature("runtime_control");
        let variant = self.rng.index(CONTROLS.len());
        self.feature(
            [
                "control_nested_break",
                "control_generic_false",
                "control_branch_short_circuit",
                "control_nested_yield",
                "control_break_closures",
            ][variant],
        );
        let goto = if variant == 4 && self.dialect.goto {
            self.feature("control_goto_scopes");
            "local jumps = {}\n  for i = 1, 4 do\n    do\n      local captured = i\n      jumps[#jumps + 1] = function() return captured end\n      if i % 2 == 0 then goto next_iteration end\n      out(@TAG@, \"before jump\", i)\n    end\n    ::next_iteration::\n  end\n  for i = 1, #jumps do out(@TAG@, jumps[i]()) end"
        } else {
            ""
        };
        self.semantic_leaf(&CONTROLS[variant].replace("@GOTO@", goto), "B")
    }

    fn semantic_leaf(&mut self, template: &str, prefix: &str) -> Node {
        let userdata = if template.contains("userdata") {
            if self.dialect.setfenv {
                self.feature("types_full_userdata");
                "  local userdata = newproxy(true)\n"
            } else {
                self.feature("types_light_userdata");
                "  local captured = {}\n  local function opaque() return captured end\n  local userdata = debug.upvalueid(opaque, 1)\n"
            }
        } else {
            ""
        };
        let tag = self.tag(prefix);
        let value = self.rng.range(2, 17).to_string();
        let float = format!("{}.5", self.rng.range(-9, 9));
        let text = template
            .replace("@TAG@", &tag)
            .replace("@VALUE@", &value)
            .replace("@FLOAT@", &float);
        self.leaf(format!("do\n{userdata}{text}\nend"))
    }
}
