//! Small, atomic probes for function calls and coroutine lifecycles.

use crate::program::Node;

impl super::Generator {
    #[allow(
        clippy::too_many_lines,
        reason = "atomic Lua probes keep setup and observations together"
    )]
    pub(super) fn function_probe(&mut self) -> Node {
        self.feature("runtime_functions");
        let tag = self.tag("F");
        let value = self.rng.range(2, 17).to_string();
        let depth = self.rng.range(2, 7).to_string();
        let text = match self.rng.index(6) {
            0 => {
                self.feature("function_recursive_results");
                self.feature("function_tail_results");
                r#"  local function tail(n, acc)
    if n == 0 then return acc, nil, "tail", nil end
    return tail(n - 1, acc + n)
  end
  local function ordinary(n)
    if n == 0 then return @VALUE@, nil, "ordinary", nil end
    local a, b, c, d = ordinary(n - 1)
    return a + n, b, c, d
  end
  local function truncated(n) return (tail(n, @VALUE@)) end
  out(@TAG@, tail(@DEPTH@, @VALUE@))
  out(@TAG@, ordinary(@DEPTH@))
  out(@TAG@, truncated(@DEPTH@))
  local a, b, c, d, e = tail(@DEPTH@, @VALUE@)
  out(@TAG@, a, b, c, d, e)"#
                    .to_owned()
            }
            1 => {
                self.feature("function_mutual_results");
                r#"  local odd
  local function even(n, acc)
    if n == 0 then return "even", acc, nil end
    return odd(n - 1, acc + n)
  end
  odd = function(n, acc)
    if n == 0 then return "odd", acc, nil end
    return even(n - 1, acc + n)
  end
  out(@TAG@, even(@DEPTH@, @VALUE@))
  out(@TAG@, odd(@DEPTH@, @VALUE@))
  local function prefix(n) return "prefix", even(n, @VALUE@) end
  out(@TAG@, prefix(@DEPTH@))"#
                    .to_owned()
            }
            2 => {
                self.feature("function_vararg_holes");
                self.feature("function_select_counts");
                let unpack = if self.dialect.table_unpack {
                    "table.unpack"
                } else {
                    "unpack"
                };
                r##"  local function forward(...) return ... end
  local function counted(...) return select("#", ...), ... end
  local function middle(...) return select(2, ...) end
  out(@TAG@, forward(@VALUE@, nil, false, nil))
  out(@TAG@, counted(@VALUE@, nil, false, nil))
  out(@TAG@, counted())
  out(@TAG@, forward())
  out(@TAG@, (forward()))
  out(@TAG@, middle(@VALUE@, nil, false, nil))
  out(@TAG@, select(-2, @VALUE@, nil, false, nil))
  local values = {@VALUE@, nil, false, nil}
  out(@TAG@, counted(@UNPACK@(values, 1, 4)))"##
                    .replace("@UNPACK@", unpack)
            }
            3 => {
                self.feature("function_shared_upvalues");
                self.feature("function_closed_upvalues");
                r"  local function siblings(start)
    local state = start
    local function add(delta) state = state + delta return state end
    local function read() return state end
    local function replace(next) local old = state state = next return old, state end
    return add, read, replace
  end
  local add, read, replace = siblings(@VALUE@)
  local other_add, other_read = siblings(@VALUE@ + 1)
  local first = add(@DEPTH@)
  out(@TAG@, first, read(), other_read())
  local old, next = replace(nil)
  out(@TAG@, old, next, read())
  local previous, current = replace(@VALUE@)
  local second = other_add(@DEPTH@)
  out(@TAG@, previous, current, read(), second, other_read())"
                    .to_owned()
            }
            4 => {
                self.feature("function_callable_table");
                self.feature("function_protected_errors");
                r##"  local function body(self, ...)
    if self.fail then error("runtime function error", 0) end
    return self.value, select("#", ...), ...
  end
  local callable = setmetatable({value = @VALUE@}, {__call = body})
  out(@TAG@, type(callable), type(body))
  out(@TAG@, callable(nil, @DEPTH@, nil))
  out(@TAG@, pcall(callable, nil, @DEPTH@, nil))
  callable.fail = true
  out(@TAG@, pcall(callable))
  local marker = {code = @VALUE@}
  local ok, err = pcall(function() error(marker, 0) end)
  out(@TAG@, ok, type(err), err == marker)
  local function handler(e) return "handled:" .. e end
  out(@TAG@, xpcall(function() error("runtime function error", 0) end, handler))"##
                    .to_owned()
            }
            _ => {
                self.feature("function_loaded_environment");
                let loader = if self.dialect.setfenv {
                    self.feature("function_loaded_setfenv");
                    r#"local loaded = assert(loadstring(source, "=runtime function"))
  setfenv(loaded, env)"#
                } else {
                    self.feature("function_loaded_env");
                    r#"local loaded = assert(load(source, "=runtime function", "t", env))"#
                };
                r#"  local env = {value = @VALUE@}
  local source = "return function(delta, ...) value = value + delta return value, ... end"
  @LOADER@
  local f = loaded()
  local a, b, c, d = f(@DEPTH@, nil, "env", nil)
  out(@TAG@, a, b, c, d, env.value)
  env.value = @VALUE@ + @DEPTH@ + 1
  local next, tail = f(1, nil)
  out(@TAG@, next, tail, env.value)
  out(@TAG@, type(loaded), type(f))"#
                    .replace("@LOADER@", loader)
            }
        };
        let text = text
            .replace("@TAG@", &tag)
            .replace("@VALUE@", &value)
            .replace("@DEPTH@", &depth);
        self.leaf(format!("do\n{text}\nend"))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "atomic Lua probes keep setup and observations together"
    )]
    pub(super) fn coroutine_probe(&mut self) -> Node {
        self.feature("runtime_coroutines");
        let tag = self.tag("C");
        let value = self.rng.range(2, 17).to_string();
        let depth = self.rng.range(2, 7).to_string();
        let text = match self.rng.index(6) {
            0 => {
                self.feature("coroutine_resume_holes");
                self.feature("coroutine_result_counts");
                r##"  local co = coroutine.create(function(...)
    out(@TAG@, "initial", select("#", ...), ...)
    local function echo(...) return select("#", ...), ... end
    local n, a, b, c, d = echo(coroutine.yield(@VALUE@, nil, "yield", nil))
    out(@TAG@, "sent", n, a, b, c, d)
    coroutine.yield()
    return nil, @DEPTH@, nil
  end)
  out(@TAG@, coroutine.resume(co, nil, @DEPTH@, nil))
  out(@TAG@, coroutine.resume(co, nil, false, @VALUE@, nil))
  out(@TAG@, coroutine.resume(co))
  out(@TAG@, coroutine.status(co))"##
                    .to_owned()
            }
            1 => {
                self.feature("coroutine_nested_resume");
                self.feature("coroutine_status_transitions");
                r#"  local outer, inner
  inner = coroutine.create(function(value)
    out(@TAG@, coroutine.status(outer), coroutine.status(inner))
    local got = coroutine.yield(value, nil)
    out(@TAG@, coroutine.status(outer), coroutine.status(inner))
    return got, nil, "inner done"
  end)
  outer = coroutine.create(function()
    out(@TAG@, coroutine.status(outer), coroutine.status(inner))
    local ok, a, b = coroutine.resume(inner, @VALUE@)
    out(@TAG@, ok, a, b, coroutine.status(inner))
    coroutine.yield("outer pause")
    local done, x, y, z = coroutine.resume(inner, @DEPTH@)
    out(@TAG@, done, x, y, z, coroutine.status(inner))
    return "outer done", nil
  end)
  out(@TAG@, coroutine.status(outer), coroutine.status(inner))
  out(@TAG@, coroutine.resume(outer))
  out(@TAG@, coroutine.status(outer), coroutine.status(inner))
  out(@TAG@, coroutine.resume(outer))
  out(@TAG@, coroutine.status(outer), coroutine.status(inner))"#
                    .to_owned()
            }
            2 => {
                self.feature("coroutine_wrap_holes");
                self.feature("coroutine_wrap_error_after_yield");
                // wrap may prepend a source location even to a level-zero
                // error. Observe the marker without printing that location.
                r##"  local wrapped = coroutine.wrap(function(...)
    out(@TAG@, "wrap initial", select("#", ...), ...)
    local a, b, c = coroutine.yield(nil, @VALUE@, nil)
    out(@TAG@, "wrap sent", a, b, c)
    error("runtime wrap error", 0)
  end)
  out(@TAG@, wrapped(nil, @DEPTH@, nil))
  local ok, err = pcall(wrapped, @VALUE@, nil, @DEPTH@)
  local marker = false
  if type(err) == "string" then
    marker = string.find(err, "runtime wrap error", 1, true) ~= nil
  end
  out(@TAG@, ok, type(err), marker)
  local again, dead_error = pcall(wrapped)
  out(@TAG@, again, type(dead_error))"##
                    .to_owned()
            }
            3 => {
                self.feature("coroutine_resume_rejections");
                r#"  local co
  co = coroutine.create(function()
    local ok, err = coroutine.resume(co)
    out(@TAG@, "running", ok, type(err), coroutine.status(co))
    coroutine.yield(@VALUE@)
    return @DEPTH@
  end)
  out(@TAG@, coroutine.resume(co))
  out(@TAG@, coroutine.resume(co))
  local ok, err = coroutine.resume(co)
  out(@TAG@, "dead", ok, type(err), coroutine.status(co))
  local failed = coroutine.create(function() error("runtime coroutine error", 0) end)
  out(@TAG@, coroutine.resume(failed))
  local retry, retry_error = coroutine.resume(failed)
  out(@TAG@, retry, type(retry_error), coroutine.status(failed))"#
                    .to_owned()
            }
            4 => {
                self.feature("coroutine_running_identity");
                let yieldable = if (self.dialect.integers || self.dialect.bit_library)
                    && self.allowed("coroutine_isyieldable")
                {
                    self.feature("coroutine_isyieldable");
                    "out(@TAG@, coroutine.isyieldable())"
                } else {
                    ""
                };
                r"  out(@TAG@, coroutine.running())
  @YIELDABLE@
  local co
  co = coroutine.create(function()
    local running, main = coroutine.running()
    out(@TAG@, type(running), running == co, main)
    @YIELDABLE@
    coroutine.yield(@VALUE@, nil)
    local resumed, is_main = coroutine.running()
    out(@TAG@, resumed == co, is_main)
    return @DEPTH@
  end)
  out(@TAG@, coroutine.resume(co))
  out(@TAG@, coroutine.resume(co))
  out(@TAG@, coroutine.running())
  @YIELDABLE@"
                    .replace("@YIELDABLE@", yieldable)
            }
            _ if self.dialect.close_and_const => {
                self.feature("coroutine_close_cleanup");
                self.feature("coroutine_close_error_cleanup");
                r#"  local log = {}
  local function resource(name)
    return setmetatable({}, {__close = function(self, err)
      log[#log + 1] = name .. ":" .. type(err)
    end})
  end
  local co = coroutine.create(function()
    local first <close> = resource("first")
    local second <close> = resource("second")
    coroutine.yield(@VALUE@, nil)
    return @DEPTH@
  end)
  out(@TAG@, coroutine.resume(co))
  out(@TAG@, #log, coroutine.status(co))
  out(@TAG@, coroutine.close(co))
  out(@TAG@, table.concat(log, ","), coroutine.status(co))
  out(@TAG@, coroutine.close(co))
  local failed = coroutine.create(function()
    local item <close> = resource("failed")
    error("runtime close error", 0)
  end)
  out(@TAG@, coroutine.resume(failed))
  out(@TAG@, #log)
  out(@TAG@, coroutine.close(failed))
  out(@TAG@, table.concat(log, ","), coroutine.status(failed))"#
                    .to_owned()
            }
            _ => {
                self.feature("coroutine_recursive_yield");
                r"  local function descend(n)
    if n == 0 then return coroutine.yield(nil, @VALUE@, nil) end
    local a, b, c = descend(n - 1)
    return a, b + n, c
  end
  local co = coroutine.create(function()
    return descend(@DEPTH@)
  end)
  out(@TAG@, coroutine.resume(co))
  out(@TAG@, coroutine.resume(co, nil, @VALUE@, nil))
  out(@TAG@, coroutine.status(co))"
                    .to_owned()
            }
        };
        let text = text
            .replace("@TAG@", &tag)
            .replace("@VALUE@", &value)
            .replace("@DEPTH@", &depth);
        self.leaf(format!("do\n{text}\nend"))
    }
}
