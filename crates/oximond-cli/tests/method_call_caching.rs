//! Hot method calls whose receiver fields are updated by compiled leaves
//! while the same loop reads fields and globals through native caches.

use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn run(source: &str) {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua51)
            .execution_mode(mode)
            .build()
            .unwrap();
        if let Err(error) = lua.exec(source) {
            panic!("{mode}: {error:?}");
        }
    }
}

#[test]
fn field_reads_observe_leaf_updates_and_method_replacement() {
    run(r"
        local object = {value = 0}
        function object:add(d) self.value = self.value + d return self.value end
        local s = 0
        for i = 1, 200000 do
            s = s + object:add(i % 7) + object.value
            if i == 150000 then
                object.add = function(self, d) self.value = self.value - d return 1 end
            end
        end
        assert(s == 86249974987, s)
        assert(object.value == 299999, object.value)
    ");
}

#[test]
fn global_reads_observe_leaf_updates_through_the_environment() {
    run(r"
        value = 0
        function _G:add(d) self.value = self.value + d return self.value end
        local t = 0
        for i = 1, 200000 do t = t + _G:add(1) + value end
        assert(t == 40000200000, t)
        assert(value == 200000, value)
    ");
}

#[test]
fn inherited_methods_and_field_type_changes_stay_exact() {
    run(r#"
        local Class = {} Class.__index = Class
        function Class:add(d) self.n = self.n + d return self.n end
        local o = setmetatable({n = 1}, Class)
        local u = 0
        for i = 1, 200000 do
            u = u + o:add(2)
            if i == 100000 then function Class:add(d) self.n = self.n * 1 return -1 end end
        end
        assert(u == 10000100000, u)
        assert(o.n == 200001, o.n)

        local add = function(self, d) self.value = self.value + d return self.value end
        local p = {value = 1, add = add}
        local w = 0
        for i = 1, 200000 do
            w = w + p:add(3) + p.value
            if i == 120000 then p.value = "7" end
        end
        assert(w == 62401960000, w)
        assert(p.value == 240007, p.value)
    "#);
}
