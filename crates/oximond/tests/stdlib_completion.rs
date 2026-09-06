use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua, StandardLibrary};

fn virtual_lua(profile: CompatibilityProfile, mode: ExecutionMode) -> Lua {
    Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .package_path("virtual/?.lua")
        .source_reader(|path| match path {
            Some(b"virtual/app/mod.lua") => Ok(b"return {name=..., answer=42}".to_vec()),
            Some(b"virtual/bad.lua") => Ok(b"local =".to_vec()),
            _ => Err(std::io::ErrorKind::NotFound.into()),
        })
        .build()
        .unwrap()
}

#[test]
fn searchpath_uses_virtual_files_and_custom_separators() {
    for profile in CompatibilityProfile::ALL {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = virtual_lua(profile, mode);
            if profile == CompatibilityProfile::Lua51 {
                assert_eq!(
                    lua.exec("return package.searchpath").unwrap(),
                    vec![Value::Nil]
                );
                continue;
            }
            assert_eq!(
                lua.exec(
                    r"
                local path = package.searchpath('app.mod',';missing/?.lua;;virtual/?.lua;')
                local custom = package.searchpath('app::mod','virtual/?.lua','::','/')
                local absent, message = package.searchpath('missing','a/?.lua;b/?.lua')
                return path, custom, absent, message
            "
                )
                .unwrap(),
                vec![
                    Value::from("virtual/app/mod.lua"),
                    Value::from("virtual/app/mod.lua"),
                    Value::Nil,
                    Value::from("\n\tno file 'a/missing.lua'\n\tno file 'b/missing.lua'")
                ],
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn source_searcher_is_callable_and_require_passes_loader_data() {
    for profile in CompatibilityProfile::ALL {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = virtual_lua(profile, mode);
            assert_eq!(
                lua.exec(
                    r"
                local searchers = package.searchers or package.loaders
                local loader, filename = searchers[2]('app.mod')
                local direct = loader('direct')
                local module = require('app.mod')
                return type(loader), filename, direct.name, module.name, module.answer,
                    type(searchers[2]('missing')), pcall(searchers[2], 'bad')
            "
                )
                .unwrap()[..7],
                [
                    Value::from("function"),
                    if profile.is_lua51_family() {
                        Value::Nil
                    } else {
                        Value::from("virtual/app/mod.lua")
                    },
                    Value::from("direct"),
                    Value::from("app.mod"),
                    if profile.is_lua53_or_later() {
                        Value::Integer(42)
                    } else {
                        Value::Number(42.0)
                    },
                    Value::from("string"),
                    Value::Boolean(false)
                ],
                "{profile:?}/{mode:?}"
            );
            if !profile.is_lua51_family() {
                let modern = matches!(
                    profile,
                    CompatibilityProfile::Lua54 | CompatibilityProfile::Lua55
                );
                let result = lua.exec(r"
                    table.insert(package.searchers,1,function(name)
                        if name == 'custom' then return function(n,data) return {n=n,data=data} end, {token=37} end
                    end)
                    local module, data = require('custom')
                    local cached, cached_data = require('custom')
                    return module.n, module.data.token, data == module.data, cached == module, cached_data
                ").unwrap();
                assert_eq!(
                    result,
                    vec![
                        Value::from("custom"),
                        if profile.is_lua53_or_later() {
                            Value::Integer(37)
                        } else {
                            Value::Number(37.0)
                        },
                        Value::Boolean(modern),
                        Value::Boolean(true),
                        Value::Nil
                    ],
                    "{profile:?}/{mode:?}"
                );
            }
        }
    }
}

#[test]
fn pure_searchpath_stays_within_the_configured_source_reader() {
    let mut lua = Lua::builder()
        .profile(CompatibilityProfile::Lua54)
        .standard_library(StandardLibrary::Pure)
        .source_reader(|_| Err(std::io::ErrorKind::NotFound.into()))
        .build()
        .unwrap();
    assert_eq!(
        lua.exec("return package.searchpath('passwd','/etc/?') == nil")
            .unwrap(),
        vec![Value::Boolean(true)]
    );
}

#[test]
fn modern_preload_metadata_is_returned_only_on_first_load() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = virtual_lua(profile, mode);
            assert_eq!(
                lua.exec(
                    r"
                package.preload.example = function(name, data)
                    return {name=name, data=data}
                end
                local module, data = require('example')
                local cached, again = require('example')
                return module.name, module.data, data, cached == module, again
            "
                )
                .unwrap(),
                vec![
                    Value::from("example"),
                    Value::from(":preload:"),
                    Value::from(":preload:"),
                    Value::Boolean(true),
                    Value::Nil
                ]
            );
        }
    }
}

#[test]
fn require_roots_searcher_metadata_while_a_loader_collects() {
    for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = virtual_lua(profile, mode);
            assert_eq!(
                lua.exec(
                    r"
                package.searchers[1] = function(name)
                    return function(name)
                        collectgarbage('collect')
                        return name
                    end, {answer=42}
                end
                local module, data = require('metadata_only')
                return module, data.answer
            "
                )
                .unwrap(),
                vec![Value::from("metadata_only"), Value::Integer(42)]
            );
        }
    }
}

#[test]
fn luajit_optional_table_modules_load_and_preserve_identity() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = virtual_lua(CompatibilityProfile::LuaJit21, mode);
        assert_eq!(
            lua.exec(
                r"
            local new = require('table.new')
            local clear = require('table.clear')
            assert(new == table.new and clear == table.clear)
            assert(require('table.new') == new and require('table.clear') == clear)
            local t = new(16,32)
            local alias = t
            local mt = {__newindex=function() error('must use raw writes') end}
            t[1],t[3],t.name = 1,3,'name'
            for i=1,40 do t['key'..i]=i end
            setmetatable(t,mt)
            local n = select('#',clear(t))
            assert(next(t) == nil and #t == 0 and alias == t and getmetatable(t) == mt)
            rawset(t,'name','reused')
            return n, t.name
        "
            )
            .unwrap(),
            vec![Value::Number(0.0), Value::from("reused")]
        );
    }
}
