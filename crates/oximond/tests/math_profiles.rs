use oximond::runtime::Value;
use oximond::{CompatibilityProfile, ExecutionMode, Lua};

fn lua(profile: CompatibilityProfile, mode: ExecutionMode) -> Lua {
    Lua::builder()
        .profile(profile)
        .execution_mode(mode)
        .build()
        .unwrap()
}

#[test]
fn atan_uses_two_coordinates_from_lua53_onward() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        for profile in [
            CompatibilityProfile::Lua53,
            CompatibilityProfile::Lua54,
            CompatibilityProfile::Lua55,
        ] {
            let mut lua = lua(profile, mode);
            assert_eq!(
                lua.exec("return math.atan(1, 0), math.atan(1), math.atan(1, nil)")
                    .unwrap(),
                vec![
                    Value::Number(std::f64::consts::FRAC_PI_2),
                    Value::Number(std::f64::consts::FRAC_PI_4),
                    Value::Number(std::f64::consts::FRAC_PI_4),
                ],
                "{profile:?}/{mode:?}"
            );
        }
        for profile in [
            CompatibilityProfile::Lua51,
            CompatibilityProfile::Lua52,
            CompatibilityProfile::LuaJit21,
        ] {
            let mut lua = lua(profile, mode);
            assert_eq!(
                lua.exec("return math.atan(1, 0)").unwrap(),
                vec![Value::Number(std::f64::consts::FRAC_PI_4)],
                "{profile:?}/{mode:?}"
            );
        }
    }
}

#[test]
fn lua54_and_lua55_randomseed_return_seeds_and_reproduce_lua54_sequence() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        for profile in [CompatibilityProfile::Lua54, CompatibilityProfile::Lua55] {
            let mut lua = lua(profile, mode);
            assert_eq!(
                lua.exec("local a,b=math.randomseed(1234,5678); return a,b,math.random(0),math.random(),math.random(10),math.random(-10,10)")
                    .unwrap(),
                vec![
                    Value::Integer(1234),
                    Value::Integer(5678),
                    Value::Integer(5_173_903_662_814_325_993),
                    Value::Number(0.991_611_477_994_134_5),
                    Value::Integer(1),
                    Value::Integer(2),
                ],
                "{profile:?}/{mode:?}"
            );
            assert_eq!(
                lua.exec("return math.randomseed(-1)").unwrap(),
                vec![Value::Integer(-1), Value::Integer(0)]
            );
            assert!(matches!(
                lua.exec("return math.randomseed()"),
                Ok(values) if values.len() == 2 && values.iter().all(|value| matches!(value, Value::Integer(_)))
            ));
        }
    }
}

#[test]
fn luajit_random_uses_seeded_tausworthe_sequence() {
    for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
        let mut lua = lua(CompatibilityProfile::LuaJit21, mode);
        assert_eq!(
            lua.exec("math.randomseed(1); return math.random(),math.random(10),math.random(0),math.random(-3,4)")
                .unwrap(),
            vec![
                Value::Number(0.323_810_562_378_636_7),
                Value::Number(1.0),
                Value::Number(1.0),
                Value::Number(-1.0),
            ],
            "{mode:?}"
        );
        assert_eq!(
            lua.exec("return math.random(0,0), math.random(1,1.5)")
                .unwrap(),
            vec![Value::Number(0.0), Value::Number(1.0)],
            "{mode:?}"
        );
    }
}
