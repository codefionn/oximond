//! A protected function writes a captured local of the frame that called
//! `pcall` or `xpcall`, then fails or returns. The caller must see the write
//! afterwards, whatever kind of function handles the error.

use super::*;

const PROFILES: [CompatibilityProfile; 3] = [
    CompatibilityProfile::Lua51,
    CompatibilityProfile::Lua54,
    CompatibilityProfile::LuaJit21,
];

fn upvalue_s() -> Vec<UpvalueDescriptor> {
    vec![UpvalueDescriptor {
        name: "s".into(),
        source: UpvalueSource::ParentRegister(0),
    }]
}

/// `function() s = 2; <fails when `fail`> end`, with `s` captured from the
/// caller's register 0.
fn writer(fail: bool) -> Chunk {
    let mut code = vec![
        Instruction::load_constant(0, 0),
        Instruction::set_upvalue(0, 0),
    ];
    if fail {
        code.extend([
            Instruction::load_constant(1, 1),
            Instruction::unary(OpCode::Negate, 1, 1),
        ]);
    }
    code.push(Instruction::return_values(0, 1));
    Chunk::with_parts(
        2,
        0,
        vec![Constant::Number(2.0), Constant::String("bad".into())],
        code,
        Vec::new(),
        upvalue_s(),
    )
}

/// `function(message) return message end`.
fn lua_handler() -> Chunk {
    Chunk::with_parts(
        1,
        1,
        Vec::new(),
        vec![Instruction::return_(0)],
        Vec::new(),
        Vec::new(),
    )
}

/// `local s = 1; local ok = <protected>(function() s = 2 ... end ...)`,
/// returning `s` and the status.
fn program(protected: &str, handler: Option<&str>, fail: bool) -> Chunk {
    let mut constants = vec![Constant::Number(1.0), Constant::String(protected.into())];
    let mut prototypes = vec![writer(fail)];
    let mut code = vec![
        Instruction::load_constant(0, 0),
        Instruction::get_global(1, 1),
        Instruction::closure(2, 0),
    ];
    let mut arguments = 2;
    match handler {
        Some("lua") => {
            prototypes.push(lua_handler());
            code.push(Instruction::closure(3, 1));
            arguments = 3;
        }
        Some(name) => {
            constants.push(Constant::String(name.into()));
            code.push(Instruction::get_global(3, 2));
            arguments = 3;
        }
        None => {}
    }
    code.push(Instruction::call(1, arguments, 2));
    code.push(Instruction::return_values(0, 2));
    Chunk::with_parts(5, 0, constants, code, prototypes, Vec::new())
}

fn run(profile: CompatibilityProfile, mode: ExecutionMode, chunk: &Chunk) -> Vec<Value> {
    let mut vm = Vm::with_profile_and_execution_mode(profile, mode);
    install_base_with_profile(vm.runtime_mut(), profile).unwrap();
    vm.execute(chunk).unwrap()
}

fn as_f64(value: &Value) -> f64 {
    match value {
        Value::Number(number) => *number,
        Value::Integer(number) => *number as f64,
        other => panic!("not a number: {other:?}"),
    }
}

#[test]
fn pcall_keeps_captured_write_after_error() {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let results = run(profile, mode, &program("pcall", None, true));
            assert_eq!(as_f64(&results[0]), 2.0, "{profile:?} {mode:?}");
            assert_eq!(results[1], Value::Boolean(false), "{profile:?} {mode:?}");
        }
    }
}

#[test]
fn xpcall_keeps_captured_write_with_lua_and_native_handlers() {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            for handler in ["lua", "tostring", "print"] {
                for fail in [true, false] {
                    let results = run(profile, mode, &program("xpcall", Some(handler), fail));
                    let context = format!("{profile:?} {mode:?} handler={handler} fail={fail}");
                    assert_eq!(as_f64(&results[0]), 2.0, "{context}");
                    assert_eq!(results[1], Value::Boolean(!fail), "{context}");
                }
            }
        }
    }
}

/// `return pcall(string.gsub, "a", "a", callback)`, where the callback raises:
/// the callback of a native that `pcall` called directly.
fn gsub_callback_program(callback: &str) -> Chunk {
    let mut prototypes = Vec::new();
    let mut code = vec![
        Instruction::get_global(0, 0),
        Instruction::get_global(1, 1),
        Instruction::get_field(1, 1, 2),
        Instruction::load_constant(2, 3),
        Instruction::load_constant(3, 3),
    ];
    if callback == "lua" {
        prototypes.push(Chunk::with_parts(
            2,
            0,
            vec![Constant::String("bad".into())],
            vec![
                Instruction::load_constant(0, 0),
                Instruction::unary(OpCode::Negate, 0, 0),
                Instruction::return_values(0, 1),
            ],
            Vec::new(),
            Vec::new(),
        ));
        code.push(Instruction::closure(4, 0));
    } else {
        code.push(Instruction::get_global(4, 4));
    }
    code.push(Instruction::call(0, 4, 3));
    code.push(Instruction::return_values(0, 2));
    Chunk::with_parts(
        6,
        0,
        vec![
            Constant::String("pcall".into()),
            Constant::String("string".into()),
            Constant::String("gsub".into()),
            Constant::String("a".into()),
            Constant::String("error".into()),
        ],
        code,
        prototypes,
        Vec::new(),
    )
}

#[test]
fn pcall_on_gsub_catches_errors_from_the_replacement_callback() {
    for profile in PROFILES {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            for callback in ["lua", "native"] {
                let results = run(profile, mode, &gsub_callback_program(callback));
                let context = format!("{profile:?} {mode:?} callback={callback}");
                assert_eq!(results[0], Value::Boolean(false), "{context}");
                assert!(matches!(results[1], Value::String(_)), "{context}");
            }
        }
    }
}
