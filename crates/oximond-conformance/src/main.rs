use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use oximond_conformance::{
    Ledger, Sources, default_ledger_path, default_sources_path, prepare, render_report,
    run_required,
};

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("oximond-conformance: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Vec<std::ffi::OsString>) -> Result<(), String> {
    let options = parse_arguments(arguments)?;
    if options.command == "help" {
        print_help();
        return Ok(());
    }
    oximond_conformance::set_execution_arguments(options.execution_arguments.clone());
    oximond_conformance::set_trace_profile_directories(
        options.trace_profile_record_dir.clone(),
        options.trace_profile_load_dir.clone(),
    );
    let sources = Sources::load(&default_sources_path())?;
    let ledger = Ledger::load(&default_ledger_path())?;

    match options.command.as_str() {
        "prepare" => prepare(&sources, &options.directory),
        "report" => {
            print!("{}", render_report(&sources, &ledger, &options.directory)?);
            Ok(())
        }
        "check-ledger" => ledger.validate(&sources),
        "run" => {
            let result = run_required(&sources, &ledger, &options.directory, &options.oximond)?;
            print!("{}", result.report());
            if result.passed() {
                Ok(())
            } else {
                Err(result.failure_message())
            }
        }
        _ => Err(format!(
            "unknown command `{}`. Run `oximond-conformance help`.",
            options.command
        )),
    }
}

struct Options {
    command: String,
    directory: PathBuf,
    oximond: PathBuf,
    execution_arguments: Vec<String>,
    trace_profile_record_dir: Option<PathBuf>,
    trace_profile_load_dir: Option<PathBuf>,
}

fn parse_arguments(arguments: Vec<std::ffi::OsString>) -> Result<Options, String> {
    let mut command = None;
    let mut directory = PathBuf::from(".oximond/conformance");
    // The per-test deadline is calibrated for optimized code, so the gate
    // tests the release CLI unless told otherwise.
    let mut oximond = PathBuf::from("target/release/oximond");
    let mut execution_arguments = Vec::new();
    let mut trace_profile_record_dir = None;
    let mut trace_profile_load_dir = None;
    let mut iterator = arguments.into_iter();

    while let Some(argument) = iterator.next() {
        if argument == "--dir" {
            let Some(value) = iterator.next() else {
                return Err("`--dir` needs a directory.".to_owned());
            };
            directory = PathBuf::from(value);
        } else if argument == "--oximond" {
            let Some(value) = iterator.next() else {
                return Err("`--oximond` needs an executable path.".to_owned());
            };
            oximond = PathBuf::from(value);
        } else if argument == "--execution" {
            let Some(value) = iterator.next() else {
                return Err(format!("`{}` needs a value.", argument.to_string_lossy()));
            };
            execution_arguments.push(argument.to_string_lossy().into_owned());
            execution_arguments.push(value.to_string_lossy().into_owned());
        } else if argument == "--trace-profile-record-dir" || argument == "--trace-profile-load-dir"
        {
            let Some(value) = iterator.next() else {
                return Err(format!(
                    "`{}` needs a directory.",
                    argument.to_string_lossy()
                ));
            };
            let directory = PathBuf::from(value);
            if argument == "--trace-profile-record-dir" {
                trace_profile_record_dir = Some(directory);
            } else {
                trace_profile_load_dir = Some(directory);
            }
        } else if argument == "--help" || argument == "-h" {
            command = Some("help".to_owned());
        } else if command.is_none() {
            command = Some(argument.to_string_lossy().into_owned());
        } else {
            return Err(format!(
                "unexpected argument `{}`.",
                argument.to_string_lossy()
            ));
        }
    }

    if (trace_profile_record_dir.is_some() || trace_profile_load_dir.is_some())
        && !execution_arguments
            .windows(2)
            .any(|args| args == ["--execution", "tracing"])
    {
        return Err("trace profile directories require `--execution tracing`.".to_owned());
    }
    if let Some(directory) = &trace_profile_record_dir {
        std::fs::create_dir_all(directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        trace_profile_record_dir =
            Some(std::fs::canonicalize(directory).map_err(|error| error.to_string())?);
    }
    if let Some(directory) = &trace_profile_load_dir {
        trace_profile_load_dir = Some(
            std::fs::canonicalize(directory)
                .map_err(|error| format!("{}: {error}", directory.display()))?,
        );
    }
    Ok(Options {
        command: command.unwrap_or_else(|| "report".to_owned()),
        directory,
        oximond,
        execution_arguments,
        trace_profile_record_dir,
        trace_profile_load_dir,
    })
}

fn print_help() {
    println!(
        "Usage: oximond-conformance <prepare|report|check-ledger|run> [OPTIONS]\n\
         \n\
         prepare       Download pinned corpora and apply their declared patch sets.\n\
         report        Print a ledger section for each declared corpus.\n\
         check-ledger  Validate the checked-in source and ledger manifests.\n\
         run           Rebuild patched suites and run exact executable ledger entries.\n\
         \n\
         Options:\n\
           --dir DIRECTORY   Store or find corpora in DIRECTORY.\n\
           --oximond PATH    Use PATH as the Oximond executable for run\n\
         \x20                 (default target/release/oximond).\n\
           --execution MODE  Pass `--execution MODE` to every test process.\n\
           --trace-profile-record-dir DIR  Record one profile per test in DIR.\n\
           --trace-profile-load-dir DIR    Load each test's profile from DIR.\n\
         \n\
         Environment:\n\
           OXIMOND_CONFORMANCE_TIMEOUT_SECS     Per-test deadline in seconds (default 60).\n\
           OXIMOND_CONFORMANCE_MEMORY_LIMIT_MIB Per-test address-space cap in MiB\n\
         \x20                                     (default 4096, 0 disables).\n\
         \n\
         Corpora live in .oximond/conformance by default and are never vendored.\n\
         Patch paths in tests/conformance/sources.toml are relative to that manifest.\n\
         Patched runs use separate copies and report the applied patch sets."
    );
}
