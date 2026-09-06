//! Embeds the Oximond Lua runtime.
//!
//! [`Lua`] owns one VM configured for a compatibility profile and execution
//! mode. Its builder wires the compiler into `load`/`require`, installs the
//! standard library, and applies an optional package path, so hosts do not
//! assemble those crates by hand.
//!
//! ```
//! use oximond::{CompatibilityProfile, Lua, Value};
//!
//! let mut lua = Lua::builder().profile(CompatibilityProfile::Lua54).build()?;
//! let results = lua.exec("return 40 + 2")?;
//! assert_eq!(results, vec![Value::Integer(42)]);
//! # Ok::<(), oximond::Error>(())
//! ```

pub mod alloc;
pub mod bytecode;
pub mod compat;
pub mod compiler;
pub mod ffi_native;
pub mod jit;
pub mod jit_ir;
pub mod pattern;
pub mod runtime;
pub mod stdlib;
pub mod string;
pub mod syntax;
pub mod vm;

mod ffi;

use core::fmt;
use std::error::Error as StdError;
use std::sync::Arc;

use crate::compiler::{CompilerConfig, CompilerLoader, compile_named};
use crate::stdlib::{
    install_bit, install_full_with_profile_and_host_libraries, install_jit,
    install_pure_with_profile, install_pure_with_profile_and_loading, set_package_path,
};

pub use crate::bytecode::Chunk;
pub use crate::compat::{CompatibilityProfile, ExecutionMode, TracingOptFlags, TracingOptLevel};
pub use crate::compiler::CompileError;
pub use crate::runtime::{ExitRequest, Runtime, RuntimeError, RuntimeResult, Value};
pub use crate::string::LuaString;
pub use crate::vm::{HostCoroutineStatus, JitStatistics, Vm, VmError};

/// A compile or runtime failure from [`Lua`].
#[derive(Debug)]
pub enum Error {
    /// The source did not compile.
    Compile(CompileError),
    /// Execution or library installation failed.
    Runtime(VmError),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => error.fmt(formatter),
            Self::Runtime(error) => error.fmt(formatter),
        }
    }
}

impl StdError for Error {}

impl From<CompileError> for Error {
    fn from(error: CompileError) -> Self {
        Self::Compile(error)
    }
}

impl From<VmError> for Error {
    fn from(error: VmError) -> Self {
        Self::Runtime(error)
    }
}

impl From<RuntimeError> for Error {
    fn from(error: RuntimeError) -> Self {
        Self::Runtime(error.into())
    }
}

/// Selects which standard libraries a [`Lua`] instance installs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StandardLibrary {
    /// All libraries, including host-facing functions such as `io` and `os`.
    #[default]
    Full,
    /// Language and data libraries, without host-facing functions or `require`.
    Pure,
    /// No standard library globals.
    None,
}

/// `LuaJIT` extension modules that can be added without changing Lua syntax rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LuaJitExtension {
    /// The 32-bit `bit` operations.
    Bit,
    /// Native C declarations and calls. Only expose this to trusted code.
    Ffi,
    /// JIT engine controls and status.
    Jit,
}

/// Configures a [`Lua`] instance.
#[derive(Clone)]
#[must_use]
pub struct Builder {
    profile: CompatibilityProfile,
    execution_mode: ExecutionMode,
    tracing_opt_level: TracingOptLevel,
    tracing_opt_flags: Option<TracingOptFlags>,
    standard_library: StandardLibrary,
    extensions: Vec<LuaJitExtension>,
    luajit_default_extensions: bool,
    installers: Vec<HostInstaller>,
    source_reader: Option<SourceReader>,
    module_resolver: Option<ModuleResolver>,
    package_path: Option<LuaString>,
}

type Installer = Arc<dyn Fn(&mut Runtime) -> RuntimeResult<()>>;
type SourceReader = Arc<dyn Fn(Option<&[u8]>) -> std::io::Result<Vec<u8>>>;
type ModuleResolver = Arc<dyn Fn(&[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>, String>>;

#[derive(Clone)]
struct HostInstaller {
    name: Option<String>,
    install: Installer,
}

impl fmt::Debug for Builder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Builder")
            .field("profile", &self.profile)
            .field("execution_mode", &self.execution_mode)
            .field("tracing_opt_level", &self.tracing_opt_level)
            .field("tracing_opt_flags", &self.tracing_opt_flags)
            .field("standard_library", &self.standard_library)
            .field("extensions", &self.extensions)
            .field("luajit_default_extensions", &self.luajit_default_extensions)
            .field("installer_count", &self.installers.len())
            .field("has_source_reader", &self.source_reader.is_some())
            .field("has_module_resolver", &self.module_resolver.is_some())
            .field("package_path", &self.package_path)
            .finish()
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            profile: CompatibilityProfile::default(),
            execution_mode: ExecutionMode::default(),
            tracing_opt_level: TracingOptLevel::default(),
            tracing_opt_flags: None,
            standard_library: StandardLibrary::Full,
            extensions: Vec::new(),
            luajit_default_extensions: true,
            installers: Vec::new(),
            source_reader: None,
            module_resolver: None,
            package_path: None,
        }
    }
}

impl Builder {
    /// Selects the Lua language and library generation.
    pub const fn profile(mut self, profile: CompatibilityProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Selects the interpreter or the tracing engine.
    pub const fn execution_mode(mut self, execution_mode: ExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    /// Selects how much compile time the tracing engine may spend on hot
    /// traces. See [`TracingOptLevel`].
    pub const fn tracing_opt_level(mut self, level: TracingOptLevel) -> Self {
        self.tracing_opt_level = level;
        self
    }

    /// Replaces the optimizations of the selected level, for example
    /// `TracingOptLevel::O0.flags().without(TracingOptFlags::MEMORY)`.
    pub const fn tracing_opt_flags(mut self, flags: TracingOptFlags) -> Self {
        self.tracing_opt_flags = Some(flags);
        self
    }

    /// Whether to install the standard library (on by default).
    pub const fn stdlib(mut self, enabled: bool) -> Self {
        self.standard_library = if enabled {
            StandardLibrary::Full
        } else {
            StandardLibrary::None
        };
        self
    }

    /// Selects the full, pure, or absent standard library.
    pub const fn standard_library(mut self, standard_library: StandardLibrary) -> Self {
        self.standard_library = standard_library;
        self
    }

    /// Whether the `LuaJIT` 2.1 profile installs its built-in extension modules
    /// `bit`, `jit` and `ffi`, as `LuaJIT` itself does (on by default). Turn
    /// this off, then add modules with [`Self::luajit_extension`], to keep
    /// `ffi` away from untrusted code.
    pub const fn luajit_default_extensions(mut self, enabled: bool) -> Self {
        self.luajit_default_extensions = enabled;
        self
    }

    /// Adds a `LuaJIT` extension module independently of the compatibility profile.
    pub fn luajit_extension(mut self, extension: LuaJitExtension) -> Self {
        if !self.extensions.contains(&extension) {
            self.extensions.push(extension);
        }
        self
    }

    /// Registers a host implementation of a standard library global.
    ///
    /// The callback runs after the selected standard library is installed and
    /// must set the named global. In full mode, naming `io` or `os` omits that
    /// default library. Installed tables are also added to `package.loaded`
    /// when a package table is available. This also accepts `load`, `require`,
    /// and other function globals, so hosts can replace those builtins.
    pub fn library_installer(
        mut self,
        name: impl Into<String>,
        install: impl Fn(&mut Runtime) -> RuntimeResult<()> + 'static,
    ) -> Self {
        self.installers.push(HostInstaller {
            name: Some(name.into()),
            install: Arc::new(install),
        });
        self
    }

    /// Configures the runtime after library installation.
    ///
    /// This can install globals such as `print` and `warn`, set output writers,
    /// or register other host functions. Use [`Self::library_installer`] for a
    /// named library that should replace a default `io` or `os` table.
    pub fn host_installer(
        mut self,
        install: impl Fn(&mut Runtime) -> RuntimeResult<()> + 'static,
    ) -> Self {
        self.installers.push(HostInstaller {
            name: None,
            install: Arc::new(install),
        });
        self
    }

    /// Supplies the `io` table through [`Self::library_installer`].
    pub fn io_installer(
        self,
        install: impl Fn(&mut Runtime) -> RuntimeResult<()> + 'static,
    ) -> Self {
        self.library_installer("io", install)
    }

    /// Supplies the `os` table through [`Self::library_installer`].
    pub fn os_installer(
        self,
        install: impl Fn(&mut Runtime) -> RuntimeResult<()> + 'static,
    ) -> Self {
        self.library_installer("os", install)
    }

    /// Replaces the Lua-visible `load` function.
    pub fn load_installer(
        self,
        install: impl Fn(&mut Runtime) -> RuntimeResult<()> + 'static,
    ) -> Self {
        self.library_installer("load", install)
    }

    /// Replaces the Lua-visible `require` function.
    pub fn require_installer(
        self,
        install: impl Fn(&mut Runtime) -> RuntimeResult<()> + 'static,
    ) -> Self {
        self.library_installer("require", install)
    }

    /// Reads source bytes for `loadfile`, `dofile`, and path-based `require`.
    ///
    /// The argument is a path as raw bytes, or `None` for `dofile()` reading
    /// standard input. This callback replaces host filesystem and stdin reads.
    /// In pure mode it also enables those loading functions and `package`.
    pub fn source_reader(
        mut self,
        read: impl Fn(Option<&[u8]>) -> std::io::Result<Vec<u8>> + 'static,
    ) -> Self {
        self.source_reader = Some(Arc::new(read));
        self
    }

    /// Resolves a module by name before searching `package.path`.
    ///
    /// Return source bytes and a source name, or `None` to search the path.
    /// In pure mode, unresolved modules never fall back to the host filesystem.
    pub fn module_resolver(
        mut self,
        resolve: impl Fn(&[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>, String> + 'static,
    ) -> Self {
        self.module_resolver = Some(Arc::new(resolve));
        self
    }

    /// Replaces `package.path` after the standard library is installed.
    pub fn package_path(mut self, path: impl Into<LuaString>) -> Self {
        self.package_path = Some(path.into());
        self
    }

    /// Creates the configured instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the standard library cannot be installed.
    pub fn build(self) -> Result<Lua, Error> {
        let Self {
            profile,
            execution_mode,
            tracing_opt_level,
            tracing_opt_flags,
            standard_library,
            mut extensions,
            luajit_default_extensions,
            installers,
            source_reader,
            module_resolver,
            package_path,
        } = self;
        if profile == CompatibilityProfile::LuaJit21
            && luajit_default_extensions
            && standard_library != StandardLibrary::None
        {
            for extension in [
                LuaJitExtension::Bit,
                LuaJitExtension::Jit,
                #[cfg(not(target_arch = "wasm32"))]
                LuaJitExtension::Ffi,
            ] {
                if !extensions.contains(&extension) {
                    extensions.push(extension);
                }
            }
        }
        let has_source_reader = source_reader.is_some();
        let has_module_resolver = module_resolver.is_some();
        let custom_io = installers
            .iter()
            .any(|entry| entry.name.as_deref() == Some("io"));
        let custom_os = installers
            .iter()
            .any(|entry| entry.name.as_deref() == Some("os"));

        let mut vm = Box::new(Vm::with_profile_and_execution_mode(profile, execution_mode));
        vm.set_tracing_opt_level(tracing_opt_level);
        if let Some(flags) = tracing_opt_flags {
            vm.set_tracing_opt_flags(flags);
        }
        vm.set_chunk_loader(Box::new(CompilerLoader::with_profile(profile)));
        if let Some(reader) = source_reader {
            vm.set_source_reader(Box::new(move |path| reader(path)));
        } else if standard_library == StandardLibrary::Pure && has_module_resolver {
            vm.set_source_reader(Box::new(|_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no source reader configured",
                ))
            }));
        }
        if let Some(resolver) = module_resolver {
            vm.set_module_resolver(Box::new(move |name| resolver(name)));
        }

        match standard_library {
            StandardLibrary::Full => {
                install_full_with_profile_and_host_libraries(
                    vm.runtime_mut(),
                    profile,
                    !custom_io,
                    !custom_os,
                )?;
            }
            StandardLibrary::Pure if has_source_reader || has_module_resolver => {
                install_pure_with_profile_and_loading(vm.runtime_mut(), profile)?;
            }
            StandardLibrary::Pure => {
                install_pure_with_profile(vm.runtime_mut(), profile)?;
            }
            StandardLibrary::None => {}
        }
        let has_package = standard_library == StandardLibrary::Full
            || (standard_library == StandardLibrary::Pure
                && (has_source_reader || has_module_resolver));
        if let Some(path) = package_path.filter(|_| has_package) {
            set_package_path(vm.runtime_mut(), path)?;
        }
        for extension in extensions {
            match extension {
                LuaJitExtension::Bit => {
                    if matches!(vm.runtime().get_global("bit"), Value::Nil) {
                        install_bit(vm.runtime_mut())?;
                    }
                }
                LuaJitExtension::Ffi => ffi::install_ffi(vm.runtime_mut())?,
                LuaJitExtension::Jit => install_jit(vm.runtime_mut())?,
            }
        }
        for entry in installers {
            (entry.install)(vm.runtime_mut())?;
            if let Some(name) = entry.name {
                let runtime = vm.runtime_mut();
                let value = runtime.get_global(&name);
                if matches!(value, Value::Nil) {
                    return Err(RuntimeError::message(format!(
                        "host installer must set the {name} global"
                    ))
                    .into());
                }
                if (name == "io" || name == "os") && !matches!(value, Value::Table(_)) {
                    return Err(RuntimeError::message(format!(
                        "host installer must set the {name} global to a table"
                    ))
                    .into());
                }
                if matches!(value, Value::Table(_))
                    && let Value::Table(package) = runtime.get_global("package")
                    && let Value::Table(loaded) =
                        runtime.table_get(package, &Value::from("loaded"))?
                {
                    runtime.table_set(loaded, &Value::from(name), value)?;
                }
            }
        }
        Ok(Lua { vm, profile })
    }
}

/// One Lua state: a VM plus the compiler configured for its profile.
pub struct Lua {
    /// Boxed: the VM is 10 KB, and a start would copy it through the stack
    /// frames that build and return a `Lua`, a page fault for each 4 KB.
    vm: Box<Vm>,
    profile: CompatibilityProfile,
}

impl Lua {
    /// Starts configuring an instance.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Creates an interpreter-mode instance with the default profile and the
    /// standard library.
    ///
    /// # Errors
    ///
    /// Returns an error if the standard library cannot be installed.
    pub fn new() -> Result<Self, Error> {
        Self::builder().build()
    }

    /// The selected compatibility profile.
    #[must_use]
    pub const fn profile(&self) -> CompatibilityProfile {
        self.profile
    }

    /// Compiles source for this instance's profile.
    ///
    /// # Errors
    ///
    /// Returns the syntax or compiler diagnostic.
    pub fn compile(
        &self,
        name: impl Into<LuaString>,
        source: impl AsRef<[u8]>,
    ) -> Result<Chunk, CompileError> {
        compile_named(name, source, CompilerConfig::new(self.profile))
    }

    /// Compiles and runs a source chunk named `=(load)`.
    ///
    /// # Errors
    ///
    /// Returns a compile error or the Lua error raised while running.
    pub fn exec(&mut self, source: impl AsRef<[u8]>) -> Result<Vec<Value>, Error> {
        let chunk = self.compile("=(load)", source)?;
        Ok(self.vm.execute(&chunk)?)
    }

    /// Runs a compiled chunk with no arguments.
    ///
    /// # Errors
    ///
    /// Returns the Lua error raised while running.
    pub fn execute(&mut self, chunk: &Chunk) -> Result<Vec<Value>, VmError> {
        self.vm.execute(chunk)
    }

    /// Runs a compiled chunk with `...` bound to `arguments`.
    ///
    /// # Errors
    ///
    /// Returns the Lua error raised while running.
    pub fn execute_with_arguments(
        &mut self,
        chunk: &Chunk,
        arguments: Vec<Value>,
    ) -> Result<Vec<Value>, VmError> {
        self.vm.execute_with_arguments(chunk, arguments)
    }

    /// The underlying VM, for statistics and advanced configuration.
    #[must_use]
    pub const fn vm(&self) -> &Vm {
        &self.vm
    }

    /// Mutable access to the underlying VM.
    pub const fn vm_mut(&mut self) -> &mut Vm {
        &mut self.vm
    }

    /// Transfers the configured VM to an embedding host.
    #[must_use]
    pub fn into_vm(self) -> Vm {
        *self.vm
    }

    /// Mutable access to the heap, globals, and native registration.
    pub fn runtime_mut(&mut self) -> &mut Runtime {
        self.vm.runtime_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    fn sandbox_read(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::from("from host")])
    }

    fn virtual_load(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::from("virtual load")])
    }

    fn virtual_require(_: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        Ok(vec![Value::from("virtual module")])
    }

    fn host_print(runtime: &mut Runtime, _: &[Value]) -> RuntimeResult<Vec<Value>> {
        runtime.write_output(b"host print\n")?;
        Ok(Vec::new())
    }

    #[test]
    fn every_profile_runs_the_standard_library_with_interpreter_fallback() {
        for profile in CompatibilityProfile::ALL {
            for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
                let mut lua = Lua::builder()
                    .profile(profile)
                    .execution_mode(mode)
                    .build()
                    .unwrap();
                let results = lua
                    .exec("return string.rep('a', 3), select('#', 1, 2), _VERSION")
                    .unwrap();
                assert_eq!(results[0], Value::from("aaa"), "{profile} {mode}");
                assert_eq!(results[2], Value::from(profile.identity().version_string()));
            }
        }
    }

    #[test]
    fn os_exit_returns_an_uncatchable_request_to_the_host() {
        for profile in CompatibilityProfile::ALL {
            let mut lua = Lua::builder().profile(profile).build().unwrap();
            let Err(Error::Runtime(error)) =
                lua.exec("pcall(os.exit, 3, true) error('not reached')")
            else {
                panic!("{profile}: os.exit must end the chunk");
            };
            let close = profile != CompatibilityProfile::Lua51;
            assert_eq!(
                error.exit_request(),
                Some(ExitRequest { status: 3, close }),
                "{profile}"
            );
            // The host decides what exiting means, and the state stays usable.
            assert_eq!(
                lua.exec("return type(os.exit)").unwrap(),
                vec![Value::from("function")],
                "{profile}"
            );
        }
    }

    #[test]
    fn pure_standard_library_has_no_os_exit() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .build()
            .unwrap();
        assert_eq!(lua.exec("return os").unwrap(), vec![Value::Nil]);
    }

    #[test]
    fn tracing_execution_runs_numeric_code() {
        let mut lua = Lua::builder()
            .execution_mode(ExecutionMode::Tracing)
            .build()
            .unwrap();
        let chunk = lua
            .compile(
                "=numeric",
                "local s = 0 for i = 1, 10 do s = s + i end return s",
            )
            .unwrap();
        assert_eq!(lua.execute(&chunk).unwrap(), vec![Value::Number(55.0)]);
    }

    #[test]
    fn load_uses_the_instance_profile() {
        let mut lua = Lua::builder()
            .profile(CompatibilityProfile::Lua54)
            .build()
            .unwrap();
        let results = lua.exec("return load('return 7 // 2')()").unwrap();
        assert_eq!(results, vec![Value::Integer(3)]);
    }

    #[test]
    fn reports_compile_and_runtime_errors_separately() {
        let mut lua = Lua::new().unwrap();
        assert!(matches!(lua.exec("return +"), Err(Error::Compile(_))));
        assert!(matches!(lua.exec("error('boom')"), Err(Error::Runtime(_))));
    }

    #[test]
    fn stdlib_can_be_omitted_and_package_path_set() {
        let mut bare = Lua::builder().stdlib(false).build().unwrap();
        assert_eq!(bare.exec("return print").unwrap(), vec![Value::Nil]);

        let mut lua = Lua::builder().package_path("/x/?.lua").build().unwrap();
        assert_eq!(
            lua.exec("return package.path").unwrap(),
            vec![Value::from("/x/?.lua")]
        );
    }

    #[test]
    fn pure_standard_library_has_no_host_functions() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .build()
            .unwrap();
        assert_eq!(
            lua.exec("return io, os, print, loadfile, dofile, require, package")
                .unwrap(),
            vec![Value::Nil; 7]
        );
        assert_eq!(
            lua.exec("return string.rep('a', 2), math.abs(-4), type({})")
                .unwrap(),
            vec![Value::from("aa"), Value::Number(4.0), Value::from("table")]
        );
        assert_eq!(
            lua.exec("return load('return 3')()").unwrap(),
            vec![Value::Number(3.0)]
        );
    }

    #[test]
    fn custom_io_replaces_default_and_is_registered_with_package() {
        let label = String::from("sandbox");
        let builder = Lua::builder()
            .standard_library(StandardLibrary::Full)
            .io_installer(move |runtime| {
                let io = runtime.create_table()?;
                runtime.table_set(io, &Value::from("kind"), Value::from(label.clone()))?;
                let read = runtime.register_native(sandbox_read)?;
                runtime.table_set(io, &Value::from("read"), Value::NativeFunction(read))?;
                runtime.set_global("io", Value::Table(io));
                Ok(())
            });
        let mut clone = builder.clone().build().unwrap();
        assert_eq!(
            clone.exec("return io.read()").unwrap(),
            vec![Value::from("from host")]
        );
        let mut lua = builder.build().unwrap();
        assert_eq!(lua.runtime_mut().default_input(), None);
        assert_eq!(lua.runtime_mut().default_output(), None);
        assert_eq!(
            lua.exec(
                "return io.kind, io.read(), io.open, require('io') == io, type(os), type(package)"
            )
            .unwrap(),
            vec![
                Value::from("sandbox"),
                Value::from("from host"),
                Value::Nil,
                Value::Boolean(true),
                Value::from("table"),
                Value::from("table"),
            ]
        );
    }

    #[test]
    fn custom_io_requires_an_io_table() {
        assert!(Lua::builder().io_installer(|_| Ok(())).build().is_err());
    }

    #[test]
    fn full_runtime_replaces_os_without_leaving_default_functions() {
        let mut lua = Lua::builder()
            .os_installer(|runtime| {
                let os = runtime.create_table()?;
                runtime.table_set(os, &Value::from("kind"), Value::from("host"))?;
                runtime.set_global("os", Value::Table(os));
                Ok(())
            })
            .build()
            .unwrap();
        assert_eq!(
            lua.exec("return os.kind, os.execute, package.loaded.os == os")
                .unwrap(),
            vec![Value::from("host"), Value::Nil, Value::Boolean(true)]
        );
    }

    #[test]
    fn host_installer_routes_default_print_to_a_supplied_writer() {
        struct Capture(Arc<Mutex<Vec<u8>>>);

        impl Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let output = Arc::new(Mutex::new(Vec::new()));
        let writer_output = Arc::clone(&output);
        let mut lua = Lua::builder()
            .host_installer(move |runtime| {
                runtime.set_output_writer(Capture(Arc::clone(&writer_output)));
                Ok(())
            })
            .build()
            .unwrap();
        lua.exec("print('hello')").unwrap();
        assert_eq!(output.lock().unwrap().as_slice(), b"hello\n");
    }

    #[test]
    fn pure_runtime_can_replace_load_and_require() {
        for mode in [ExecutionMode::Interpreter, ExecutionMode::Tracing] {
            let mut lua = Lua::builder()
                .execution_mode(mode)
                .standard_library(StandardLibrary::Pure)
                .load_installer(|runtime| {
                    runtime.register_global_native("load", virtual_load)?;
                    Ok(())
                })
                .require_installer(|runtime| {
                    runtime.register_global_native("require", virtual_require)?;
                    Ok(())
                })
                .build()
                .unwrap();
            assert_eq!(
                lua.exec("return load('return 1'), require('virtual')")
                    .unwrap(),
                vec![Value::from("virtual load"), Value::from("virtual module")]
            );
        }
    }

    #[test]
    fn pure_runtime_can_run_a_virtual_project_with_selected_host_libraries() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .package_path("virtual/?.lua")
            .source_reader(|path| match path {
                Some(b"virtual/main.lua") => Ok(b"return require('app.mod') + 1".to_vec()),
                Some(b"virtual/app/mod.lua" | b"virtual/app\\mod.lua") => Ok(b"return 41".to_vec()),
                None => Ok(b"return 7".to_vec()),
                _ => Err(std::io::Error::from(std::io::ErrorKind::NotFound)),
            })
            .io_installer(|runtime| {
                let io = runtime.create_table()?;
                runtime.table_set(io, &Value::from("kind"), Value::from("virtual"))?;
                runtime.set_global("io", Value::Table(io));
                Ok(())
            })
            .os_installer(|runtime| {
                let os = runtime.create_table()?;
                runtime.table_set(os, &Value::from("kind"), Value::from("virtual"))?;
                runtime.set_global("os", Value::Table(os));
                Ok(())
            })
            .host_installer(|runtime| {
                runtime.register_global_native("print", host_print)?;
                Ok(())
            })
            .build()
            .unwrap();
        assert_eq!(
            lua.exec("return dofile('virtual/main.lua'), loadfile('virtual/main.lua')()")
                .unwrap(),
            vec![Value::Number(42.0), Value::Number(42.0)]
        );
        assert_eq!(
            lua.exec("return io.kind, os.kind, os.execute, package.loaded.io == io, package.loaded.os == os")
                .unwrap(),
            vec![
                Value::from("virtual"),
                Value::from("virtual"),
                Value::Nil,
                Value::Boolean(true),
                Value::Boolean(true),
            ]
        );
        lua.exec("print('hello')").unwrap();
        assert_eq!(lua.runtime_mut().take_output(), b"host print\n");
    }

    #[test]
    fn pure_module_resolver_does_not_enable_host_file_fallback() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .module_resolver(|name| {
                if name == b"virtual" {
                    Ok(Some((b"return 9".to_vec(), b"=virtual".to_vec())))
                } else {
                    Ok(None)
                }
            })
            .build()
            .unwrap();
        assert_eq!(
            lua.exec("return require('virtual')").unwrap(),
            vec![Value::Number(9.0)]
        );
        assert_eq!(
            lua.exec("return loadfile('/etc/passwd') == nil").unwrap(),
            vec![Value::Boolean(true)]
        );
    }

    #[test]
    fn bit_extension_can_be_added_without_changing_language_profile() {
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
            let mut lua = Lua::builder()
                .profile(profile)
                .standard_library(StandardLibrary::Pure)
                .luajit_extension(LuaJitExtension::Bit)
                .build()
                .unwrap();
            assert_eq!(lua.profile(), profile);
            assert_eq!(
                lua.exec("return bit.bxor(6, 3)").unwrap(),
                vec![Value::Number(5.0)]
            );
        }
    }

    #[test]
    fn jit_extension_reports_and_controls_the_selected_engine() {
        let mut interpreter = Lua::builder()
            .execution_mode(ExecutionMode::Interpreter)
            .luajit_extension(LuaJitExtension::Jit)
            .build()
            .unwrap();
        assert_eq!(
            interpreter.exec("jit.on(); return jit.status()").unwrap(),
            vec![Value::Boolean(false)]
        );

        let mut tracing = Lua::builder()
            .execution_mode(ExecutionMode::Tracing)
            .luajit_extension(LuaJitExtension::Jit)
            .build()
            .unwrap();
        assert_eq!(
            tracing
                .exec("local before = jit.status(); jit.off(); local after = jit.status(); jit.on(); return before, after, jit.status(), package.loaded.jit == jit")
                .unwrap(),
            vec![
                Value::Boolean(true),
                Value::Boolean(false),
                Value::Boolean(true),
                Value::Boolean(true),
            ]
        );
        tracing.exec("jit.flush()").unwrap();
        assert!(!tracing.runtime_mut().jit_flush_requested());
    }

    #[test]
    fn ffi_extension_calls_declared_native_symbols_on_multiple_profiles() {
        for profile in [CompatibilityProfile::Lua51, CompatibilityProfile::Lua54] {
            let mut lua = Lua::builder()
                .profile(profile)
                .luajit_extension(LuaJitExtension::Ffi)
                .build()
                .unwrap();
            assert_eq!(lua.profile(), profile);
            let results = lua
                .exec(
                    "ffi.cdef('int abs(int); size_t strlen(const char *);'); return ffi.C.abs(-42), ffi.C.strlen('hello'), require('ffi') == ffi, ffi.sizeof('int')",
                )
                .unwrap();
            assert_eq!(
                results,
                vec![
                    Value::Integer(42),
                    Value::Integer(5),
                    Value::Boolean(true),
                    Value::Integer(4),
                ]
            );
            assert_eq!(
                lua.exec("local n = ffi.cast('int', 7); return ffi.istype('int', n), type(ffi.new('int', 9))")
                    .unwrap(),
                vec![Value::Boolean(true), Value::from("userdata")]
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ffi_extension_loads_an_explicit_native_library() {
        let mut lua = Lua::builder()
            .standard_library(StandardLibrary::Pure)
            .luajit_extension(LuaJitExtension::Ffi)
            .build()
            .unwrap();
        assert_eq!(
            lua.exec("ffi.cdef('int abs(int);'); local libc = ffi.load('libc.so.6'); return libc.abs(-17)")
                .unwrap(),
            vec![Value::Integer(17)]
        );
    }
}
