#![allow(unsafe_code)]

//! Native C calls for the optional `ffi` module.
//!
//! This module deliberately contains unsafe code, the only place in the crate
//! that does. A C declaration is a promise by the caller that the native symbol
//! has that ABI and that pointers are valid. A wrong declaration or invalid
//! pointer can crash the process. Embedders must only expose this engine to
//! trusted scripts.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::ptr::{self, NonNull};

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
    fn oximond_ffi_open(name: *const c_char) -> *mut c_void;
    fn oximond_ffi_close(handle: *mut c_void);
    fn oximond_ffi_symbol(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn oximond_ffi_error() -> *const c_char;
    fn oximond_ffi_errno() -> c_int;
    fn oximond_ffi_set_errno(value: c_int);
    fn oximond_ffi_call(
        function: *mut c_void,
        result_type: u8,
        arg_types: *const u8,
        arg_values: *mut *mut c_void,
        arg_count: usize,
        fixed_count: usize,
        result: *mut c_void,
    ) -> c_int;
}

// Keep the public API linkable on wasm without invoking a C compiler. Opening a
// native library always fails, so native calls cannot reach the inert shims.
#[cfg(target_arch = "wasm32")]
mod wasm_unsupported {
    use super::{c_char, c_int, c_void};
    use std::ptr;

    pub(super) unsafe fn oximond_ffi_open(_: *const c_char) -> *mut c_void {
        ptr::null_mut()
    }

    pub(super) unsafe fn oximond_ffi_close(_: *mut c_void) {}

    pub(super) unsafe fn oximond_ffi_symbol(_: *mut c_void, _: *const c_char) -> *mut c_void {
        ptr::null_mut()
    }

    pub(super) unsafe fn oximond_ffi_error() -> *const c_char {
        c"native FFI is unsupported on this target".as_ptr()
    }

    pub(super) unsafe fn oximond_ffi_errno() -> c_int {
        0
    }

    pub(super) unsafe fn oximond_ffi_set_errno(_: c_int) {}

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn oximond_ffi_call(
        _: *mut c_void,
        _: u8,
        _: *const u8,
        _: *mut *mut c_void,
        _: usize,
        _: usize,
        _: *mut c_void,
    ) -> c_int {
        -1
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_unsupported::*;

/// Types accepted in a C function declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FfiType {
    Void,
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    SizeT,
    Float,
    Double,
    Pointer,
    CString,
}

impl FfiType {
    fn code(self) -> u8 {
        match self {
            Self::Void => 0,
            Self::Char => 10,
            Self::UChar => 11,
            Self::Short => 12,
            Self::UShort => 13,
            Self::Int => 1,
            Self::UInt => 2,
            Self::Long => 3,
            Self::ULong => 4,
            Self::LongLong => 5,
            Self::ULongLong => 6,
            Self::SizeT => {
                if cfg!(target_pointer_width = "64") {
                    6
                } else {
                    2
                }
            }
            Self::Float => 7,
            Self::Double => 8,
            Self::Pointer | Self::CString => 9,
        }
    }
}

/// Values passed to and returned from native calls.
#[derive(Clone, Debug, PartialEq)]
pub enum FfiValue {
    Void,
    Integer(i64),
    Unsigned(u64),
    Float(f32),
    Double(f64),
    Pointer(usize),
    /// Bytes are passed with a terminating NUL and must contain no interior NUL.
    CString(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfiError(pub String);

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FfiError {}

type Result<T> = std::result::Result<T, FfiError>;

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub result: FfiType,
    pub args: Vec<FfiType>,
    pub variadic: bool,
}

/// A scalar C object with stable storage. Its address can be passed as a pointer.
pub struct FfiScalar {
    ty: FfiType,
    storage: Box<Slot>,
    _strings: Vec<CString>,
}

impl FfiScalar {
    #[must_use]
    pub fn pointer(&self) -> usize {
        self.storage.0.as_ptr() as usize
    }

    // Decoding a supported scalar cannot fail; the `Result` matches the other readers.
    #[allow(clippy::missing_errors_doc)]
    pub fn value(&self) -> Result<FfiValue> {
        decode_result(self.ty, &self.storage)
    }
}

/// A contiguous native array of one supported scalar or pointer type.
/// The storage remains at one address until this object is dropped.
pub struct FfiArray {
    ty: FfiType,
    len: usize,
    element_size: usize,
    layout: Layout,
    memory: NonNull<u8>,
    strings: Vec<Option<CString>>,
}

impl FfiArray {
    #[must_use]
    pub fn pointer(&self) -> usize {
        self.memory.as_ptr() as usize
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read the element at `index` as this array's C type.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Result<FfiValue> {
        let offset = self.offset(index)?;
        let mut slot = Slot([0; 16]);
        // SAFETY: offset is checked against the allocation and slot fits the scalar.
        unsafe {
            ptr::copy_nonoverlapping(
                self.memory.as_ptr().add(offset),
                slot.0.as_mut_ptr(),
                self.element_size,
            );
        }
        decode_result(self.ty, &slot)
    }

    /// Write `value` to the element at `index`, keeping any C string alive.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `index` is out of bounds, `value` does not match
    /// the element type, or a C string value contains a NUL byte.
    pub fn set(&mut self, index: usize, value: &FfiValue) -> Result<()> {
        let offset = self.offset(index)?;
        let mut strings = Vec::new();
        let slot = encode_argument(self.ty, value, &mut strings)?;
        // SAFETY: offset is checked against the allocation and slot contains the scalar bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                slot.0.as_ptr(),
                self.memory.as_ptr().add(offset),
                self.element_size,
            );
        }
        self.strings[index] = strings.pop();
        Ok(())
    }

    fn offset(&self, index: usize) -> Result<usize> {
        if index >= self.len {
            return Err(FfiError(format!(
                "array index {index} out of bounds for length {}",
                self.len
            )));
        }
        Ok(index * self.element_size)
    }
}

impl Drop for FfiArray {
    fn drop(&mut self) {
        // SAFETY: memory was allocated with this exact layout and is owned here.
        unsafe {
            dealloc(self.memory.as_ptr(), self.layout);
        }
    }
}

/// A dynamically opened native library. The library stays loaded until dropped.
pub struct Library {
    handle: *mut c_void,
}

impl Library {
    /// `None` opens the process namespace, as with `LuaJIT`'s `ffi.C`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `name` contains a NUL byte or the library cannot
    /// be loaded.
    pub fn open(name: Option<&str>) -> Result<Self> {
        let name = name
            .map(|s| CString::new(s).map_err(|_| FfiError("library name contains NUL".into())))
            .transpose()?;
        // SAFETY: `name` is a valid, NUL-terminated C string or null.
        let handle = unsafe { oximond_ffi_open(name.as_ref().map_or(ptr::null(), |s| s.as_ptr())) };
        if handle.is_null() {
            return Err(loader_error("dlopen failed"));
        }
        Ok(Self { handle })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: `handle` is a successful dlopen result owned by this object.
        unsafe {
            oximond_ffi_close(self.handle);
        }
    }
}

/// Declaration registry and the process namespace used by `ffi.C`.
pub struct FfiEngine {
    process: Library,
    declarations: HashMap<String, FunctionDecl>,
    typedefs: HashMap<String, FfiType>,
}

impl FfiEngine {
    /// Create an engine backed by the process namespace.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if the process namespace cannot be opened.
    pub fn new() -> Result<Self> {
        Ok(Self {
            process: Library::open(None)?,
            declarations: HashMap::new(),
            typedefs: HashMap::new(),
        })
    }

    /// Register C declarations, applied atomically: nothing is added on failure.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] for an unterminated block comment, a struct or
    /// union declaration, a duplicate or unsupported type name, or any
    /// declaration that cannot be parsed.
    pub fn cdef(&mut self, declarations: &str) -> Result<()> {
        let declarations = strip_c_comments(declarations)?;
        if declarations.contains(['{', '}']) {
            return Err(FfiError(
                "struct and union declarations are unsupported".into(),
            ));
        }
        let mut functions = self.declarations.clone();
        let mut typedefs = self.typedefs.clone();
        for declaration in declarations
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            if let Some(body) = declaration.strip_prefix("typedef ") {
                let (name, ty) = parse_typedef(body, &typedefs)?;
                if is_known_type(&name)
                    || typedefs.get(&name).is_some_and(|existing| *existing != ty)
                {
                    return Err(FfiError(format!("duplicate C type name: {name}")));
                }
                typedefs.insert(name, ty);
            } else {
                let (name, signature) = parse_declaration(declaration, &typedefs)?;
                functions.insert(name, signature);
            }
        }
        self.declarations = functions;
        self.typedefs = typedefs;
        Ok(())
    }

    /// Resolve a type declared through `cdef`, including typedef aliases.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is not a supported or declared type.
    pub fn type_of(&self, ctype: &str) -> Result<FfiType> {
        parse_type_with_aliases(ctype, false, &self.typedefs)
    }

    /// Size in bytes of a C type declared through `cdef`, including arrays.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is unknown or unsupported (including
    /// `void`), names a malformed array, or its size overflows `usize`.
    pub fn sizeof_type(&self, ctype: &str) -> Result<usize> {
        if ctype.contains('[') {
            let (element, count) = parse_array_declaration(ctype)?;
            return self
                .sizeof_type(element)?
                .checked_mul(count)
                .ok_or_else(|| FfiError("array size overflow".into()));
        }
        size_of_type(self.type_of(ctype)?)
    }

    /// Alignment in bytes of a C type declared through `cdef`, including arrays.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is unknown or unsupported (including
    /// `void`) or names a malformed array.
    pub fn alignof_type(&self, ctype: &str) -> Result<usize> {
        if ctype.contains('[') {
            let (element, _) = parse_array_declaration(ctype)?;
            return self.alignof_type(element);
        }
        align_of_type(self.type_of(ctype)?)
    }

    /// Convert a Lua numeric or pointer value to a type declared through `cdef`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is unknown or `value` does not fit it,
    /// including a temporary C string cast to a lasting pointer.
    pub fn cast_type(&self, ctype: &str, value: &FfiValue) -> Result<FfiValue> {
        cast_to_type(self.type_of(ctype)?, value)
    }

    /// Allocate a scalar C object of a type declared through `cdef`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is unknown or `value` does not fit it.
    pub fn new_scalar_type(&self, ctype: &str, value: &FfiValue) -> Result<FfiScalar> {
        scalar_of_type(self.type_of(ctype)?, value)
    }

    /// Allocate a fixed-size array from a declaration such as `int[4]`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] for a malformed or unsupported declaration, a size
    /// overflow, or a failed allocation.
    pub fn new_array_type(&self, cdecl: &str) -> Result<FfiArray> {
        let (ctype, count) = parse_array_declaration(cdecl)?;
        array_of_type(self.type_of(ctype)?, count)
    }

    #[must_use]
    pub fn declaration(&self, name: &str) -> Option<&FunctionDecl> {
        self.declarations.get(name)
    }

    pub fn declared_functions(&self) -> impl Iterator<Item = (&str, &FunctionDecl)> {
        self.declarations
            .iter()
            .map(|(name, declaration)| (name.as_str(), declaration))
    }

    /// Size in bytes of a supported scalar or pointer C type.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is not a supported scalar or pointer
    /// type (including `void`), names a malformed array, or its size overflows
    /// `usize`.
    pub fn sizeof(ctype: &str) -> Result<usize> {
        if ctype.contains('[') {
            let (element, count) = parse_array_declaration(ctype)?;
            return Self::sizeof(element)?
                .checked_mul(count)
                .ok_or_else(|| FfiError("array size overflow".into()));
        }
        size_of_type(parse_type(ctype, false)?)
    }

    /// Alignment in bytes of a supported scalar or pointer C type.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is not a supported scalar or pointer
    /// type (including `void`) or names a malformed array.
    pub fn alignof(ctype: &str) -> Result<usize> {
        if ctype.contains('[') {
            let (element, _) = parse_array_declaration(ctype)?;
            return Self::alignof(element);
        }
        align_of_type(parse_type(ctype, false)?)
    }

    /// Query a `LuaJIT` ABI flag for this compiled native target.
    // Each arm names one accepted `LuaJIT` ABI flag; `softfp` is recognized and reported false.
    #[allow(clippy::match_same_arms)]
    #[must_use]
    pub fn abi(flag: &str) -> bool {
        match flag {
            "32bit" => cfg!(target_pointer_width = "32"),
            "64bit" => cfg!(target_pointer_width = "64"),
            "le" => cfg!(target_endian = "little"),
            "be" => cfg!(target_endian = "big"),
            "win" => cfg!(target_os = "windows"),
            "unix" => cfg!(unix),
            "fpu" => cfg!(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64"
            )),
            "softfp" => false,
            "hardfp" => cfg!(all(target_arch = "arm", target_abi = "eabihf")),
            "eabi" => cfg!(target_arch = "arm"),
            _ => false,
        }
    }
    /// Get the calling thread's native `errno` value.
    #[must_use]
    pub fn errno() -> i32 {
        // SAFETY: the C shim reads the platform's thread-local errno slot.
        unsafe { oximond_ffi_errno() }
    }

    /// Set the calling thread's native `errno` value.
    pub fn set_errno(value: i32) {
        // SAFETY: the C shim writes the platform's thread-local errno slot.
        unsafe { oximond_ffi_set_errno(value) }
    }

    /// Copy `len` bytes between native pointers. Overlapping ranges are allowed.
    /// Both ranges must be live and large enough or this can crash the process.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if either range is null or wraps the address space.
    pub fn copy(destination: usize, source: usize, len: usize) -> Result<()> {
        validate_range(destination, len)?;
        validate_range(source, len)?;
        if len != 0 {
            // SAFETY: pointer validity and capacity are the caller's FFI contract.
            unsafe { ptr::copy(source as *const u8, destination as *mut u8, len) };
        }
        Ok(())
    }

    /// Copy bytes from a Rust slice to native memory.
    /// The destination must be live and large enough.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if the destination range is null or wraps the
    /// address space.
    pub fn copy_bytes(destination: usize, source: &[u8]) -> Result<()> {
        validate_range(destination, source.len())?;
        if !source.is_empty() {
            // SAFETY: the source slice is valid; destination validity is the FFI contract.
            unsafe { ptr::copy(source.as_ptr(), destination as *mut u8, source.len()) };
        }
        Ok(())
    }

    /// Copy a UTF-8 string plus its terminating NUL to native memory.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if the copy length overflows `usize` or the
    /// destination range is null or wraps the address space.
    pub fn copy_c_string(destination: usize, source: &str) -> Result<()> {
        let len = source
            .len()
            .checked_add(1)
            .ok_or_else(|| FfiError("copy length overflow".into()))?;
        validate_range(destination, len)?;
        // SAFETY: the source and destination contract are as above. The final byte fits.
        unsafe {
            ptr::copy(source.as_ptr(), destination as *mut u8, source.len());
            ptr::write((destination as *mut u8).add(source.len()), 0);
        }
        Ok(())
    }

    /// Fill `len` native bytes with the low eight bits of `value`.
    /// The destination must be live and large enough.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if the destination range is null or wraps the
    /// address space.
    pub fn fill(destination: usize, value: u8, len: usize) -> Result<()> {
        validate_range(destination, len)?;
        if len != 0 {
            // SAFETY: destination validity and capacity are the FFI contract.
            unsafe { ptr::write_bytes(destination as *mut u8, value, len) };
        }
        Ok(())
    }

    /// Convert a Lua numeric or pointer value to a supported C scalar.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is not a supported scalar or pointer
    /// type or `value` does not fit it, including a temporary C string cast to
    /// a lasting pointer.
    pub fn cast(ctype: &str, value: &FfiValue) -> Result<FfiValue> {
        cast_to_type(parse_type(ctype, false)?, value)
    }

    /// Allocate a supported scalar C object, initialized with `value`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `ctype` is not a supported scalar or pointer
    /// type or `value` does not fit it.
    pub fn new_scalar(ctype: &str, value: &FfiValue) -> Result<FfiScalar> {
        scalar_of_type(parse_type(ctype, false)?, value)
    }

    /// Allocate a zero-initialized fixed-size scalar array such as `int[1]`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] for a malformed or unsupported array declaration,
    /// a size overflow, or a failed allocation.
    pub fn new_array(cdecl: &str) -> Result<FfiArray> {
        let (ctype, count) = parse_array_declaration(cdecl)?;
        array_of_type(parse_type(ctype, false)?, count)
    }

    /// Allocate an array and initialize consecutive elements from `values`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] as for [`Self::new_array`], if `values` holds more
    /// elements than the array, or if an element value does not fit the
    /// element type.
    pub fn new_array_with(cdecl: &str, values: &[FfiValue]) -> Result<FfiArray> {
        let mut array = Self::new_array(cdecl)?;
        if values.len() > array.len() {
            return Err(FfiError("too many array initializers".into()));
        }
        for (index, value) in values.iter().enumerate() {
            array.set(index, value)?;
        }
        Ok(array)
    }
    /// Read a NUL-terminated native string from a pointer returned by C.
    /// The caller must ensure the pointer is valid and remains live. Invalid
    /// pointers can crash the process, as with `LuaJIT`'s `ffi.string`.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `pointer` is null.
    pub fn string(pointer: usize) -> Result<String> {
        if pointer == 0 {
            return Err(FfiError("null C string pointer".into()));
        }
        // SAFETY: pointer validity is part of the native FFI contract.
        Ok(unsafe { CStr::from_ptr(pointer as *const c_char) }
            .to_string_lossy()
            .into_owned())
    }

    /// Copy `len` bytes from a native pointer without UTF-8 conversion.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if the range is null, wraps the address space, or
    /// exceeds the addressable slice length.
    pub fn string_bytes(pointer: usize, len: usize) -> Result<Vec<u8>> {
        validate_range(pointer, len)?;
        if len > isize::MAX as usize {
            return Err(FfiError(
                "native memory range exceeds addressable slice".into(),
            ));
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: the caller promises that the pointer is live for `len` bytes.
        Ok(unsafe { std::slice::from_raw_parts(pointer as *const u8, len) }.to_vec())
    }

    /// Open a native library by name, trying platform-suffixed fallbacks for
    /// bare names.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if none of the candidate names can be loaded.
    pub fn load(&self, name: &str) -> Result<Library> {
        match Library::open(Some(name)) {
            Ok(library) => Ok(library),
            Err(first_error)
                if !name.contains('/') && !name.contains('\\') && !name.contains('.') =>
            {
                #[cfg(target_os = "macos")]
                let candidates = [format!("lib{name}.dylib")];
                #[cfg(windows)]
                let candidates = [format!("{name}.dll")];
                #[cfg(not(any(target_os = "macos", windows)))]
                let candidates = [format!("lib{name}.so"), format!("lib{name}.so.6")];
                for candidate in candidates {
                    if let Ok(library) = Library::open(Some(&candidate)) {
                        return Ok(library);
                    }
                }
                Err(first_error)
            }
            Err(error) => Err(error),
        }
    }

    /// Call a declared C function from the process namespace.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `name` is not declared, the arguments do not fit
    /// the declaration, or the call cannot be prepared or dispatched.
    pub fn call(&self, name: &str, args: &[FfiValue]) -> Result<FfiValue> {
        self.call_in(&self.process, name, args)
    }

    /// Call a declared C function from an explicitly loaded library.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError`] if `name` is not declared, the arguments do not fit
    /// the declaration, the symbol is missing from `library`, or the call
    /// cannot be prepared or dispatched.
    pub fn call_in(&self, library: &Library, name: &str, args: &[FfiValue]) -> Result<FfiValue> {
        let declaration = self
            .declarations
            .get(name)
            .ok_or_else(|| FfiError(format!("undeclared C symbol: {name}")))?;
        if args.len() < declaration.args.len()
            || (!declaration.variadic && args.len() != declaration.args.len())
        {
            return Err(FfiError(format!(
                "{name} expects {}{} arguments, got {}",
                declaration.args.len(),
                if declaration.variadic { " or more" } else { "" },
                args.len()
            )));
        }
        let symbol = CString::new(name).map_err(|_| FfiError("symbol contains NUL".into()))?;
        // SAFETY: `library.handle` is live and `symbol` is NUL-terminated.
        let function = unsafe { oximond_ffi_symbol(library.handle, symbol.as_ptr()) };
        if function.is_null() {
            return Err(loader_error(&format!("symbol not found: {name}")));
        }

        let mut strings = Vec::new();
        let mut slots = Vec::with_capacity(args.len());
        let mut types = Vec::with_capacity(args.len());
        for (index, value) in args.iter().enumerate() {
            let ty = if let Some(ty) = declaration.args.get(index) {
                *ty
            } else {
                promoted_type(value)?
            };
            types.push(ty.code());
            slots.push(encode_argument(ty, value, &mut strings)?);
        }
        let mut pointers: Vec<*mut c_void> = slots
            .iter_mut()
            .map(|slot| slot.0.as_mut_ptr().cast())
            .collect();
        let mut output = Slot([0; 16]);
        // SAFETY: libffi receives correctly aligned stable argument buffers and
        // a return buffer. The C declaration itself is trusted by contract.
        let status = unsafe {
            oximond_ffi_call(
                function,
                declaration.result.code(),
                types.as_ptr(),
                pointers.as_mut_ptr(),
                args.len(),
                declaration.args.len(),
                output.0.as_mut_ptr().cast(),
            )
        };
        if status == -4 {
            // native.c opens libffi on the first call and found none.
            let library = option_env!("OXIMOND_LIBFFI_SONAME").unwrap_or("libffi");
            return Err(FfiError(format!("cannot load {library} to call {name}")));
        }
        if status != 0 {
            return Err(FfiError(format!(
                "libffi failed to prepare {name} ({status})"
            )));
        }
        decode_result(declaration.result, &output)
    }
}

impl Default for FfiEngine {
    fn default() -> Self {
        Self::new().expect("process namespace must be available")
    }
}

#[repr(align(16))]
struct Slot([u8; 16]);

fn write_slot<T: Copy>(value: T) -> Slot {
    let mut slot = Slot([0; 16]);
    // SAFETY: slot is 16-byte aligned and large enough for every supported scalar.
    unsafe {
        ptr::write(slot.0.as_mut_ptr().cast::<T>(), value);
    }
    slot
}

fn read_slot<T: Copy>(slot: &Slot) -> T {
    // SAFETY: slot is aligned and was filled by libffi for the declared return type.
    unsafe { ptr::read(slot.0.as_ptr().cast::<T>()) }
}

fn encode_argument(ty: FfiType, value: &FfiValue, strings: &mut Vec<CString>) -> Result<Slot> {
    let integer = match value {
        FfiValue::Integer(n) => Some(*n as u64),
        FfiValue::Unsigned(n) => Some(*n),
        _ => None,
    };
    match ty {
        FfiType::Void => Err(FfiError("void is not an argument type".into())),
        FfiType::Char => integer.map(|n| write_slot(n as i8)).ok_or_else(type_error),
        FfiType::UChar => integer.map(|n| write_slot(n as u8)).ok_or_else(type_error),
        FfiType::Short => integer.map(|n| write_slot(n as i16)).ok_or_else(type_error),
        FfiType::UShort => integer.map(|n| write_slot(n as u16)).ok_or_else(type_error),
        FfiType::Int => integer.map(|n| write_slot(n as i32)).ok_or_else(type_error),
        FfiType::UInt => integer.map(|n| write_slot(n as u32)).ok_or_else(type_error),
        FfiType::Long => integer
            .map(|n| write_slot(n as std::ffi::c_long))
            .ok_or_else(type_error),
        FfiType::ULong => integer
            .map(|n| write_slot(n as std::ffi::c_ulong))
            .ok_or_else(type_error),
        FfiType::SizeT => integer
            .map(|n| write_slot(n as usize))
            .ok_or_else(type_error),
        FfiType::LongLong => integer.map(|n| write_slot(n as i64)).ok_or_else(type_error),
        FfiType::ULongLong => integer.map(write_slot).ok_or_else(type_error),
        FfiType::Float => match value {
            FfiValue::Float(n) => Ok(write_slot(*n)),
            FfiValue::Double(n) => Ok(write_slot(*n as f32)),
            _ => Err(type_error()),
        },
        FfiType::Double => match value {
            FfiValue::Float(n) => Ok(write_slot(f64::from(*n))),
            FfiValue::Double(n) => Ok(write_slot(*n)),
            _ => Err(type_error()),
        },
        FfiType::Pointer | FfiType::CString => match value {
            FfiValue::Pointer(n) => Ok(write_slot(*n)),
            FfiValue::Integer(n) => Ok(write_slot(*n as usize)),
            FfiValue::Unsigned(n) => Ok(write_slot(*n as usize)),
            FfiValue::CString(s) => {
                let string = CString::new(s.as_str())
                    .map_err(|_| FfiError("C string contains NUL".into()))?;
                let pointer = string.as_ptr() as usize;
                strings.push(string);
                Ok(write_slot(pointer))
            }
            _ => Err(type_error()),
        },
    }
}

fn type_error() -> FfiError {
    FfiError("argument does not match C declaration".into())
}

fn promoted_type(value: &FfiValue) -> Result<FfiType> {
    match value {
        FfiValue::Integer(_) => Ok(FfiType::LongLong),
        FfiValue::Unsigned(_) => Ok(FfiType::ULongLong),
        FfiValue::Float(_) | FfiValue::Double(_) => Ok(FfiType::Double),
        FfiValue::Pointer(_) => Ok(FfiType::Pointer),
        FfiValue::CString(_) => Ok(FfiType::CString),
        FfiValue::Void => Err(FfiError(
            "void cannot be passed as a variadic argument".into(),
        )),
    }
}

// c_long is narrower than i64 on some targets, even though they match here.
#[allow(clippy::useless_conversion)]
fn decode_result(ty: FfiType, slot: &Slot) -> Result<FfiValue> {
    Ok(match ty {
        FfiType::Void => FfiValue::Void,
        FfiType::Char => FfiValue::Integer(i64::from(read_slot::<i8>(slot))),
        FfiType::UChar => FfiValue::Unsigned(u64::from(read_slot::<u8>(slot))),
        FfiType::Short => FfiValue::Integer(i64::from(read_slot::<i16>(slot))),
        FfiType::UShort => FfiValue::Unsigned(u64::from(read_slot::<u16>(slot))),
        FfiType::Int => FfiValue::Integer(i64::from(read_slot::<i32>(slot))),
        FfiType::UInt => FfiValue::Unsigned(u64::from(read_slot::<u32>(slot))),
        FfiType::Long => FfiValue::Integer(i64::from(read_slot::<std::ffi::c_long>(slot))),
        FfiType::ULong => FfiValue::Unsigned(u64::from(read_slot::<std::ffi::c_ulong>(slot))),
        FfiType::SizeT => FfiValue::Unsigned(read_slot::<usize>(slot) as u64),
        FfiType::LongLong => FfiValue::Integer(read_slot::<i64>(slot)),
        FfiType::ULongLong => FfiValue::Unsigned(read_slot::<u64>(slot)),
        FfiType::Float => FfiValue::Float(read_slot::<f32>(slot)),
        FfiType::Double => FfiValue::Double(read_slot::<f64>(slot)),
        FfiType::Pointer | FfiType::CString => FfiValue::Pointer(read_slot::<usize>(slot)),
    })
}

fn loader_error(default: &str) -> FfiError {
    // SAFETY: dlerror returns either null or a live NUL-terminated error string.
    let message = unsafe { oximond_ffi_error() };
    if message.is_null() {
        FfiError(default.into())
    } else {
        FfiError(
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

fn validate_range(pointer: usize, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if pointer == 0 || pointer.checked_add(len - 1).is_none() {
        return Err(FfiError("invalid native memory range".into()));
    }
    Ok(())
}

fn strip_c_comments(input: &str) -> Result<String> {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '/' && chars.peek() == Some(&'/') {
            chars.next();
            for next in chars.by_ref() {
                if next == '\n' {
                    output.push('\n');
                    break;
                }
            }
        } else if ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut closed = false;
            while let Some(next) = chars.next() {
                if next == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    closed = true;
                    break;
                }
            }
            if !closed {
                return Err(FfiError("unterminated C block comment".into()));
            }
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    Ok(output)
}

fn size_of_type(ty: FfiType) -> Result<usize> {
    Ok(match ty {
        FfiType::Void => return Err(FfiError("void has no size".into())),
        FfiType::Char | FfiType::UChar => 1,
        FfiType::Short | FfiType::UShort => 2,
        FfiType::Int | FfiType::UInt | FfiType::Float => 4,
        FfiType::Long | FfiType::ULong => std::mem::size_of::<std::ffi::c_long>(),
        FfiType::LongLong | FfiType::ULongLong | FfiType::Double => 8,
        FfiType::SizeT | FfiType::Pointer | FfiType::CString => std::mem::size_of::<usize>(),
    })
}

fn align_of_type(ty: FfiType) -> Result<usize> {
    Ok(match ty {
        FfiType::Void => return Err(FfiError("void has no alignment".into())),
        FfiType::Char | FfiType::UChar => std::mem::align_of::<u8>(),
        FfiType::Short | FfiType::UShort => std::mem::align_of::<u16>(),
        FfiType::Int | FfiType::UInt => std::mem::align_of::<u32>(),
        FfiType::Long | FfiType::ULong => std::mem::align_of::<std::ffi::c_long>(),
        FfiType::LongLong | FfiType::ULongLong => std::mem::align_of::<u64>(),
        FfiType::SizeT | FfiType::Pointer | FfiType::CString => std::mem::align_of::<usize>(),
        FfiType::Float => std::mem::align_of::<f32>(),
        FfiType::Double => std::mem::align_of::<f64>(),
    })
}

fn cast_to_type(ty: FfiType, value: &FfiValue) -> Result<FfiValue> {
    if matches!(ty, FfiType::Pointer | FfiType::CString) && matches!(value, FfiValue::CString(_)) {
        return Err(FfiError(
            "cannot cast a temporary string to a lasting pointer".into(),
        ));
    }
    let mut strings = Vec::new();
    let slot = encode_argument(ty, value, &mut strings)?;
    decode_result(ty, &slot)
}

fn scalar_of_type(ty: FfiType, value: &FfiValue) -> Result<FfiScalar> {
    let mut strings = Vec::new();
    let storage = Box::new(encode_argument(ty, value, &mut strings)?);
    Ok(FfiScalar {
        ty,
        storage,
        _strings: strings,
    })
}

fn array_of_type(ty: FfiType, count: usize) -> Result<FfiArray> {
    let element_size = size_of_type(ty)?;
    let alignment = align_of_type(ty)?;
    let size = element_size
        .checked_mul(count)
        .ok_or_else(|| FfiError("array size overflow".into()))?;
    let layout = Layout::from_size_align(size.max(1), alignment)
        .map_err(|_| FfiError("invalid array layout".into()))?;
    // SAFETY: layout has nonzero size and valid alignment.
    let memory = NonNull::new(unsafe { alloc_zeroed(layout) })
        .ok_or_else(|| FfiError("native array allocation failed".into()))?;
    Ok(FfiArray {
        ty,
        len: count,
        element_size,
        layout,
        memory,
        strings: vec![None; count],
    })
}

fn parse_array_declaration(input: &str) -> Result<(&str, usize)> {
    let input = input.trim();
    let open = input
        .rfind('[')
        .ok_or_else(|| FfiError(format!("invalid array declaration: {input}")))?;
    let count = input[open + 1..]
        .strip_suffix(']')
        .ok_or_else(|| FfiError(format!("invalid array declaration: {input}")))?
        .trim()
        .parse::<usize>()
        .map_err(|_| FfiError(format!("invalid array length: {input}")))?;
    let ctype = input[..open].trim();
    if ctype.is_empty() || ctype.contains('[') {
        return Err(FfiError(format!("invalid array element type: {input}")));
    }
    Ok((ctype, count))
}

fn parse_declaration(
    input: &str,
    aliases: &HashMap<String, FfiType>,
) -> Result<(String, FunctionDecl)> {
    let open = input
        .find('(')
        .ok_or_else(|| FfiError(format!("unsupported C declaration: {input}")))?;
    let close = input
        .rfind(')')
        .ok_or_else(|| FfiError(format!("unsupported C declaration: {input}")))?;
    if close < open || !input[close + 1..].trim().is_empty() {
        return Err(FfiError(format!("unsupported C declaration: {input}")));
    }
    let head = input[..open].trim();
    let end = head
        .trim_end()
        .trim_end_matches(|c: char| c.is_alphanumeric() || c == '_')
        .len();
    let name = head[end..].trim();
    if !is_identifier(name) {
        return Err(FfiError(format!("invalid C function name: {input}")));
    }
    let result = parse_type_with_aliases(head[..end].trim(), false, aliases)?;
    let body = input[open + 1..close].trim();
    let mut args = Vec::new();
    let mut variadic = false;
    if !body.is_empty() && body != "void" {
        let parts: Vec<_> = body.split(',').map(str::trim).collect();
        for (index, part) in parts.iter().enumerate() {
            if *part == "..." {
                if index != parts.len() - 1 || args.is_empty() {
                    return Err(FfiError(
                        "variadic marker must follow fixed arguments".into(),
                    ));
                }
                variadic = true;
            } else {
                args.push(parse_type_with_aliases(part, true, aliases)?);
            }
        }
    }
    Ok((
        name.to_owned(),
        FunctionDecl {
            result,
            args,
            variadic,
        },
    ))
}

fn parse_type(input: &str, argument: bool) -> Result<FfiType> {
    parse_type_with_aliases(input, argument, &HashMap::new())
}

fn parse_typedef(input: &str, aliases: &HashMap<String, FfiType>) -> Result<(String, FfiType)> {
    let end = input
        .trim_end_matches(|c: char| c.is_ascii_alphanumeric() || c == '_')
        .len();
    let name = input[end..].trim();
    if !is_identifier(name) || input[..end].trim().is_empty() {
        return Err(FfiError(format!("unsupported C typedef: {input}")));
    }
    Ok((
        name.to_owned(),
        parse_type_with_aliases(input[..end].trim(), false, aliases)?,
    ))
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_type_with_aliases(
    input: &str,
    argument: bool,
    aliases: &HashMap<String, FfiType>,
) -> Result<FfiType> {
    let normalized = input.replace('*', " * ");
    let mut words: Vec<_> = normalized
        .split_whitespace()
        .filter(|word| !matches!(*word, "const" | "volatile" | "restrict"))
        .collect();
    if words.is_empty()
        || words
            .iter()
            .any(|word| word.contains(['(', ')', '[', ']', '{', '}']))
    {
        return Err(FfiError(format!("unsupported C type: {input}")));
    }
    if argument && words.len() > 1 && is_identifier(words[words.len() - 1]) {
        let candidate = words[..words.len() - 1]
            .iter()
            .filter(|word| **word != "*")
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        if builtin_type(&candidate).is_some() || aliases.contains_key(&candidate) {
            words.pop();
        }
    }
    let stars = words.iter().filter(|word| **word == "*").count();
    let base_words: Vec<_> = words.into_iter().filter(|word| *word != "*").collect();
    let base = base_words.join(" ");
    let ty = builtin_type(&base)
        .or_else(|| aliases.get(&base).copied())
        .ok_or_else(|| FfiError(format!("unsupported C type: {input}")))?;
    let ty = if stars == 0 {
        ty
    } else if stars == 1 && ty == FfiType::Char {
        FfiType::CString
    } else {
        FfiType::Pointer
    };
    if argument && ty == FfiType::Void {
        return Err(FfiError("void is not an argument type".into()));
    }
    Ok(ty)
}

fn builtin_type(input: &str) -> Option<FfiType> {
    Some(match input {
        "void" => FfiType::Void,
        "char" | "signed char" | "int8_t" => FfiType::Char,
        "unsigned char" | "uint8_t" => FfiType::UChar,
        "short" | "short int" | "signed short" | "int16_t" => FfiType::Short,
        "unsigned short" | "unsigned short int" | "uint16_t" => FfiType::UShort,
        "int" | "signed" | "signed int" | "int32_t" => FfiType::Int,
        "unsigned" | "unsigned int" | "uint32_t" => FfiType::UInt,
        "long" | "long int" | "signed long" | "signed long int" => FfiType::Long,
        "unsigned long" | "unsigned long int" => FfiType::ULong,
        "long long" | "long long int" | "signed long long" | "signed long long int" | "int64_t" => {
            FfiType::LongLong
        }
        "unsigned long long" | "unsigned long long int" | "uint64_t" => FfiType::ULongLong,
        "size_t" => FfiType::SizeT,
        "float" => FfiType::Float,
        "double" => FfiType::Double,
        _ => return None,
    })
}

fn is_known_type(input: &str) -> bool {
    matches!(
        input,
        "void"
            | "char"
            | "signed char"
            | "unsigned char"
            | "short"
            | "short int"
            | "signed short"
            | "unsigned short"
            | "unsigned short int"
            | "int8_t"
            | "uint8_t"
            | "int16_t"
            | "uint16_t"
            | "int32_t"
            | "uint32_t"
            | "int64_t"
            | "uint64_t"
            | "int"
            | "signed"
            | "signed int"
            | "unsigned"
            | "unsigned int"
            | "long"
            | "long int"
            | "signed long"
            | "signed long int"
            | "unsigned long"
            | "unsigned long int"
            | "long long"
            | "long long int"
            | "signed long long"
            | "signed long long int"
            | "unsigned long long"
            | "unsigned long long int"
            | "size_t"
            | "float"
            | "double"
    )
}

#[cfg(all(
    test,
    any(
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly",
        target_os = "macos"
    )
))]
mod tests {
    use super::{FfiEngine, FfiType, FfiValue};

    #[test]
    fn calls_real_process_symbols() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("int abs(int value); size_t strlen(const char *text);")
            .unwrap();
        assert_eq!(
            ffi.call("abs", &[FfiValue::Integer(-42)]).unwrap(),
            FfiValue::Integer(42)
        );
        assert_eq!(
            ffi.call("strlen", &[FfiValue::CString("hello".into())])
                .unwrap(),
            FfiValue::Unsigned(5)
        );
        assert_eq!(
            ffi.declaration("strlen").unwrap().args,
            vec![FfiType::CString]
        );
    }

    /// `cbrt` is in libm only, which a binary may not link (the CLI loads it on
    /// demand), so a process lookup has to reach it anyway.
    #[test]
    fn process_symbols_include_libm() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("double cbrt(double value);").unwrap();
        let FfiValue::Double(root) = ffi.call("cbrt", &[FfiValue::Double(27.0)]).unwrap() else {
            panic!("cbrt returns a double");
        };
        assert!((root - 3.0).abs() < 1e-12, "{root}");
    }

    #[test]
    fn typedefs_resolve_in_signatures_and_instance_type_operations() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("// native integer aliases\n typedef int number_t; /* pointer alias */ typedef const char * text_t; typedef number_t another_t; another_t abs(const another_t value); size_t strlen(text_t text);")
            .unwrap();
        assert_eq!(ffi.type_of("number_t").unwrap(), FfiType::Int);
        assert_eq!(ffi.type_of("const text_t").unwrap(), FfiType::CString);
        assert_eq!(ffi.sizeof_type("number_t[3]").unwrap(), 12);
        assert_eq!(
            ffi.alignof_type("number_t").unwrap(),
            std::mem::align_of::<i32>()
        );
        assert_eq!(
            ffi.cast_type("number_t", &FfiValue::Integer(7)).unwrap(),
            FfiValue::Integer(7)
        );
        let scalar = ffi
            .new_scalar_type("number_t", &FfiValue::Integer(-2))
            .unwrap();
        assert_eq!(scalar.value().unwrap(), FfiValue::Integer(-2));
        let array = ffi.new_array_type("number_t[2]").unwrap();
        assert_eq!(array.len(), 2);
        assert_eq!(
            ffi.call("abs", &[FfiValue::Integer(-17)]).unwrap(),
            FfiValue::Integer(17)
        );
        assert_eq!(
            ffi.call("strlen", &[FfiValue::CString("abc".into())])
                .unwrap(),
            FfiValue::Unsigned(3)
        );
    }

    #[test]
    fn repeated_compatible_typedefs_are_accepted_but_conflicts_fail() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("typedef unsigned char char_u;").unwrap();
        ffi.cdef("typedef unsigned char char_u;").unwrap();
        ffi.cdef("typedef uint8_t char_u; typedef char_u byte_t; typedef unsigned char byte_t;")
            .unwrap();
        assert_eq!(ffi.type_of("char_u").unwrap(), FfiType::UChar);
        assert_eq!(ffi.type_of("byte_t").unwrap(), FfiType::UChar);

        assert_eq!(
            ffi.cdef("typedef int char_u;").unwrap_err().0,
            "duplicate C type name: char_u"
        );
        assert_eq!(ffi.type_of("char_u").unwrap(), FfiType::UChar);
    }

    #[test]
    fn unsupported_declarations_fail_without_partial_registration() {
        let mut ffi = FfiEngine::new().unwrap();
        assert!(
            ffi.cdef("typedef int good_t; int broken(unknown_type *);")
                .is_err()
        );
        assert!(ffi.type_of("good_t").is_err());
        assert!(ffi.declaration("broken").is_none());
        assert!(
            ffi.cdef("struct Entry { int value; };")
                .unwrap_err()
                .0
                .contains("struct")
        );
        assert!(ffi.cdef("int broken(int (*callback)(int));").is_err());
        assert!(ffi.cdef("typedef int x; /* unfinished").is_err());
    }

    #[test]
    fn char_pointer_result_stays_a_pointer_until_explicitly_read() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("char *strchr(const char *s, int c);").unwrap();
        let result = ffi
            .call(
                "strchr",
                &[FfiValue::CString("abc".into()), FfiValue::Integer(98)],
            )
            .unwrap();
        assert!(matches!(result, FfiValue::Pointer(_)));
        // The call's temporary input string is gone, so do not dereference its pointer.
        assert_eq!(FfiEngine::string_bytes(0, 0).unwrap(), Vec::<u8>::new());
        assert!(FfiEngine::string_bytes(0, 1).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn calls_explicit_library() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("int abs(int);").unwrap();
        let libc = ffi.load("libc.so.6").unwrap();
        assert_eq!(
            ffi.call_in(&libc, "abs", &[FfiValue::Integer(-7)]).unwrap(),
            FfiValue::Integer(7)
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn loads_library_by_short_name() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("double cos(double);").unwrap();
        let math = ffi.load("m").unwrap();
        assert_eq!(
            ffi.call_in(&math, "cos", &[FfiValue::Double(0.0)]).unwrap(),
            FfiValue::Double(1.0)
        );
    }

    #[test]
    fn calls_variadic_function_with_scalar_storage() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("int sscanf(const char *input, const char *format, ...);")
            .unwrap();
        let number = FfiEngine::new_scalar("int", &FfiValue::Integer(0)).unwrap();
        assert_eq!(
            ffi.call(
                "sscanf",
                &[
                    FfiValue::CString("123".into()),
                    FfiValue::CString("%d".into()),
                    FfiValue::Pointer(number.pointer()),
                ],
            )
            .unwrap(),
            FfiValue::Integer(1)
        );
        assert_eq!(number.value().unwrap(), FfiValue::Integer(123));
        assert_eq!(FfiEngine::sizeof("int").unwrap(), 4);
        assert_eq!(
            FfiEngine::cast("unsigned char", &FfiValue::Integer(257)).unwrap(),
            FfiValue::Unsigned(1)
        );
    }

    #[test]
    fn native_memory_and_errno_helpers() {
        let mut bytes = [0_u8; 8];
        let pointer = bytes.as_mut_ptr() as usize;
        FfiEngine::fill(pointer, 0x41, bytes.len()).unwrap();
        assert_eq!(&bytes, b"AAAAAAAA");
        FfiEngine::copy_bytes(pointer, b"xy").unwrap();
        FfiEngine::copy(pointer + 2, pointer, 2).unwrap();
        assert_eq!(&bytes, b"xyxyAAAA");
        FfiEngine::copy_c_string(pointer, "cat").unwrap();
        assert_eq!(&bytes[..4], b"cat\0");
        assert_eq!(FfiEngine::string(pointer).unwrap(), "cat");
        assert_eq!(FfiEngine::string_bytes(pointer, 4).unwrap(), b"cat\0");
        assert!(FfiEngine::copy(0, pointer, 1).is_err());
        assert!(FfiEngine::fill(usize::MAX, 0, 2).is_err());

        let previous = FfiEngine::errno();
        FfiEngine::set_errno(123);
        assert_eq!(FfiEngine::errno(), 123);
        FfiEngine::set_errno(previous);
    }

    #[test]
    fn alignment_and_abi_match_native_target() {
        assert_eq!(
            FfiEngine::alignof("int").unwrap(),
            std::mem::align_of::<i32>()
        );
        assert_eq!(
            FfiEngine::alignof("void *").unwrap(),
            std::mem::align_of::<usize>()
        );
        assert_eq!(FfiEngine::abi("64bit"), cfg!(target_pointer_width = "64"));
        assert_eq!(FfiEngine::abi("le"), cfg!(target_endian = "little"));
        assert!(!FfiEngine::abi("unknown"));
    }

    #[test]
    fn libc_writes_into_aligned_scalar_array() {
        let mut ffi = FfiEngine::new().unwrap();
        ffi.cdef("int sscanf(const char *input, const char *format, ...);")
            .unwrap();
        let mut numbers = FfiEngine::new_array("int[1]").unwrap();
        assert_eq!(numbers.len(), 1);
        assert_eq!(FfiEngine::sizeof("int[1]").unwrap(), 4);
        assert_eq!(numbers.pointer() % FfiEngine::alignof("int").unwrap(), 0);
        assert_eq!(numbers.get(0).unwrap(), FfiValue::Integer(0));
        assert_eq!(
            ffi.call(
                "sscanf",
                &[
                    FfiValue::CString("318".into()),
                    FfiValue::CString("%d".into()),
                    FfiValue::Pointer(numbers.pointer()),
                ]
            )
            .unwrap(),
            FfiValue::Integer(1)
        );
        assert_eq!(numbers.get(0).unwrap(), FfiValue::Integer(318));
        numbers.set(0, &FfiValue::Integer(-4)).unwrap();
        assert_eq!(numbers.get(0).unwrap(), FfiValue::Integer(-4));
        assert!(numbers.get(1).is_err());
        assert!(numbers.set(1, &FfiValue::Integer(1)).is_err());
    }

    #[test]
    fn byte_array_is_contiguous_and_can_be_initialized() {
        let mut bytes =
            FfiEngine::new_array_with("char[4]", &[FfiValue::Integer(65), FfiValue::Integer(66)])
                .unwrap();
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes.get(0).unwrap(), FfiValue::Integer(65));
        assert_eq!(bytes.get(2).unwrap(), FfiValue::Integer(0));
        FfiEngine::copy_c_string(bytes.pointer(), "cat").unwrap();
        assert_eq!(FfiEngine::string(bytes.pointer()).unwrap(), "cat");
        bytes.set(0, &FfiValue::Integer(90)).unwrap();
        assert_eq!(bytes.get(0).unwrap(), FfiValue::Integer(90));
        assert!(FfiEngine::new_array("void[2]").is_err());
        assert!(FfiEngine::new_array("int[-1]").is_err());
    }
}
