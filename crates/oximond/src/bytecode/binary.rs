//! Stable, Rust-owned Oximond bytecode serialization.
//!
//! The format stores only portable compiler data. It deliberately excludes VM
//! cells, Rust pointers, handles, and live Lua values.

use core::fmt;
use std::sync::Arc;

use crate::string::LuaString;

use crate::bytecode::{
    CallKind, CallSite, Chunk, Constant, DebugMetadata, DiagnosticSite, Instruction, InvalidOpCode,
    LocalVariable, UpvalueDescriptor, UpvalueSource, ValidationError,
};

const MAGIC: &[u8; 8] = b"\x1bOXIMOND";
const VERSION: u16 = 13;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_DEPTH: usize = 128;
const MAX_PROTOTYPES: usize = 65_536;
// Lua 5.1's `verybig.lua` compiles to 1,065,076 instructions. Code and its
// line table are therefore allowed to exceed the general collection limit.
// The total still caps all allocations decoded from a single bytecode file.
const MAX_TOTAL_ELEMENTS: usize = 4_000_000;
const MAX_COLLECTION_ELEMENTS: usize = 1_000_000;
const MAX_CODE_OR_LINE_ELEMENTS: usize = 2_000_000;
const MAX_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOTAL_STRING_BYTES: usize = 32 * 1024 * 1024;

const FLAG_VARARG: u8 = 1;
const FLAG_ARG_REGISTER: u8 = 2;
const CALL_SITE_NAME: u8 = 1;
const CALL_SITE_KIND: u8 = 2;
const CALL_SITE_ITERATOR: u8 = 4;

/// Returns whether `bytes` begin with the Oximond binary-bytecode magic.
#[must_use]
pub fn claims(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

/// Serializes a chunk using the portable Oximond binary format.
///
/// Root upvalue sources describe an in-memory parent frame and cannot survive
/// dumping. They are therefore written as [`UpvalueSource::Detached`]. Nested
/// capture descriptors retain their original source.
///
/// # Errors
///
/// Returns an error when the chunk is malformed or exceeds the format limit.
pub fn dump(chunk: &Chunk) -> Result<Vec<u8>, BinaryError> {
    let mut portable = chunk.clone();
    for upvalue in &mut portable.upvalues {
        upvalue.source = UpvalueSource::Detached;
    }
    portable.validate().map_err(BinaryError::InvalidChunk)?;

    let mut encoder = Encoder::default();
    encoder.write(MAGIC)?;
    encoder.u16(VERSION)?;
    encoder.u16(0)?;
    encoder.chunk(&portable, 0)?;
    Ok(encoder.bytes)
}

/// Decodes a portable Oximond binary chunk.
///
/// # Errors
///
/// Returns an error for malformed, oversized, non-portable, or invalid data.
pub fn load(bytes: &[u8]) -> Result<Chunk, BinaryError> {
    if bytes.len() > MAX_BYTES {
        return Err(BinaryError::InputTooLarge {
            length: bytes.len(),
        });
    }
    let mut decoder = Decoder::new(bytes);
    if decoder.take(MAGIC.len())? != MAGIC {
        return Err(BinaryError::InvalidMagic);
    }
    let version = decoder.u16()?;
    if version != VERSION {
        return Err(BinaryError::UnsupportedVersion { version });
    }
    let flags = decoder.u16()?;
    if flags != 0 {
        return Err(BinaryError::UnknownHeaderFlags { flags });
    }
    let chunk = decoder.chunk(0)?;
    if decoder.remaining() != 0 {
        return Err(BinaryError::TrailingData {
            length: decoder.remaining(),
        });
    }
    chunk.validate().map_err(BinaryError::InvalidChunk)?;
    // The live register analysis runs on a function's first call, as for
    // compiled source: an embedder such as Luanti loads thousands of dumped
    // callbacks that never run in the loading state.
    Ok(chunk)
}

/// A binary bytecode encoding or decoding failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BinaryError {
    InputTooLarge { length: usize },
    OutputTooLarge,
    InvalidMagic,
    UnsupportedVersion { version: u16 },
    UnknownHeaderFlags { flags: u16 },
    UnexpectedEof,
    TrailingData { length: usize },
    LengthLimit { kind: &'static str, length: usize },
    StringBudgetExceeded { length: usize },
    DepthLimit { depth: usize },
    PrototypeLimit { count: usize },
    UnknownConstantTag { tag: u8 },
    MalformedBoolean { value: u8 },
    UnknownOpcode { opcode: u8 },
    UnknownUpvalueSource { kind: u8 },
    UnknownCallSiteFlags { flags: u8 },
    MalformedCallSite,
    UnknownCallKind { kind: u8 },
    InvalidChunk(ValidationError),
}

impl fmt::Display for BinaryError {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLarge { length } => {
                write!(
                    formatter,
                    "binary input of {length} bytes exceeds the limit"
                )
            }
            Self::OutputTooLarge => write!(formatter, "binary output exceeds the 64 MiB limit"),
            Self::InvalidMagic => write!(formatter, "input is not Oximond binary bytecode"),
            Self::UnsupportedVersion { version } => {
                write!(
                    formatter,
                    "unsupported Oximond binary bytecode version {version}"
                )
            }
            Self::UnknownHeaderFlags { flags } => {
                write!(formatter, "unknown Oximond binary header flags {flags:#x}")
            }
            Self::UnexpectedEof => write!(formatter, "unexpected end of binary bytecode"),
            Self::TrailingData { length } => {
                write!(formatter, "binary bytecode has {length} trailing bytes")
            }
            Self::LengthLimit { kind, length } => {
                write!(formatter, "binary {kind} length {length} exceeds the limit")
            }
            Self::StringBudgetExceeded { length } => {
                write!(
                    formatter,
                    "binary strings use {length} bytes, beyond the limit"
                )
            }
            Self::DepthLimit { depth } => write!(
                formatter,
                "binary prototype depth {depth} exceeds the limit"
            ),
            Self::PrototypeLimit { count } => {
                write!(formatter, "binary has {count} prototypes, beyond the limit")
            }
            Self::UnknownConstantTag { tag } => {
                write!(formatter, "unknown binary constant tag {tag}")
            }
            Self::MalformedBoolean { value } => {
                write!(formatter, "malformed binary boolean value {value}")
            }
            Self::UnknownOpcode { opcode } => write!(formatter, "unknown binary opcode {opcode}"),
            Self::UnknownUpvalueSource { kind } => {
                write!(formatter, "unknown binary upvalue source {kind}")
            }
            Self::UnknownCallSiteFlags { flags } => {
                write!(formatter, "unknown binary call-site flags {flags:#x}")
            }
            Self::MalformedCallSite => {
                write!(formatter, "binary call site has only a name or kind")
            }
            Self::UnknownCallKind { kind } => write!(formatter, "unknown binary call kind {kind}"),
            Self::InvalidChunk(error) => write!(formatter, "invalid binary chunk: {error}"),
        }
    }
}

impl std::error::Error for BinaryError {}

#[derive(Default)]
struct Encoder {
    bytes: Vec<u8>,
    elements: usize,
    string_bytes: usize,
    prototypes: usize,
}

impl Encoder {
    fn write(&mut self, bytes: &[u8]) -> Result<(), BinaryError> {
        if self.bytes.len().saturating_add(bytes.len()) > MAX_BYTES {
            return Err(BinaryError::OutputTooLarge);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), BinaryError> {
        self.write(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), BinaryError> {
        self.write(&value.to_le_bytes())
    }

    fn u32(&mut self, value: u32) -> Result<(), BinaryError> {
        self.write(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), BinaryError> {
        self.write(&value.to_le_bytes())
    }

    fn count(
        &mut self,
        kind: &'static str,
        length: usize,
        collection_limit: usize,
    ) -> Result<(), BinaryError> {
        let length =
            u32::try_from(length).map_err(|_| BinaryError::LengthLimit { kind, length })?;
        let length = usize::try_from(length).expect("u32 fits usize");
        if length > collection_limit {
            return Err(BinaryError::LengthLimit { kind, length });
        }
        self.elements = self.elements.saturating_add(length);
        if self.elements > MAX_TOTAL_ELEMENTS {
            return Err(BinaryError::LengthLimit {
                kind: "total elements",
                length: self.elements,
            });
        }
        self.u32(u32::try_from(length).expect("checked above"))
    }

    fn string(&mut self, value: &LuaString) -> Result<(), BinaryError> {
        if value.len() > MAX_STRING_BYTES {
            return Err(BinaryError::LengthLimit {
                kind: "string",
                length: value.len(),
            });
        }
        self.string_bytes = self.string_bytes.saturating_add(value.len());
        if self.string_bytes > MAX_TOTAL_STRING_BYTES {
            return Err(BinaryError::StringBudgetExceeded {
                length: self.string_bytes,
            });
        }
        let length = u32::try_from(value.len()).map_err(|_| BinaryError::LengthLimit {
            kind: "string",
            length: value.len(),
        })?;
        self.u32(length)?;
        self.write(value.as_bytes())
    }

    #[allow(clippy::too_many_lines)]
    fn chunk(&mut self, chunk: &Chunk, depth: usize) -> Result<(), BinaryError> {
        if depth > MAX_DEPTH {
            return Err(BinaryError::DepthLimit { depth });
        }
        self.u8(chunk.register_count)?;
        self.u8(chunk.parameter_count)?;
        let flags = u8::from(chunk.is_vararg)
            | if chunk.arg_register.is_some() {
                FLAG_ARG_REGISTER
            } else {
                0
            };
        self.u8(flags)?;
        if let Some(register) = chunk.arg_register {
            self.u8(register)?;
        }
        self.string(chunk.debug.source_name())?;
        self.u32(chunk.debug.definition_line())?;
        self.u32(chunk.debug.last_definition_line())?;

        self.count("constants", chunk.constants.len(), MAX_COLLECTION_ELEMENTS)?;
        for constant in &chunk.constants {
            match constant {
                Constant::Nil => self.u8(0)?,
                Constant::Boolean(value) => {
                    self.u8(1)?;
                    self.u8(u8::from(*value))?;
                }
                Constant::Integer(value) => {
                    self.u8(2)?;
                    self.write(&value.to_le_bytes())?;
                }
                Constant::Number(value) => {
                    self.u8(3)?;
                    self.u64(value.to_bits())?;
                }
                Constant::String(value) => {
                    self.u8(4)?;
                    self.string(value)?;
                }
                Constant::Int64(value) => {
                    self.u8(5)?;
                    self.write(&value.to_le_bytes())?;
                }
                Constant::UInt64(value) => {
                    self.u8(6)?;
                    self.u64(*value)?;
                }
            }
        }

        self.count("instructions", chunk.code.len(), MAX_CODE_OR_LINE_ELEMENTS)?;
        for instruction in &chunk.code {
            self.u32(instruction.raw())?;
        }

        self.count(
            "debug line entries",
            chunk.debug.line_info().len(),
            MAX_CODE_OR_LINE_ELEMENTS,
        )?;
        for line in chunk.debug.line_info() {
            self.u32(*line)?;
        }

        self.count(
            "debug local variables",
            chunk.debug.local_variables().len(),
            MAX_COLLECTION_ELEMENTS,
        )?;
        for local in chunk.debug.local_variables() {
            self.string(local.name())?;
            self.u8(local.register())?;
            let start_pc =
                u32::try_from(local.start_pc()).map_err(|_| BinaryError::LengthLimit {
                    kind: "debug-local start pc",
                    length: local.start_pc(),
                })?;
            let end_pc = u32::try_from(local.end_pc()).map_err(|_| BinaryError::LengthLimit {
                kind: "debug-local end pc",
                length: local.end_pc(),
            })?;
            self.u32(start_pc)?;
            self.u32(end_pc)?;
        }

        self.count("upvalues", chunk.upvalues.len(), MAX_COLLECTION_ELEMENTS)?;
        for upvalue in &chunk.upvalues {
            self.string(&upvalue.name)?;
            match upvalue.source {
                UpvalueSource::ParentRegister(register) => {
                    self.u8(0)?;
                    self.u8(register)?;
                }
                UpvalueSource::ParentUpvalue(index) => {
                    self.u8(1)?;
                    self.u16(index)?;
                }
                UpvalueSource::Detached => self.u8(2)?,
            }
        }

        self.count(
            "call sites",
            chunk.debug.call_sites().len(),
            MAX_COLLECTION_ELEMENTS,
        )?;
        for call_site in chunk.debug.call_sites() {
            let pc = u32::try_from(call_site.pc()).map_err(|_| BinaryError::LengthLimit {
                kind: "call-site pc",
                length: call_site.pc(),
            })?;
            self.u32(pc)?;
            let flags = u8::from(call_site.name().is_some())
                | if call_site.kind().is_some() {
                    CALL_SITE_KIND
                } else {
                    0
                }
                | if call_site.is_iterator() {
                    CALL_SITE_ITERATOR
                } else {
                    0
                };
            self.u8(flags)?;
            if let Some(name) = call_site.name() {
                self.string(name)?;
            }
            if let Some(kind) = call_site.kind() {
                self.u8(call_kind_tag(kind))?;
            }
        }

        self.count(
            "diagnostic sites",
            chunk.debug.diagnostic_sites().len(),
            MAX_COLLECTION_ELEMENTS,
        )?;
        for site in chunk.debug.diagnostic_sites() {
            let pc = u32::try_from(site.pc()).map_err(|_| BinaryError::LengthLimit {
                kind: "diagnostic-site pc",
                length: site.pc(),
            })?;
            self.u32(pc)?;
            self.u8(site.register())?;
            self.string(site.name())?;
            self.u8(call_kind_tag(site.kind()))?;
        }

        self.count(
            "prototypes",
            chunk.prototypes.len(),
            MAX_COLLECTION_ELEMENTS,
        )?;
        self.prototypes = self.prototypes.saturating_add(chunk.prototypes.len());
        if self.prototypes > MAX_PROTOTYPES {
            return Err(BinaryError::PrototypeLimit {
                count: self.prototypes,
            });
        }
        for prototype in &chunk.prototypes {
            self.chunk(prototype, depth + 1)?;
        }
        Ok(())
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    position: usize,
    elements: usize,
    string_bytes: usize,
    prototypes: usize,
}

impl<'a> Decoder<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            elements: 0,
            string_bytes: 0,
            prototypes: 0,
        }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], BinaryError> {
        let end = self
            .position
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(BinaryError::UnexpectedEof)?;
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, BinaryError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, BinaryError> {
        let bytes: [u8; 2] = self.take(2)?.try_into().expect("fixed length");
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, BinaryError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("fixed length");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, BinaryError> {
        let bytes: [u8; 8] = self.take(8)?.try_into().expect("fixed length");
        Ok(u64::from_le_bytes(bytes))
    }

    fn count(&mut self, kind: &'static str, collection_limit: usize) -> Result<usize, BinaryError> {
        let length = usize::try_from(self.u32()?).expect("u32 always fits usize");
        if length > collection_limit {
            return Err(BinaryError::LengthLimit { kind, length });
        }
        self.elements = self.elements.saturating_add(length);
        if self.elements > MAX_TOTAL_ELEMENTS {
            return Err(BinaryError::LengthLimit {
                kind: "total elements",
                length: self.elements,
            });
        }
        Ok(length)
    }

    fn string(&mut self) -> Result<LuaString, BinaryError> {
        let length = usize::try_from(self.u32()?).expect("u32 always fits usize");
        if length > MAX_STRING_BYTES {
            return Err(BinaryError::LengthLimit {
                kind: "string",
                length,
            });
        }
        self.string_bytes = self.string_bytes.saturating_add(length);
        if self.string_bytes > MAX_TOTAL_STRING_BYTES {
            return Err(BinaryError::StringBudgetExceeded {
                length: self.string_bytes,
            });
        }
        Ok(LuaString::from(self.take(length)?))
    }

    #[allow(clippy::too_many_lines)]
    fn chunk(&mut self, depth: usize) -> Result<Chunk, BinaryError> {
        if depth > MAX_DEPTH {
            return Err(BinaryError::DepthLimit { depth });
        }
        let register_count = self.u8()?;
        let parameter_count = self.u8()?;
        let flags = self.u8()?;
        if flags & !(FLAG_VARARG | FLAG_ARG_REGISTER) != 0 {
            return Err(BinaryError::UnknownHeaderFlags {
                flags: u16::from(flags),
            });
        }
        let is_vararg = flags & FLAG_VARARG != 0;
        let arg_register = if flags & FLAG_ARG_REGISTER != 0 {
            Some(self.u8()?)
        } else {
            None
        };
        let source_name = self.string()?;
        let definition_line = self.u32()?;
        let last_definition_line = self.u32()?;

        let constant_count = self.count("constants", MAX_COLLECTION_ELEMENTS)?;
        let mut constants = Vec::with_capacity(constant_count);
        for _ in 0..constant_count {
            let constant = match self.u8()? {
                0 => Constant::Nil,
                1 => match self.u8()? {
                    0 => Constant::Boolean(false),
                    1 => Constant::Boolean(true),
                    value => return Err(BinaryError::MalformedBoolean { value }),
                },
                2 => {
                    let bytes: [u8; 8] = self.take(8)?.try_into().expect("fixed length");
                    Constant::Integer(i64::from_le_bytes(bytes))
                }
                3 => Constant::Number(f64::from_bits(self.u64()?)),
                4 => Constant::String(self.string()?),
                5 => {
                    let bytes: [u8; 8] = self.take(8)?.try_into().expect("fixed length");
                    Constant::Int64(i64::from_le_bytes(bytes))
                }
                6 => Constant::UInt64(self.u64()?),
                tag => return Err(BinaryError::UnknownConstantTag { tag }),
            };
            constants.push(constant);
        }

        let code_count = self.count("instructions", MAX_CODE_OR_LINE_ELEMENTS)?;
        let mut code = Vec::with_capacity(code_count);
        for _ in 0..code_count {
            let instruction = Instruction::from_raw(self.u32()?);
            instruction
                .opcode()
                .map_err(|InvalidOpCode(opcode)| BinaryError::UnknownOpcode { opcode })?;
            code.push(instruction);
        }

        let line_count = self.count("debug line entries", MAX_CODE_OR_LINE_ELEMENTS)?;
        let mut line_info = Vec::with_capacity(line_count);
        for _ in 0..line_count {
            line_info.push(self.u32()?);
        }

        let local_count = self.count("debug local variables", MAX_COLLECTION_ELEMENTS)?;
        let mut local_variables = Vec::with_capacity(local_count);
        for _ in 0..local_count {
            let name = self.string()?;
            let register = self.u8()?;
            let start_pc = usize::try_from(self.u32()?).expect("u32 always fits usize");
            let end_pc = usize::try_from(self.u32()?).expect("u32 always fits usize");
            local_variables.push(LocalVariable::new(name, register, start_pc, end_pc));
        }

        let upvalue_count = self.count("upvalues", MAX_COLLECTION_ELEMENTS)?;
        let mut upvalues = Vec::with_capacity(upvalue_count);
        for _ in 0..upvalue_count {
            let name = self.string()?;
            let source = match self.u8()? {
                0 => UpvalueSource::ParentRegister(self.u8()?),
                1 => UpvalueSource::ParentUpvalue(self.u16()?),
                2 => UpvalueSource::Detached,
                kind => return Err(BinaryError::UnknownUpvalueSource { kind }),
            };
            upvalues.push(UpvalueDescriptor { name, source });
        }

        let call_site_count = self.count("call sites", MAX_COLLECTION_ELEMENTS)?;
        let mut call_sites = Vec::with_capacity(call_site_count);
        for _ in 0..call_site_count {
            let pc = usize::try_from(self.u32()?).expect("u32 always fits usize");
            let flags = self.u8()?;
            if flags & !(CALL_SITE_NAME | CALL_SITE_KIND | CALL_SITE_ITERATOR) != 0 {
                return Err(BinaryError::UnknownCallSiteFlags { flags });
            }
            let name = if flags & CALL_SITE_NAME != 0 {
                Some(self.string()?)
            } else {
                None
            };
            let kind = if flags & CALL_SITE_KIND != 0 {
                Some(call_kind_from_tag(self.u8()?)?)
            } else {
                None
            };
            let iterator = flags & CALL_SITE_ITERATOR != 0;
            let call_site = match (name, kind) {
                (Some(name), Some(kind)) => CallSite::named(pc, name, kind),
                (None, Some(CallKind::ForIterator)) => CallSite::for_iterator(pc),
                (None, None) => CallSite::anonymous(pc),
                _ => return Err(BinaryError::MalformedCallSite),
            };
            call_sites.push(call_site.with_iterator(iterator));
        }

        let diagnostic_site_count = self.count("diagnostic sites", MAX_COLLECTION_ELEMENTS)?;
        let mut diagnostic_sites = Vec::with_capacity(diagnostic_site_count);
        for _ in 0..diagnostic_site_count {
            let pc = usize::try_from(self.u32()?).expect("u32 always fits usize");
            let register = self.u8()?;
            let name = self.string()?;
            let kind = call_kind_from_tag(self.u8()?)?;
            diagnostic_sites.push(DiagnosticSite::new(pc, register, name, kind));
        }

        let prototype_count = self.count("prototypes", MAX_COLLECTION_ELEMENTS)?;
        self.prototypes = self.prototypes.saturating_add(prototype_count);
        if self.prototypes > MAX_PROTOTYPES {
            return Err(BinaryError::PrototypeLimit {
                count: self.prototypes,
            });
        }
        let mut prototypes = Vec::with_capacity(prototype_count);
        for _ in 0..prototype_count {
            prototypes.push(Arc::new(self.chunk(depth + 1)?));
        }

        let code_len = code.len();
        let chunk = Chunk {
            register_count,
            parameter_count,
            constants,
            code,
            prototypes,
            upvalues,
            is_vararg,
            arg_register,
            debug: DebugMetadata::new(source_name, call_sites)
                .with_function_lines(definition_line, last_definition_line)
                .with_diagnostic_sites(diagnostic_sites)
                .with_local_variables(local_variables)
                .with_line_info(line_info),
            captures: crate::bytecode::CaptureVolatility::default(),
            constructors: crate::bytecode::ConstructorShapes::default(),
            live: crate::bytecode::LiveRegisters::default(),
            field_sites: crate::bytecode::FieldSites::for_code_len(code_len),
        };
        Ok(chunk)
    }
}

const fn call_kind_tag(kind: CallKind) -> u8 {
    match kind {
        CallKind::Global => 0,
        CallKind::Local => 1,
        CallKind::Upvalue => 2,
        CallKind::Field => 3,
        CallKind::Method => 4,
        CallKind::Constant => 5,
        CallKind::ForIterator => 6,
    }
}

fn call_kind_from_tag(kind: u8) -> Result<CallKind, BinaryError> {
    match kind {
        0 => Ok(CallKind::Global),
        1 => Ok(CallKind::Local),
        2 => Ok(CallKind::Upvalue),
        3 => Ok(CallKind::Field),
        4 => Ok(CallKind::Method),
        5 => Ok(CallKind::Constant),
        6 => Ok(CallKind::ForIterator),
        kind => Err(BinaryError::UnknownCallKind { kind }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portable_chunk() -> Chunk {
        let child = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::get_upvalue(0, 0), Instruction::return_(0)],
            vec![],
            vec![UpvalueDescriptor {
                name: LuaString::from(b"inner\0\xff" as &[u8]),
                source: UpvalueSource::ParentUpvalue(0),
            }],
        )
        .with_debug_metadata(
            DebugMetadata::new("child", Vec::new())
                .with_local_variables(vec![LocalVariable::new("parameter", 0, 0, 2)])
                .with_line_info(vec![7, 7]),
        );
        let mut root = Chunk::with_parts(
            2,
            0,
            vec![
                Constant::Integer(-9),
                Constant::Number(1.25),
                Constant::String(LuaString::from(b"raw\0\xff" as &[u8])),
            ],
            vec![
                Instruction::get_upvalue(0, 0),
                Instruction::closure(1, 0),
                Instruction::call(1, 0, 0),
                Instruction::halt(),
            ],
            vec![child],
            vec![],
        )
        .with_debug_metadata(
            DebugMetadata::new(
                LuaString::from(b"source\xff" as &[u8]),
                vec![CallSite::named(2, "work", CallKind::Global)],
            )
            .with_diagnostic_sites(vec![DiagnosticSite::new(
                0,
                0,
                "captured",
                CallKind::Upvalue,
            )])
            .with_local_variables(vec![
                LocalVariable::new("outer", 0, 0, 4),
                LocalVariable::new("inner", 1, 2, 3),
            ])
            .with_function_lines(3, 12)
            .with_line_info(vec![3, 3, 4, 4]),
        );
        root.upvalues.push(UpvalueDescriptor {
            name: LuaString::from(b"root\0\xff" as &[u8]),
            source: UpvalueSource::ParentRegister(0),
        });
        root
    }

    fn simple_chunk(constants: Vec<Constant>) -> Chunk {
        Chunk::new(1, constants, vec![Instruction::halt()])
            .with_debug_metadata(DebugMetadata::new("", Vec::new()))
    }

    fn skip_string(bytes: &[u8], position: &mut usize) {
        let length = u32::from_le_bytes(bytes[*position..*position + 4].try_into().unwrap());
        *position += 4 + usize::try_from(length).unwrap();
    }

    fn first_constant_tag(bytes: &[u8]) -> usize {
        let mut position = MAGIC.len() + 4 + 3;
        skip_string(bytes, &mut position);
        position += 8; // function definition lines
        position + 4
    }

    fn first_instruction(bytes: &[u8]) -> usize {
        let mut position = MAGIC.len() + 4 + 3;
        skip_string(bytes, &mut position);
        position += 8; // function definition lines
        let constant_count = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        assert_eq!(constant_count, 0);
        position + 4 + 4
    }

    fn first_upvalue_source(bytes: &[u8]) -> usize {
        let mut position = MAGIC.len() + 4 + 3;
        skip_string(bytes, &mut position);
        position += 8; // function definition lines
        let constants = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        assert_eq!(constants, 0);
        position += 4;
        let code = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        position += 4 + usize::try_from(code).unwrap() * 4;
        let line_info = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        position += 4 + usize::try_from(line_info).unwrap() * 4;
        let locals = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        assert_eq!(locals, 0);
        position += 4;
        let upvalues = u32::from_le_bytes(bytes[position..position + 4].try_into().unwrap());
        assert_eq!(upvalues, 1);
        position += 4;
        skip_string(bytes, &mut position);
        position
    }

    fn raw_chunk(remaining_children: usize) -> Vec<u8> {
        let mut bytes = vec![1, 0, 0];
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // source name
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // definition line
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // last definition line
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // constants
        bytes.extend_from_slice(&1_u32.to_le_bytes()); // code
        bytes.extend_from_slice(&Instruction::halt().raw().to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // debug line entries
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // debug local variables
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // upvalues
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // call sites
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // diagnostic sites
        bytes.extend_from_slice(&u32::from(remaining_children > 0).to_le_bytes());
        if remaining_children > 0 {
            bytes.extend(raw_chunk(remaining_children - 1));
        }
        bytes
    }

    #[test]
    fn round_trips_the_unnamed_generic_for_iterator_call_site() {
        let chunk = Chunk::with_parts(
            2,
            0,
            vec![],
            vec![Instruction::call(0, 0, 0), Instruction::halt()],
            vec![],
            vec![],
        )
        .with_debug_metadata(DebugMetadata::new("loop", vec![CallSite::for_iterator(0)]));
        let loaded = load(&dump(&chunk).unwrap()).unwrap();
        assert_eq!(loaded.debug.call_sites(), &[CallSite::for_iterator(0)]);
        assert!(loaded.debug.call_sites()[0].is_iterator());
        assert!(loaded.debug.call_sites()[0].name().is_none());
    }

    #[test]
    fn round_trips_full_portable_chunks_and_detaches_the_root() {
        let root = portable_chunk();
        let bytes = dump(&root).unwrap();
        assert!(claims(&bytes));
        let loaded = load(&bytes).unwrap();
        assert_eq!(loaded.debug.source_name().as_bytes(), b"source\xff");
        assert_eq!(
            loaded.constants[2],
            Constant::String(LuaString::from(b"raw\0\xff" as &[u8]))
        );
        assert_eq!(
            loaded.debug.call_sites()[0].name().unwrap().as_bytes(),
            b"work"
        );
        assert_eq!(loaded.debug.call_sites()[0].kind(), Some(CallKind::Global));
        assert_eq!(
            loaded.debug.diagnostic_sites()[0].name().as_bytes(),
            b"captured"
        );
        assert_eq!(loaded.debug.diagnostic_sites()[0].kind(), CallKind::Upvalue);
        assert_eq!(loaded.debug.line_info(), &[3, 3, 4, 4]);
        assert_eq!(loaded.debug.active_lines(), vec![3, 4]);
        assert_eq!(loaded.debug.definition_line(), 3);
        assert_eq!(loaded.debug.last_definition_line(), 12);
        assert_eq!(
            loaded
                .debug
                .locals_at(2)
                .map(|local| (local.name().as_bytes(), local.register()))
                .collect::<Vec<_>>(),
            vec![(b"outer" as &[u8], 0), (b"inner" as &[u8], 1)]
        );
        assert_eq!(
            loaded.prototypes[0].debug.local_variables(),
            &[LocalVariable::new("parameter", 0, 0, 2)]
        );
        assert_eq!(loaded.prototypes[0].debug.line_info(), &[7, 7]);
        assert_eq!(loaded.upvalues[0].name.as_bytes(), b"root\0\xff");
        assert_eq!(loaded.upvalues[0].source, UpvalueSource::Detached);
        assert_eq!(
            loaded.prototypes[0].upvalues[0].source,
            UpvalueSource::ParentUpvalue(0)
        );
        assert_eq!(
            loaded.prototypes[0].upvalues[0].name.as_bytes(),
            b"inner\0\xff"
        );
        assert_eq!(loaded.validate(), Ok(()));
    }

    #[test]
    fn dump_is_deterministic() {
        let chunk = portable_chunk();
        assert_eq!(dump(&chunk).unwrap(), dump(&chunk).unwrap());
    }

    #[test]
    fn round_trips_lua51_verybig_sized_code_and_line_info() {
        const VERYBIG_INSTRUCTION_COUNT: usize = 1_065_076;

        let mut code = vec![Instruction::load_nil(0); VERYBIG_INSTRUCTION_COUNT - 1];
        code.push(Instruction::halt());
        let chunk = Chunk::new(1, Vec::new(), code).with_debug_metadata(
            DebugMetadata::new("verybig.lua", Vec::new())
                .with_line_info(vec![1; VERYBIG_INSTRUCTION_COUNT]),
        );

        let bytes = dump(&chunk).unwrap();
        assert_eq!(load(&bytes).unwrap(), chunk);
    }

    #[test]
    fn round_trips_wide_references() {
        let index = u32::from(u16::MAX) + 1;
        let mut constants = vec![Constant::Nil; usize::try_from(index + 1).unwrap()];
        constants[usize::try_from(index).unwrap()] = Constant::Number(7.0);
        let chunk = Chunk::new(
            1,
            constants,
            vec![
                Instruction::load_constant_wide(0),
                Instruction::extra_argument(index).unwrap(),
                Instruction::halt(),
            ],
        );
        let bytes = dump(&chunk).unwrap();
        assert_eq!(load(&bytes).unwrap(), chunk);
    }

    #[test]
    fn round_trips_compare_and_branch_tests() {
        let jump = Instruction::aj(crate::bytecode::OpCode::Jump, 0).unwrap();
        let chunk = Chunk::new(
            2,
            vec![Constant::Number(1.5), Constant::Nil],
            vec![
                Instruction::test(crate::bytecode::OpCode::TestEqual, 0, 1, true, false),
                jump,
                Instruction::test(crate::bytecode::OpCode::TestLessThanK, 0, 0, false, true),
                jump,
                Instruction::test(crate::bytecode::OpCode::TestEqualK, 1, 1, false, false),
                jump,
                Instruction::halt(),
            ],
        );
        let bytes = dump(&chunk).unwrap();
        let loaded = load(&bytes).unwrap();
        assert_eq!(loaded, chunk);
        assert!(loaded.code[2].test_constant_left() && !loaded.code[2].test_expected());
        // A test without its jump word does not dump.
        let mut broken = chunk.clone();
        broken.code[1] = Instruction::halt();
        assert!(dump(&broken).is_err());
    }

    #[test]
    fn round_trips_generic_for_instructions() {
        let chunk = Chunk::new(
            8,
            vec![],
            vec![
                Instruction::aj(crate::bytecode::OpCode::Jump, 1).unwrap(),
                Instruction::halt(),
                Instruction::for_in_call(2, 2),
                Instruction::for_in_loop(2, -2),
                Instruction::halt(),
            ],
        )
        .with_debug_metadata(DebugMetadata::new("loop", vec![CallSite::for_iterator(2)]));
        let bytes = dump(&chunk).unwrap();
        let loaded = load(&bytes).unwrap();
        assert_eq!(loaded, chunk);
        assert_eq!(loaded.code[2].b(), 2);
        assert_eq!(loaded.debug.call_sites(), &[CallSite::for_iterator(2)]);
        // The version moved with the new opcodes, so older dumps do not load.
        let mut old = bytes.clone();
        old[MAGIC.len()..MAGIC.len() + 2].copy_from_slice(&(VERSION - 1).to_le_bytes());
        assert!(load(&old).is_err());
        // A call-site entry on a loop test is not a call.
        let broken = Chunk::new(
            8,
            vec![],
            vec![Instruction::for_in_loop(0, -1), Instruction::halt()],
        )
        .with_debug_metadata(DebugMetadata::new("loop", vec![CallSite::for_iterator(0)]));
        assert!(dump(&broken).is_err());
    }

    #[test]
    fn round_trips_call_one_and_clear_range() {
        let chunk = Chunk::new(
            4,
            Vec::new(),
            vec![
                Instruction::call_one(0, 1, 2),
                Instruction::clear_range(1, 3),
                Instruction::halt(),
            ],
        );
        let bytes = dump(&chunk).unwrap();
        assert_eq!(load(&bytes).unwrap(), chunk);
    }

    #[test]
    fn rejects_every_other_format_version() {
        let bytes = dump(&simple_chunk(Vec::new())).unwrap();
        assert_eq!(
            u16::from_le_bytes(bytes[MAGIC.len()..MAGIC.len() + 2].try_into().unwrap()),
            VERSION
        );
        for version in [2, 7, VERSION + 1] {
            let mut other = bytes.clone();
            other[MAGIC.len()..MAGIC.len() + 2].copy_from_slice(&version.to_le_bytes());
            assert_eq!(
                load(&other),
                Err(BinaryError::UnsupportedVersion { version })
            );
        }
    }

    #[test]
    fn rejects_every_truncated_prefix() {
        let bytes = dump(&portable_chunk()).unwrap();
        for end in 0..bytes.len() {
            assert!(matches!(
                load(&bytes[..end]),
                Err(BinaryError::UnexpectedEof | BinaryError::InvalidMagic)
            ));
        }
    }

    #[test]
    fn rejects_hostile_tags_counts_opcodes_captures_and_trailing_data() {
        let simple = dump(&simple_chunk(Vec::new())).unwrap();
        let instruction = first_instruction(&simple);
        let mut unknown_opcode = simple.clone();
        unknown_opcode[instruction] = u8::MAX;
        assert!(matches!(
            load(&unknown_opcode),
            Err(BinaryError::UnknownOpcode { opcode: u8::MAX })
        ));

        let mut hostile_count = simple.clone();
        let mut count_position = MAGIC.len() + 4 + 3;
        skip_string(&hostile_count, &mut count_position);
        count_position += 8; // function definition lines
        hostile_count[count_position..count_position + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            load(&hostile_count),
            Err(BinaryError::LengthLimit { .. })
        ));

        let mut excessive_code = simple.clone();
        let code_count = first_instruction(&excessive_code) - 4;
        excessive_code[code_count..code_count + 4].copy_from_slice(
            &u32::try_from(MAX_CODE_OR_LINE_ELEMENTS + 1)
                .unwrap()
                .to_le_bytes(),
        );
        assert_eq!(
            load(&excessive_code),
            Err(BinaryError::LengthLimit {
                kind: "instructions",
                length: MAX_CODE_OR_LINE_ELEMENTS + 1,
            })
        );

        let boolean = dump(&simple_chunk(vec![Constant::Boolean(true)])).unwrap();
        let tag = first_constant_tag(&boolean);
        let mut malformed_boolean = boolean.clone();
        malformed_boolean[tag + 1] = 9;
        assert!(matches!(
            load(&malformed_boolean),
            Err(BinaryError::MalformedBoolean { value: 9 })
        ));
        let mut unknown_tag = boolean;
        unknown_tag[tag] = u8::MAX;
        assert!(matches!(
            load(&unknown_tag),
            Err(BinaryError::UnknownConstantTag { tag: u8::MAX })
        ));

        let mut root = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::get_upvalue(0, 0), Instruction::halt()],
            vec![],
            vec![UpvalueDescriptor {
                name: "root".into(),
                source: UpvalueSource::Detached,
            }],
        )
        .with_debug_metadata(DebugMetadata::new("", Vec::new()));
        let bytes = dump(&root).unwrap();
        let source = first_upvalue_source(&bytes);
        root.upvalues[0].source = UpvalueSource::ParentRegister(0);
        let mut unknown_capture = bytes;
        unknown_capture[source] = u8::MAX;
        assert!(matches!(
            load(&unknown_capture),
            Err(BinaryError::UnknownUpvalueSource { kind: u8::MAX })
        ));

        let mut trailing = dump(&simple_chunk(Vec::new())).unwrap();
        trailing.push(0);
        assert!(matches!(
            load(&trailing),
            Err(BinaryError::TrailingData { length: 1 })
        ));
    }

    #[test]
    fn checks_depth_and_rejects_detached_nested_descriptors() {
        fn nested(depth: usize) -> Chunk {
            let prototypes = if depth == 0 {
                Vec::new()
            } else {
                vec![nested(depth - 1)]
            };
            Chunk::with_parts(
                1,
                0,
                vec![],
                vec![Instruction::halt()],
                prototypes,
                Vec::new(),
            )
        }
        assert!(matches!(
            dump(&nested(MAX_DEPTH + 1)),
            Err(BinaryError::DepthLimit { .. })
        ));

        let mut hostile_depth = Vec::from(MAGIC.as_slice());
        hostile_depth.extend_from_slice(&VERSION.to_le_bytes());
        hostile_depth.extend_from_slice(&0_u16.to_le_bytes());
        hostile_depth.extend(raw_chunk(MAX_DEPTH + 1));
        assert!(matches!(
            load(&hostile_depth),
            Err(BinaryError::DepthLimit { .. })
        ));

        let detached_child = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::halt()],
            vec![],
            vec![UpvalueDescriptor {
                name: "bad".into(),
                source: UpvalueSource::Detached,
            }],
        );
        let parent = Chunk::with_parts(
            1,
            0,
            vec![],
            vec![Instruction::halt()],
            vec![detached_child],
            vec![],
        );
        assert!(matches!(
            dump(&parent),
            Err(BinaryError::InvalidChunk(
                ValidationError::DetachedUpvalueInNestedChunk { .. }
            ))
        ));
    }
}
