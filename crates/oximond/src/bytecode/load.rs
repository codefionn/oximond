//! The move-only byte-stream loading seam shared by the VM and compilers.

mod file;
pub use file::{load_file_reader, load_file_reader_pipelined, load_small_file_reader};

use crate::bytecode::Chunk;
use crate::string::LuaString;

/// Starts independent source-loading sessions.
pub trait ChunkLoader {
    /// Starts a fresh session for an owned, byte-preserving source name.
    fn begin(&self, source_name: LuaString) -> Box<dyn ChunkLoad>;

    /// Starts a concurrent parser/compiler session when supported.
    ///
    /// Callers must only select this for readers where reading ahead cannot
    /// block indefinitely after a compiler failure, such as regular files.
    /// Custom loaders keep their synchronous behavior by default.
    fn begin_pipelined(&self, _source_name: LuaString) -> Option<Box<dyn ChunkLoad>> {
        None
    }

    /// Loads a source whose bytes are all known up front, such as a string.
    /// It gives the same result as streaming the bytes through
    /// [`Self::begin`], which is what the default does; a loader may
    /// override it with a faster path for complete input.
    fn load_complete(&self, source_name: LuaString, source: LuaString) -> LoadPoll {
        let session = self.begin(source_name);
        let poll = if source.as_bytes().is_empty() {
            LoadPoll::NeedInput(session)
        } else {
            session.resume(LoadInput::Bytes(source))
        };
        match poll {
            LoadPoll::NeedInput(session) => session.resume(LoadInput::End),
            terminal => terminal,
        }
    }
}

/// A move-only source-loading session.
///
/// Resuming consumes the session. Only [`LoadPoll::NeedInput`] returns a new
/// session, which makes loaded and failed states terminal by construction.
pub trait ChunkLoad {
    /// Takes an available terminal result without blocking or supplying input.
    /// Concurrent sessions use this to stop reads after a worker failure.
    fn poll(&mut self) -> Option<LoadResult> {
        None
    }

    /// Supplies one owned input item and advances this session.
    fn resume(self: Box<Self>, input: LoadInput) -> LoadPoll;
}

/// One owned item delivered to a [`ChunkLoad`] session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoadInput {
    /// A byte fragment. Empty fragments are valid and retain pull scheduling.
    Bytes(LuaString),
    /// The explicit end of the source stream.
    End,
}

/// A terminal result produced by a concurrent loading session.
#[allow(clippy::large_enum_variant)]
pub enum LoadResult {
    /// The complete source compiled successfully.
    Loaded(Chunk),
    /// Parsing or compilation failed.
    Failed(LuaString),
}

impl LoadResult {
    /// Converts this terminal result into the common loading outcome.
    #[must_use]
    pub fn into_poll(self) -> LoadPoll {
        match self {
            Self::Loaded(chunk) => LoadPoll::Loaded(chunk),
            Self::Failed(error) => LoadPoll::Failed(error),
        }
    }
}

/// The outcome of advancing a [`ChunkLoad`] session.
#[allow(clippy::large_enum_variant)] // Keeping the loaded chunk inline avoids heap allocation on every successful load.
pub enum LoadPoll {
    /// The session needs another fragment or end marker.
    NeedInput(Box<dyn ChunkLoad>),
    /// The source compiled successfully. This has no continuation.
    Loaded(Chunk),
    /// The source failed with a byte-preserving diagnostic. This has no continuation.
    Failed(LuaString),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CollectingLoader;

    impl ChunkLoader for CollectingLoader {
        fn begin(&self, source_name: LuaString) -> Box<dyn ChunkLoad> {
            Box::new(CollectingLoad {
                source_name,
                bytes: Vec::new(),
            })
        }
    }

    struct CollectingLoad {
        source_name: LuaString,
        bytes: Vec<u8>,
    }

    impl ChunkLoad for CollectingLoad {
        fn resume(mut self: Box<Self>, input: LoadInput) -> LoadPoll {
            match input {
                LoadInput::Bytes(bytes) => {
                    self.bytes.extend_from_slice(bytes.as_bytes());
                    LoadPoll::NeedInput(self)
                }
                LoadInput::End => {
                    let mut message = self.source_name.as_bytes().to_vec();
                    message.extend_from_slice(&self.bytes);
                    LoadPoll::Failed(LuaString::from(message))
                }
            }
        }
    }

    #[test]
    fn moves_arbitrary_bytes_between_resumes_and_end_is_terminal() {
        let loader = CollectingLoader;
        let session = loader.begin(LuaString::from(b"name\xff" as &[u8]));
        let session = match session.resume(LoadInput::Bytes(LuaString::from(b"a\0" as &[u8]))) {
            LoadPoll::NeedInput(session) => session,
            LoadPoll::Loaded(_) | LoadPoll::Failed(_) => panic!("fragment ended a live session"),
        };
        let session = match session.resume(LoadInput::Bytes(LuaString::from(b"\xff" as &[u8]))) {
            LoadPoll::NeedInput(session) => session,
            LoadPoll::Loaded(_) | LoadPoll::Failed(_) => panic!("fragment ended a live session"),
        };
        match session.resume(LoadInput::End) {
            LoadPoll::Failed(message) => assert_eq!(message.as_bytes(), b"name\xffa\0\xff"),
            LoadPoll::NeedInput(_) | LoadPoll::Loaded(_) => panic!("end must terminate a session"),
        }
    }
}
