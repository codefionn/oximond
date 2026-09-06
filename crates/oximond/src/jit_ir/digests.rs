//! Semantic chunk digests.

use crate::bytecode::Chunk;
use crate::jit_ir::encoder::SemanticEncoder;

/// A deterministic digest of the executable meaning of a bytecode chunk.
///
/// Debug names, source paths, line tables, and allocation addresses are not
/// included. This is an in-process cache identity, not a cryptographic signature.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ChunkDigest([u8; 32]);

impl ChunkDigest {
    #[must_use]
    pub fn of(chunk: &Chunk) -> Self {
        let mut digest = SemanticEncoder::new();
        digest.bytes(b"oximond-semantic-chunk-v1");
        digest.chunk(chunk);
        Self(digest.finish())
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}
