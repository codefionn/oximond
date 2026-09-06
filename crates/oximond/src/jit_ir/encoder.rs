//! The semantic chunk encoder and SHA-256 used for cache digests.

use crate::bytecode::{Chunk, Constant, UpvalueSource};

pub(crate) struct SemanticEncoder {
    bytes: Vec<u8>,
}

impl SemanticEncoder {
    pub(crate) const fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }
    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes(&[value]);
    }
    pub(crate) fn len(&mut self, value: usize) {
        self.bytes(&u64::try_from(value).expect("length fits u64").to_le_bytes());
    }
    pub(crate) fn chunk(&mut self, chunk: &Chunk) {
        self.u8(chunk.register_count);
        self.u8(chunk.parameter_count);
        self.u8(u8::from(chunk.is_vararg));
        match chunk.arg_register {
            Some(register) => {
                self.u8(1);
                self.u8(register);
            }
            None => self.u8(0),
        }
        self.len(chunk.constants.len());
        for constant in &chunk.constants {
            match constant {
                Constant::Nil => self.u8(0),
                Constant::Boolean(value) => {
                    self.u8(1);
                    self.u8(u8::from(*value));
                }
                Constant::Integer(value) => {
                    self.u8(2);
                    self.bytes(&value.to_le_bytes());
                }
                Constant::Number(value) => {
                    self.u8(3);
                    self.bytes(&value.to_bits().to_le_bytes());
                }
                Constant::String(value) => {
                    self.u8(4);
                    self.len(value.len());
                    self.bytes(value.as_bytes());
                }
                Constant::Int64(value) => {
                    self.u8(5);
                    self.bytes(&value.to_le_bytes());
                }
                Constant::UInt64(value) => {
                    self.u8(6);
                    self.bytes(&value.to_le_bytes());
                }
            }
        }
        self.len(chunk.code.len());
        for instruction in &chunk.code {
            self.bytes(&instruction.raw().to_le_bytes());
        }
        self.len(chunk.upvalues.len());
        for upvalue in &chunk.upvalues {
            match upvalue.source {
                UpvalueSource::ParentRegister(register) => {
                    self.u8(0);
                    self.u8(register);
                }
                UpvalueSource::ParentUpvalue(index) => {
                    self.u8(1);
                    self.bytes(&index.to_le_bytes());
                }
                UpvalueSource::Detached => self.u8(2),
            }
        }
        self.len(chunk.prototypes.len());
        for prototype in &chunk.prototypes {
            self.chunk(prototype);
        }
    }
    pub(crate) fn finish(self) -> [u8; 32] {
        sha256(&self.bytes)
    }
}

// Kept here instead of depending on a native hashing library. This cache key
// must be identical on every supported host and cannot tolerate practical
// collisions because a collision could select code for another chunk.
// The round variable names follow the SHA-256 specification.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub(crate) fn sha256(input: &[u8]) -> [u8; 32] {
    pub(crate) const INITIAL: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    pub(crate) const ROUND: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_length = u64::try_from(input.len())
        .expect("semantic chunk encoding fits u64")
        .checked_mul(8)
        .expect("semantic chunk bit length fits u64");
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = INITIAL;
    for block in message.as_chunks::<64>().0 {
        let mut schedule = [0_u32; 64];
        for (word, bytes) in schedule.iter_mut().take(16).zip(block.as_chunks::<4>().0) {
            *word = u32::from_be_bytes(*bytes);
        }
        for index in 16..64 {
            let sigma0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let sigma1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(sigma0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(sigma1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (constant, word) in ROUND.into_iter().zip(schedule) {
            let choice = (e & f) ^ (!e & g);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(constant)
                .wrapping_add(word);
            let temporary2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0; 32];
    for (destination, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *destination = word.to_be_bytes();
    }
    digest
}
