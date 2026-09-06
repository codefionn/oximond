//! Compare complete, sequential, and concurrent loading with a delayed reader.
use oximond::bytecode::load::{LoadPoll, load_file_reader, load_file_reader_pipelined};
use oximond::compat::CompatibilityProfile;
use oximond::compiler::{CompilerConfig, CompilerLoader, compile_named};
use oximond::string::LuaString;
use std::io::{self, Read};
use std::time::Duration;

struct DelayedReader {
    file: std::fs::File,
    delay: Duration,
}
impl Read for DelayedReader {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let limit = output.len().min(8192);
        let read = self.file.read(&mut output[..limit])?;
        if read != 0 && !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        Ok(read)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().collect();
    let mode = arguments
        .get(1)
        .ok_or("expected mode, file, and delay in microseconds")?;
    let path = arguments.get(2).ok_or("expected source file")?;
    let delay = arguments.get(3).ok_or("expected read delay")?.parse()?;
    let file = std::fs::File::open(path)?;
    let capacity = usize::try_from(file.metadata()?.len())?;
    let mut reader = DelayedReader {
        file,
        delay: Duration::from_micros(delay),
    };
    let name = LuaString::from(format!("@{path}"));
    let config = CompilerConfig::new(CompatibilityProfile::Lua54);
    let loader = CompilerLoader::new(config);
    let chunk = if mode == "complete" {
        let mut source = Vec::with_capacity(capacity);
        reader.read_to_end(&mut source)?;
        compile_named(name, source, config)?
    } else {
        let poll = match mode.as_str() {
            "sequential" => load_file_reader(&loader, &name, reader, None)?,
            "pipeline" => load_file_reader_pipelined(&loader, &name, reader, None)?,
            _ => return Err("expected complete, sequential, or pipeline".into()),
        };
        match poll {
            LoadPoll::Loaded(chunk) => chunk,
            LoadPoll::Failed(message) => return Err(message.to_lossy().into_owned().into()),
            LoadPoll::NeedInput(_) => return Err("EOF requested more input".into()),
        }
    };
    if arguments
        .get(4)
        .is_some_and(|argument| argument == "fingerprint")
    {
        let bytes = oximond::bytecode::binary::dump(&chunk)?;
        let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
        });
        println!("{hash:016x} {}", bytes.len());
    }
    std::hint::black_box(chunk);
    Ok(())
}
