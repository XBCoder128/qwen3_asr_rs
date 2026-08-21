//! Streaming smoke test: feed real speech in 1s windows and print the final
//! transcript. Exercises the incremental KV path, the per-call decode budget
//! scaling and the junk-token guard in streaming mode.
//!
//! Usage:
//!   cargo run --release --no-default-features --features mlx \
//!     --example streaming_smoke -- <model_dir> <audio_file> [chunk_ms] [rollback]

use anyhow::{Context, Result};
use std::path::Path;

use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::tensor::Device;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: streaming_smoke <model_dir> <audio_file> [chunk_ms] [rollback]");
        std::process::exit(1);
    }
    let model_dir = Path::new(&args[1]);
    let audio_file = &args[2];
    let chunk_ms: usize = args.get(3).map(|s| s.parse().unwrap_or(1000)).unwrap_or(1000);
    let rollback: usize = args.get(4).map(|s| s.parse().unwrap_or(5)).unwrap_or(5);

    #[cfg(feature = "mlx")]
    {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
    }

    let mut model = AsrInference::load(model_dir, Device::Gpu(0))
        .context("Failed to load model")?;

    let samples =
        qwen3_asr_rs::audio::load_audio(audio_file, 16_000).context("Failed to load audio")?;
    let chunk = (chunk_ms * 16_000) / 1000; // samples per call
    eprintln!(
        "[smoke] {} samples ({:.1}s), chunk={}ms rollback={}",
        samples.len(),
        samples.len() as f64 / 16_000.0,
        chunk_ms,
        rollback
    );

    let mut state = model.init_streaming(None, rollback)?;
    let mut last_text = String::new();
    for end in (chunk..=samples.len()).step_by(chunk) {
        let r = model.streaming_transcribe(&samples[..end], &mut state)?;
        last_text = r.text;
        eprintln!("[smoke] @ {:.1}s: {}", end as f64 / 16_000.0, last_text);
    }
    // Flush any tail shorter than a full chunk.
    if samples.len() % chunk != 0 {
        let r = model.streaming_transcribe(&samples, &mut state)?;
        last_text = r.text;
        eprintln!("[smoke] final: {}", last_text);
    }

    println!("Streaming final text: {}", last_text);
    let offline = model.transcribe(audio_file, None)?;
    println!("Offline       text: {}", offline.text);
    Ok(())
}
