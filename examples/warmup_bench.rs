//! Warmup + transcription benchmark.
//!
//! Demonstrates the warmup() API and measures cold-start vs warm performance.
//!
//! Usage:
//!   cargo run --release --no-default-features --features mlx --example warmup_bench -- <model_dir> <audio_file>

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::tensor::Device;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: warmup_bench <model_dir> <audio_file>");
        std::process::exit(1);
    }

    let model_dir = Path::new(&args[1]);
    let audio_file = &args[2];

    #[cfg(feature = "mlx")]
    {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
    }

    let device = Device::Gpu(0);

    eprintln!("[bench] Loading model...");
    let t_load = Instant::now();
    let mut model = AsrInference::load(model_dir, device).context("Failed to load model")?;
    eprintln!("[bench] Model loaded in {:.0}ms", t_load.elapsed().as_millis());

    // --- Warmup ---
    eprintln!("\n[bench] === Running warmup() ===");
    let t_warmup = Instant::now();
    model.warmup()?;
    let warmup_ms = t_warmup.elapsed().as_millis();
    eprintln!("[bench] Warmup completed in {}ms", warmup_ms);

    // --- First real transcription (should be fast now) ---
    eprintln!("\n[bench] === First transcription (post-warmup) ===");
    let t1 = Instant::now();
    let result1 = model.transcribe(audio_file, None)?;
    let t1_ms = t1.elapsed().as_millis();
    eprintln!("[bench] First transcription: {}ms", t1_ms);
    eprintln!("[bench] Language: {}", result1.language);
    eprintln!("[bench] Text: {}", result1.text);

    // --- Second transcription (fully warm) ---
    eprintln!("\n[bench] === Second transcription (fully warm) ===");
    let t2 = Instant::now();
    let result2 = model.transcribe(audio_file, None)?;
    let t2_ms = t2.elapsed().as_millis();
    eprintln!("[bench] Second transcription: {}ms", t2_ms);
    eprintln!("[bench] Text: {}", result2.text);

    // --- Consistency check ---
    if result1.text == result2.text {
        eprintln!("\n[bench] ✓ Output consistent across runs");
    } else {
        anyhow::bail!(
            "post-warmup output differs between runs\nRun 1: {}\nRun 2: {}",
            result1.text,
            result2.text
        );
    }

    eprintln!("\n[bench] Summary:");
    eprintln!("  Warmup:          {}ms", warmup_ms);
    eprintln!("  1st transcribe:  {}ms (post-warmup)", t1_ms);
    eprintln!("  2nd transcribe:  {}ms (steady-state)", t2_ms);

    Ok(())
}
