//! Fast streaming-only test — only runs incremental mode, no offline/re-transcribe.
//!
//! Usage:
//!   cargo run --release --no-default-features --features mlx --bin test_streaming_fast -- <model_dir> <audio_file> [chunk_sec] [rollback_tokens]

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::tensor::Device;

const SAMPLE_RATE: usize = 16_000;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Fast streaming-only test");
        eprintln!(
            "Usage: test_streaming_fast <model_dir> <audio_file> [chunk_sec] [rollback_tokens] [partial]"
        );
        eprintln!(
            "  partial: only decode at complete encoder-chunk (1s) boundaries \
             (old behavior; default decodes the tail every call)"
        );
        std::process::exit(1);
    }

    let model_dir = Path::new(&args[1]);
    let audio_file = &args[2];
    let chunk_sec: f64 = args.get(3).map(|s| s.parse().unwrap_or(1.0)).unwrap_or(1.0);
    let rollback: usize = args.get(4).map(|s| s.parse().unwrap_or(3)).unwrap_or(3);
    let partial_mode = args.get(5).map(|s| s == "partial").unwrap_or(false);

    #[cfg(feature = "mlx")]
    {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
    }

    let device = Device::Gpu(0);

    eprintln!("[test] Loading model...");
    let mut model = AsrInference::load(model_dir, device).context("Failed to load model")?;
    eprintln!("[test] Model loaded");

    let samples = qwen3_asr_rs::audio::load_audio(audio_file, SAMPLE_RATE as u32)
        .context("Failed to load audio")?;
    let total_samples = samples.len();
    let duration = total_samples as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[test] Audio: {:.1}s, chunk={}s, rollback={}",
        duration, chunk_sec, rollback
    );

    let chunk_size = (chunk_sec * SAMPLE_RATE as f64) as usize;
    let num_chunks = (total_samples + chunk_size - 1) / chunk_size;

    // --- Incremental streaming only ---
    let t0 = Instant::now();
    let mut stream_state = model.init_streaming(None, rollback)?;
    let mut accumulated: Vec<f32> = Vec::new();
    let mut last_text = String::new();

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, total_samples);
        accumulated.extend_from_slice(&samples[start..end]);

        if accumulated.len() < SAMPLE_RATE {
            continue;
        }

        // Default: process the tail every call so the hypothesis updates at
        // sub-second granularity. "partial" mode skips the tail and only
        // decodes at complete encoder-chunk (1s) boundaries.
        let is_last = i == num_chunks - 1;
        let t_call = Instant::now();
        let result = if is_last || !partial_mode {
            model.streaming_transcribe(&accumulated, &mut stream_state)
        } else {
            model.streaming_transcribe_partial(&accumulated, &mut stream_state)
        }
        .with_context(|| format!("Failed at chunk {}", i + 1))?;

        let acc_dur = accumulated.len() as f64 / SAMPLE_RATE as f64;
        eprintln!(
            "[chunk {}] {:.1}s ({:.0}ms) → {}",
            i + 1,
            acc_dur,
            t_call.elapsed().as_millis(),
            result.text
        );
        last_text = result.text;

        #[cfg(feature = "mlx")]
        {
            // Don't clear_cache during streaming — it would invalidate KV cache.
        }
    }

    let elapsed = t0.elapsed();
    eprintln!();
    eprintln!("=== Final result ===");
    eprintln!("{}", last_text);
    eprintln!();
    eprintln!(
        "Timing: {:.2}s total, {} chunks, {:.2}s/chunk",
        elapsed.as_secs_f64(),
        num_chunks,
        elapsed.as_secs_f64() / num_chunks as f64
    );

    Ok(())
}
