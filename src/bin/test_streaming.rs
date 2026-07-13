//! Streaming vs offline transcription validation.
//!
//! Tests that incremental streaming transcription produces the same output
//! as offline (full audio) transcription.
//!
//! Usage:
//!   cargo run --release --no-default-features --features mlx --bin test_streaming -- <model_dir> <audio_file> [chunk_sec] [rollback_tokens]
//!
//! The test runs three modes:
//!   1. Offline: full audio → transcribe_samples (ground truth)
//!   2. Streaming (re-transcribe): chunks → transcribe_samples each time (O(n²))
//!   3. Streaming (incremental): chunks → streaming_transcribe (O(n), KV cache reuse)
//!
//! All three should produce identical text when rollback_tokens=0.

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
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Streaming validation test");
        eprintln!();
        eprintln!("Usage: test_streaming <model_dir> <audio_file> [chunk_sec]");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  model_dir   Path to Qwen3-ASR model directory");
        eprintln!("  audio_file  Path to test audio file");
        eprintln!("  chunk_sec   Chunk size in seconds (default: 2.0)");
        eprintln!("  rollback_tokens  Number of trailing tokens to re-decode (default: 0)");
        std::process::exit(1);
    }

    let model_dir = Path::new(&args[1]);
    let audio_file = &args[2];
    let chunk_sec: f64 = args.get(3).map(|s| s.parse().unwrap_or(2.0)).unwrap_or(2.0);
    let rollback: usize = args.get(4).map(|s| s.parse().unwrap_or(0)).unwrap_or(0);

    #[cfg(feature = "mlx")]
    {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
    }

    let device = Device::Gpu(0);

    eprintln!("[test] Loading model from {:?}", model_dir);
    let mut model = AsrInference::load(model_dir, device).context("Failed to load model")?;
    eprintln!("[test] Model loaded");

    let samples = qwen3_asr_rs::audio::load_audio(audio_file, SAMPLE_RATE as u32)
        .context("Failed to load audio")?;
    let total_samples = samples.len();
    let duration = total_samples as f64 / SAMPLE_RATE as f64;
    eprintln!(
        "[test] Audio: {} samples ({:.1}s) @ {}Hz → {:.1}s @ 16kHz",
        total_samples, duration, SAMPLE_RATE, duration
    );

    let chunk_size = (chunk_sec * SAMPLE_RATE as f64) as usize;
    let num_chunks = (total_samples + chunk_size - 1) / chunk_size;
    eprintln!(
        "[test] Chunk size: {} samples ({:.1}s), {} chunks, rollback={}",
        chunk_size, chunk_sec, num_chunks, rollback
    );

    // -----------------------------------------------------------------------
    // Step 1: Offline transcription (ground truth)
    // -----------------------------------------------------------------------
    eprintln!("\n[test] === Step 1: Offline transcription ===");
    let t0 = Instant::now();
    let offline_result = model
        .transcribe_samples(&samples, None)
        .context("Offline transcription failed")?;
    let offline_time = t0.elapsed();
    eprintln!("[test] Offline text:     \"{}\"", offline_result.text);
    eprintln!("[test] Offline language: {}", offline_result.language);
    eprintln!(
        "[test] Offline time:     {:.2}s",
        offline_time.as_secs_f64()
    );

    // -----------------------------------------------------------------------
    // Step 2: Streaming — full re-transcribe (O(n²), current approach)
    // -----------------------------------------------------------------------
    eprintln!("\n[test] === Step 2: Streaming (full re-transcribe) ===");
    let t0 = Instant::now();
    let mut last_retranscribe_text = String::new();
    let mut accumulated: Vec<f32> = Vec::new();

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, total_samples);
        accumulated.extend_from_slice(&samples[start..end]);

        if accumulated.len() < SAMPLE_RATE {
            continue;
        }

        let result = model
            .transcribe_samples(&accumulated, None)
            .with_context(|| format!("Re-transcribe failed at chunk {}", i + 1))?;

        eprintln!(
            "[test]   chunk {}: {:.1}s → \"{}\"",
            i + 1,
            accumulated.len() as f64 / SAMPLE_RATE as f64,
            result.text
        );
        last_retranscribe_text = result.text;

        #[cfg(feature = "mlx")]
        {
            qwen3_asr_rs::backend::mlx::stream::synchronize();
            qwen3_asr_rs::backend::mlx::stream::clear_cache();
        }
    }
    let retranscribe_time = t0.elapsed();
    eprintln!(
        "[test] Final re-transcribe text: \"{}\"",
        last_retranscribe_text
    );
    eprintln!(
        "[test] Re-transcribe time:       {:.2}s",
        retranscribe_time.as_secs_f64()
    );

    // -----------------------------------------------------------------------
    // Step 3: Streaming — incremental KV cache reuse (O(n), new approach)
    // -----------------------------------------------------------------------
    eprintln!("\n[test] === Step 3: Streaming (incremental KV cache) ===");
    let t0 = Instant::now();
    let mut stream_state = model.init_streaming(None, rollback)?;
    let mut last_incremental_text = String::new();
    let mut accumulated: Vec<f32> = Vec::new();

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, total_samples);
        accumulated.extend_from_slice(&samples[start..end]);

        if accumulated.len() < SAMPLE_RATE {
            continue;
        }

        let result = model
            .streaming_transcribe(&accumulated, &mut stream_state)
            .with_context(|| format!("Incremental streaming failed at chunk {}", i + 1))?;

        eprintln!(
            "[test]   chunk {}: {:.1}s → \"{}\"",
            i + 1,
            accumulated.len() as f64 / SAMPLE_RATE as f64,
            result.text
        );
        last_incremental_text = result.text;

        #[cfg(feature = "mlx")]
        {
            qwen3_asr_rs::backend::mlx::stream::synchronize();
            qwen3_asr_rs::backend::mlx::stream::clear_cache();
        }
    }
    let incremental_time = t0.elapsed();
    eprintln!(
        "[test] Final incremental text: \"{}\"",
        last_incremental_text
    );
    eprintln!(
        "[test] Incremental time:       {:.2}s",
        incremental_time.as_secs_f64()
    );

    // -----------------------------------------------------------------------
    // Step 4: Compare all three
    // -----------------------------------------------------------------------
    eprintln!("\n[test] === Step 4: Comparison ===");
    eprintln!("[test] Offline:       \"{}\"", offline_result.text);
    eprintln!("[test] Re-transcribe: \"{}\"", last_retranscribe_text);
    eprintln!("[test] Incremental:   \"{}\"", last_incremental_text);

    let pass1 = offline_result.text == last_retranscribe_text;
    let pass2 = offline_result.text == last_incremental_text;
    let pass3 = last_retranscribe_text == last_incremental_text;

    eprintln!();
    eprintln!(
        "[test] Offline == Re-transcribe: {}",
        if pass1 { "✅ PASS" } else { "❌ FAIL" }
    );
    eprintln!(
        "[test] Offline == Incremental:   {}",
        if pass2 { "✅ PASS" } else { "❌ FAIL" }
    );
    eprintln!(
        "[test] Re-transcribe == Incremental: {}",
        if pass3 { "✅ PASS" } else { "❌ FAIL" }
    );
    eprintln!();
    eprintln!(
        "[test] Timing: offline={:.2}s  re-transcribe={:.2}s  incremental={:.2}s  (speedup={:.1}x)",
        offline_time.as_secs_f64(),
        retranscribe_time.as_secs_f64(),
        incremental_time.as_secs_f64(),
        retranscribe_time.as_secs_f64() / incremental_time.as_secs_f64()
    );

    if pass1 && pass2 && pass3 {
        eprintln!("\n[test] 🎉 ALL PASS — incremental streaming produces identical results");
    } else {
        eprintln!("\n[test] ⚠️  Some checks failed — see above for details");
    }

    Ok(())
}
