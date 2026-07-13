use anyhow::{Context, Result};
use std::path::Path;

use qwen3_asr_rs::align::AlignInference;
use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::tensor::Device;

fn print_usage() {
    eprintln!("Qwen3 ASR / ForcedAligner");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  asr <model_path> <audio_file> [language]");
    eprintln!("  asr align <aligner_model> <audio_file> <text_or_@file> <language>");
    eprintln!();
    eprintln!("Arguments (transcribe):");
    eprintln!("  model_path   Path to the Qwen3-ASR model directory");
    eprintln!("  audio_file   Path to the input audio file");
    eprintln!("  language     Optional: force language (e.g., chinese, english)");
    eprintln!();
    eprintln!("Arguments (align):");
    eprintln!("  aligner_model  Path to Qwen3-ForcedAligner model directory");
    eprintln!("  audio_file     Path to the input audio file");
    eprintln!("  text_or_@file  Transcript text, or @path to read text from a file");
    eprintln!("  language       Language for tokenization (e.g., Chinese, English)");
    eprintln!();
    eprintln!("Environment variables:");
    #[cfg(feature = "tch-backend")]
    eprintln!("  LIBTORCH     Path to libtorch installation");
    eprintln!("  RUST_LOG     Set logging level (e.g., info, debug, trace)");
}

fn select_device() -> Device {
    #[cfg(feature = "tch-backend")]
    {
        if tch::Cuda::is_available() {
            tracing::info!("Using CUDA device");
            Device::Gpu(0)
        } else {
            tracing::info!("Using CPU device");
            Device::Cpu
        }
    }

    #[cfg(feature = "mlx")]
    {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
        tracing::info!("Using MLX Metal GPU");
        Device::Gpu(0)
    }
}

fn resolve_text(arg: &str) -> Result<String> {
    if let Some(path) = arg.strip_prefix('@') {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read text file: {}", path))?;
        Ok(text.trim().to_string())
    } else {
        Ok(arg.to_string())
    }
}

fn run_align(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        print_usage();
        std::process::exit(1);
    }

    let model_path = &args[1];
    let audio_file = &args[2];
    let text = resolve_text(&args[3])?;
    let language = &args[4];

    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", model_path);
    }
    if !Path::new(audio_file).exists() {
        anyhow::bail!("Audio file not found: {}", audio_file);
    }

    let device = select_device();
    let model = AlignInference::load(model_dir, device).context("Failed to load ForcedAligner")?;

    tracing::info!("Aligning: {} ({})", audio_file, language);
    let result = model
        .align(audio_file, &text, language)
        .context("Alignment failed")?;

    println!("items: {}", result.items.len());
    for item in &result.items {
        println!(
            "{:.3}\t{:.3}\t{}",
            item.start_time, item.end_time, item.text
        );
    }

    Ok(())
}

fn run_transcribe(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let model_path = &args[1];
    let audio_file = &args[2];
    let language = args.get(3).map(|s| s.as_str());

    let model_dir = Path::new(model_path);
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", model_path);
    }
    if !Path::new(audio_file).exists() {
        anyhow::bail!("Audio file not found: {}", audio_file);
    }

    let device = select_device();
    let model = AsrInference::load(model_dir, device).context("Failed to load model")?;

    tracing::info!("Transcribing: {}", audio_file);
    let result = model
        .transcribe(audio_file, language)
        .context("Transcription failed")?;

    println!("Language: {}", result.language);
    println!("Text: {}", result.text);

    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    if args[1] == "align" {
        // asr align <model> <audio> <text> <language>
        run_align(&args[1..])
    } else {
        run_transcribe(&args)
    }
}
