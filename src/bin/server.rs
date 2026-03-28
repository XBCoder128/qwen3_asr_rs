use anyhow::{Context, Result};
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::Serialize;
use std::io::Write;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use qwen3_asr_rs::inference::AsrInference;
use qwen3_asr_rs::tensor::Device;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "asr-server", about = "OpenAI-compatible ASR API server for Qwen3-ASR")]
struct Args {
    /// Path to the Qwen3-ASR model directory
    #[arg(long)]
    model_dir: String,

    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Default language for transcription (e.g., chinese, english)
    #[arg(long)]
    language: Option<String>,

    /// Verbose output (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AppState {
    model: Arc<std::sync::Mutex<AsrInference>>,
    default_language: Option<String>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Serialize)]
struct VerboseTranscriptionResponse {
    task: String,
    language: String,
    duration: f64,
    text: String,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        tracing::error!("Request error: {:?}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": self.0.to_string(),
                    "type": "server_error",
                }
            })),
        )
            .into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn models_handler() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: "qwen3-asr".to_string(),
            object: "model".to_string(),
            owned_by: "qwen".to_string(),
        }],
    })
}

async fn transcribe_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, AppError> {
    let mut file_bytes: Option<(String, Vec<u8>)> = None;
    let mut language: Option<String> = None;
    let mut response_format = "json".to_string();

    // Parse multipart fields
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError(anyhow::anyhow!("Multipart error: {}", e)))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let filename = field.file_name().unwrap_or("audio.wav").to_string();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| AppError(anyhow::anyhow!("Failed to read file: {}", e)))?;
                file_bytes = Some((filename, bytes.to_vec()));
            }
            "language" => {
                let val = field
                    .text()
                    .await
                    .map_err(|e| AppError(anyhow::anyhow!("Failed to read language: {}", e)))?;
                if !val.is_empty() {
                    language = Some(val);
                }
            }
            "response_format" => {
                let val = field
                    .text()
                    .await
                    .map_err(|e| AppError(anyhow::anyhow!("Failed to read response_format: {}", e)))?;
                if !val.is_empty() {
                    response_format = val;
                }
            }
            _ => {
                // Accept and ignore: model, temperature, prompt, etc.
                let _ = field.bytes().await;
            }
        }
    }

    // Validate file
    let (filename, bytes) = file_bytes
        .ok_or_else(|| AppError(anyhow::anyhow!("Missing required field: file")))?;
    if bytes.is_empty() {
        return Err(AppError(anyhow::anyhow!("Uploaded file is empty")));
    }

    // Language: request field > CLI default > None
    let lang = language.or(state.default_language.clone());

    // Write to temp file, preserving extension for ffmpeg format detection
    let extension = Path::new(&filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav");
    let mut tmp = tempfile::Builder::new()
        .suffix(&format!(".{}", extension))
        .tempfile()
        .map_err(|e| AppError(anyhow::anyhow!("Failed to create temp file: {}", e)))?;
    tmp.write_all(&bytes)
        .map_err(|e| AppError(anyhow::anyhow!("Failed to write temp file: {}", e)))?;

    let tmp_path = tmp.into_temp_path();
    let tmp_path_str = tmp_path.to_string_lossy().to_string();

    // Run inference in blocking task (GPU-bound)
    let model = state.model.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _keep = tmp_path; // ensure temp file lives until transcription completes
        let model = model
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        model.transcribe(&tmp_path_str, lang.as_deref())
    })
    .await
    .map_err(|e| AppError(anyhow::anyhow!("Blocking task failed: {}", e)))??;

    // Format response
    match response_format.as_str() {
        "text" => Ok((StatusCode::OK, result.text).into_response()),
        "verbose_json" => Ok(Json(VerboseTranscriptionResponse {
            task: "transcribe".to_string(),
            language: result.language,
            duration: result.duration_seconds,
            text: result.text,
        })
        .into_response()),
        _ => Ok(Json(TranscriptionResponse { text: result.text }).into_response()),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let filter = match args.verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .init();

    // Verify model directory
    let model_dir = Path::new(&args.model_dir);
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", args.model_dir);
    }

    // Select device
    #[cfg(feature = "tch-backend")]
    let device = if tch::Cuda::is_available() {
        tracing::info!("Using CUDA device");
        Device::Gpu(0)
    } else {
        tracing::info!("Using CPU device");
        Device::Cpu
    };

    #[cfg(feature = "mlx")]
    let device = {
        qwen3_asr_rs::backend::mlx::stream::init_mlx(true);
        tracing::info!("Using MLX Metal GPU");
        Device::Gpu(0)
    };

    // Load model
    tracing::info!("Loading model from {:?}", model_dir);
    let model = AsrInference::load(model_dir, device).context("Failed to load model")?;
    tracing::info!("Model loaded successfully");

    let state = AppState {
        model: Arc::new(std::sync::Mutex::new(model)),
        default_language: args.language,
    };

    // Build router
    let app = Router::new()
        .route("/v1/audio/transcriptions", post(transcribe_handler))
        .route("/v1/models", get(models_handler))
        .route("/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .context("Invalid host:port")?;
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
