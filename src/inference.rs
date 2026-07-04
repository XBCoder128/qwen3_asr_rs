use crate::tensor::{Device, Tensor};
use anyhow::{Context, Result};
use std::path::Path;

use crate::audio;
use crate::audio_encoder::AudioEncoder;
use crate::config::AsrConfig;
use crate::layers::compute_mrope_cos_sin;
use crate::mel::WhisperFeatureExtractor;
use crate::text_decoder::{create_causal_mask, KvCache, TextDecoder};
use crate::tokenizer::{AsrTokenizer, AUDIO_PAD_TOKEN_ID, ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID};
use crate::weights;

const MEL_SAMPLE_RATE: u32 = 16000;

/// ASR inference engine.
pub struct AsrInference {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    mel_extractor: WhisperFeatureExtractor,
    tokenizer: AsrTokenizer,
    config: AsrConfig,
    device: Device,
}

impl AsrInference {
    /// Load model from directory containing config.json, model.safetensors, tokenizer.json
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        tracing::info!("Loading model from {:?}", model_dir);

        // Load config
        let config = AsrConfig::from_file(&model_dir.join("config.json"))
            .context("Failed to load config")?;

        // Load weights (supports both single-file and sharded safetensors)
        let all_weights =
            weights::load_model_weights(model_dir, device).context("Failed to load weights")?;

        tracing::info!("Loaded {} weight tensors", all_weights.len());

        // Load audio encoder
        tracing::info!("Loading audio encoder...");
        let audio_encoder = AudioEncoder::load(
            &all_weights,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
            device,
        )
        .context("Failed to load audio encoder")?;

        // Load text decoder
        tracing::info!("Loading text decoder...");
        let text_decoder = TextDecoder::load(
            &all_weights,
            "thinker.model",
            &config.thinker_config.text_config,
        )
        .context("Failed to load text decoder")?;

        // Load tokenizer
        tracing::info!("Loading tokenizer...");
        let tokenizer = AsrTokenizer::from_dir(model_dir).context("Failed to load tokenizer")?;

        // Create mel feature extractor
        let mel_extractor = WhisperFeatureExtractor::new(
            400, // n_fft
            160, // hop_length
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
            device,
        );

        tracing::info!("Model loaded successfully");

        Ok(Self {
            audio_encoder,
            text_decoder,
            mel_extractor,
            tokenizer,
            config,
            device,
        })
    }

    /// Transcribe an audio file.
    pub fn transcribe(&self, audio_path: &str, language: Option<&str>) -> Result<TranscribeResult> {
        // Step 1: Load and preprocess audio
        tracing::info!("Loading audio from {}", audio_path);
        let samples = audio::load_audio(audio_path, MEL_SAMPLE_RATE)?;
        self.transcribe_samples(&samples, language)
    }

    /// Transcribe raw 16kHz mono f32 samples directly (no file I/O).
    pub fn transcribe_samples(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscribeResult> {
        let duration_seconds = samples.len() as f64 / MEL_SAMPLE_RATE as f64;

        // Step 2: Compute mel spectrogram
        let mel = self.mel_extractor.extract(samples, self.device)?;
        let num_mel_frames = mel.size()[1] as usize;
        tracing::info!("Mel spectrogram: {} frames", num_mel_frames);

        // Step 3: Run audio encoder
        let audio_embeds = self.audio_encoder.forward(&mel);
        audio_embeds.eval(); // Materialize encoder output before decode phase
        let num_audio_tokens = audio_embeds.size()[0] as usize;
        tracing::info!("Audio encoder: {} tokens", num_audio_tokens);

        // Step 4: Build input token sequence
        let (input_ids, audio_positions) = self.build_prompt(num_audio_tokens, language)?;
        let seq_len = input_ids.len();

        // Step 5: Build embeddings with audio injection
        let input_tensor = Tensor::from_slice_i64(&input_ids).to_device(self.device);
        let mut hidden_states = self.text_decoder.embed(&input_tensor).unsqueeze(0);

        // Replace audio_pad positions with audio encoder embeddings.
        // For long audio, this loop creates a deep computation graph in MLX's
        // lazy eval. We eval() periodically to break the chain and free
        // intermediate tensors — otherwise memory grows linearly with the
        // number of audio tokens.
        for (embed_idx, &seq_pos) in audio_positions.iter().enumerate() {
            let audio_embed = audio_embeds.get(embed_idx as i64);
            hidden_states = hidden_states.slice_scatter(
                &audio_embed.unsqueeze(0).unsqueeze(0),
                1,
                seq_pos as i64,
                seq_pos as i64 + 1,
                1,
            );
            // Materialize every 32 tokens to break the computation graph.
            if embed_idx % 32 == 31 {
                hidden_states.eval();
            }
        }
        hidden_states.eval(); // Final materialize

        // Step 6: Precompute MRoPE cos/sin for all positions (prefill + max decode)
        let text_config = &self.config.thinker_config.text_config;
        let max_new_tokens: usize = 256; // ASR rarely exceeds 256 tokens
                                         // Precompute enough positions for prefill + decode budget
        let max_total_positions = seq_len + max_new_tokens + 8;
        let all_positions: Vec<i64> = (0..max_total_positions as i64).collect();
        let all_pos_ids: [Vec<i64>; 3] =
            [all_positions.clone(), all_positions.clone(), all_positions];
        let (all_cos, all_sin) = compute_mrope_cos_sin(
            &all_pos_ids,
            text_config.head_dim,
            text_config.rope_theta,
            &text_config.mrope_section(),
            text_config.mrope_interleaved(),
            self.device,
        );

        // Prefill cos/sin: positions 0..seq_len
        let cos = all_cos.narrow(0, 0, seq_len as i64);
        let sin = all_sin.narrow(0, 0, seq_len as i64);

        // Step 7: Prefill
        let mask = create_causal_mask(seq_len as i64, 0, self.device);
        let mut kv_cache = KvCache::new(text_config.num_hidden_layers);

        let logits =
            self.text_decoder
                .forward(&hidden_states, &cos, &sin, &mut kv_cache, Some(&mask));
        // Eval prefill output to materialize computation graph before decode loop
        logits.eval();

        // Step 8: Autoregressive generation
        let mut generated_ids: Vec<i64> = Vec::new();
        let eos_token_ids = vec![ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        let mut next_logits = logits.narrow(1, seq_len as i64 - 1, 1).squeeze_dim(1);

        let mut current_pos = seq_len;

        for _ in 0..max_new_tokens {
            let next_token = next_logits.argmax(-1, false).int64_value(&[0]);

            if eos_token_ids.contains(&next_token) {
                break;
            }

            generated_ids.push(next_token);

            let next_input = Tensor::from_slice_i64(&[next_token]).to_device(self.device);
            let next_hidden = self.text_decoder.embed(&next_input).unsqueeze(0);

            // Index into precomputed cos/sin for this position
            let new_cos = all_cos.narrow(0, current_pos as i64, 1);
            let new_sin = all_sin.narrow(0, current_pos as i64, 1);

            // Single-token decode: causal mask is all-zeros (no masking needed)
            next_logits =
                self.text_decoder
                    .forward(&next_hidden, &new_cos, &new_sin, &mut kv_cache, None);
            next_logits = next_logits.squeeze_dim(1);
            // Materialize to prevent the decode loop from building a deep graph.
            next_logits.eval();

            current_pos += 1;
        }

        // Step 9: Parse output
        tracing::info!("Generated {} tokens", generated_ids.len());

        // Explicitly drop large intermediates before synchronize.
        drop(kv_cache);
        drop(all_cos);
        drop(all_sin);
        drop(hidden_states);
        drop(audio_embeds);
        drop(mel);

        // Flush the entire MLX computation graph so all intermediate tensors
        // are freed before we return. Critical for streaming where this
        // function is called repeatedly.
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::stream::synchronize();
            crate::backend::mlx::stream::clear_cache();
            eprintln!(
                "[mlx] memory after clear: active={}MB cache={}MB",
                crate::backend::mlx::stream::active_memory() / 1_000_000,
                crate::backend::mlx::stream::cache_memory() / 1_000_000,
            );
        }

        let raw_text = self.tokenizer.decode(&generated_ids)?;
        tracing::debug!("Raw output: {:?}", raw_text);
        let (language_detected, transcription) = parse_asr_output(&raw_text, language.is_some());

        Ok(TranscribeResult {
            text: transcription,
            language: language_detected,
            raw_output: raw_text,
            duration_seconds,
        })
    }

    fn build_prompt(
        &self,
        num_audio_tokens: usize,
        language: Option<&str>,
    ) -> Result<(Vec<i64>, Vec<usize>)> {
        let mut tokens: Vec<i64> = vec![
            151644, // <|im_start|>
            8948,   // system
            198,    // \n
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
            872,    // user
            198,    // \n
            151669, // <|audio_start|>
        ];

        let audio_start_pos = tokens.len();
        for _ in 0..num_audio_tokens {
            tokens.push(AUDIO_PAD_TOKEN_ID);
        }
        let audio_positions: Vec<usize> =
            (audio_start_pos..audio_start_pos + num_audio_tokens).collect();

        tokens.extend_from_slice(&[
            151670, // <|audio_end|>
            151645, // <|im_end|>
            198,    // \n
            151644, // <|im_start|>
        ]);

        if let Some(lang) = language {
            tokens.push(77091); // assistant
            tokens.push(198); // \n
            let prefix = format!("language {}", capitalize_first(lang));
            tokens.extend(self.tokenizer.encode(&prefix)?);
        } else {
            tokens.push(77091); // assistant
            tokens.push(198); // \n
        }

        Ok((tokens, audio_positions))
    }
}

/// Result of ASR transcription.
pub struct TranscribeResult {
    pub text: String,
    pub language: String,
    pub raw_output: String,
    pub duration_seconds: f64,
}

// ===========================================================================
// Streaming inference — incremental KV cache reuse
// ===========================================================================
//
// Instead of re-transcribing the full audio each time (O(n²)), we reuse the
// KV cache from previous iterations. The key insight:
//
// With causal attention, K/V at position i only depend on inputs at positions
// 0..=i. Adding new tokens at later positions does NOT invalidate existing KV.
//
// Prompt layout:
//   [pre_audio(9)] [audio(N)] [post_audio(6)]
//    positions 0-8   9..9+N-1   9+N..14+N
//
// When N grows from N_old to N_new:
//   1. KV for positions 0..9+N_old-1 is still valid (causal attention)
//   2. Truncate KV cache to 9+N_old (remove old post_audio tokens)
//   3. Prefill new audio tokens at positions 9+N_old..9+N_new-1
//   4. Prefill post_audio tokens at positions 9+N_new..9+N_new+5
//   5. Decode autoregressively
//
// Cost per iteration: O(ΔN + 6) instead of O(N).

/// Number of tokens before audio: im_start system \n im_end \n im_start user \n audio_start
const PRE_AUDIO_TOKEN_COUNT: usize = 9;
/// Tokens after audio: audio_end im_end \n im_start assistant \n
const POST_AUDIO_TOKENS: &[i64] = &[
    151670, // <|audio_end|>
    151645, // <|im_end|>
    198,    // \n
    151644, // <|im_start|>
    77091,  // assistant
    198,    // \n
];

/// State for incremental streaming ASR.
pub struct StreamingState {
    /// KV cache from previous iteration (None on first call).
    kv_cache: Option<KvCache>,
    /// Number of audio tokens already encoded and in the KV cache.
    audio_tokens_in_cache: usize,
    /// Total sequence length currently in the KV cache.
    cache_seq_len: i64,
    /// Precomputed MRoPE cos/sin for all positions.
    cos: Tensor,
    sin: Tensor,
    /// Pre-audio token embeddings (computed once, reused).
    pre_audio_embeds: Tensor,
    /// Cached audio embeddings from the previous call. Reused to avoid
    /// Conv2d batch-size-dependent numerical differences that would
    /// invalidate the cached KV.
    cached_audio_embeds: Option<Tensor>,
    /// Number of mel frames already encoded (for incremental encoding).
    cached_mel_frames: usize,
    /// Token IDs generated in the previous call (for prefix rollback).
    last_generated_ids: Vec<i64>,
    /// Number of trailing tokens to re-decode each call. The preceding
    /// tokens are kept as a prefix prompt (re-prefilled, not re-decoded).
    /// 0 = re-decode all tokens every call (default, exact match).
    /// N = only re-decode the last N tokens, prefix the rest.
    rollback_tokens: usize,
    /// Forced language (e.g. "english", "chinese"). None = auto-detect.
    forced_language: Option<String>,
    /// Token IDs for the language prefix (e.g. "language English").
    /// Computed once in init_streaming, reused in every prefill.
    language_prefix_ids: Vec<i64>,
    /// Device for tensor creation.
    device: Device,
}

impl AsrInference {
    /// Initialize streaming state. Call once before the first
    /// `streaming_transcribe` call.
    ///
    /// `rollback_tokens`: number of trailing generated tokens to re-decode
    /// each call. 0 = re-decode all (exact match with offline). N = keep
    /// earlier tokens as prefix, only re-decode last N. This speeds up
    /// streaming at the cost of potentially slightly different output at
    /// the rollback boundary.
    pub fn init_streaming(
        &mut self,
        language: Option<&str>,
        rollback_tokens: usize,
    ) -> Result<StreamingState> {
        // Enable chunk-local attention for stable incremental encoding.
        self.audio_encoder.set_chunk_local(true);
        let text_config = &self.config.thinker_config.text_config;

        // Compute language prefix token IDs if language is forced.
        // Format: "language English" (matches build_prompt in offline mode).
        let (forced_language, language_prefix_ids) = if let Some(lang) = language {
            let prefix = format!("language {}", capitalize_first(lang));
            let ids = self.tokenizer.encode(&prefix)?;
            (Some(lang.to_string()), ids)
        } else {
            (None, Vec::new())
        };

        // Pre-audio token IDs (constant).
        let pre_audio_ids: Vec<i64> =
            vec![151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669];

        // Pre-allocate cos/sin for generous position budget.
        // Must be large enough for pre_audio(9) + max_audio_tokens +
        // post_audio(6) + language_prefix + max_generated_text.
        // For 120s audio: ~1560 audio tokens + 9 + 6 + ~500 text = ~2075.
        let max_positions = 4096i64;
        let positions: Vec<i64> = (0..max_positions).collect();
        let pos_ids: [Vec<i64>; 3] = [positions.clone(), positions.clone(), positions];
        let (cos, sin) = compute_mrope_cos_sin(
            &pos_ids,
            text_config.head_dim,
            text_config.rope_theta,
            &text_config.mrope_section(),
            text_config.mrope_interleaved(),
            self.device,
        );

        // Embed pre-audio tokens once.
        let pre_tensor = Tensor::from_slice_i64(&pre_audio_ids).to_device(self.device);
        let pre_audio_embeds = self.text_decoder.embed(&pre_tensor).unsqueeze(0);
        pre_audio_embeds.eval();

        Ok(StreamingState {
            kv_cache: None,
            audio_tokens_in_cache: 0,
            cache_seq_len: 0,
            cos,
            sin,
            pre_audio_embeds,
            cached_audio_embeds: None,
            cached_mel_frames: 0,
            last_generated_ids: Vec::new(),
            rollback_tokens,
            forced_language,
            language_prefix_ids,
            device: self.device,
        })
    }

    /// Incremental streaming transcription.
    ///
    /// On the first call, does a full prefill. On subsequent calls, reuses
    /// the KV cache and only prefills the new audio tokens + post-audio
    /// tokens + optional prefix text. Returns the transcription result.
    pub fn streaming_transcribe(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
    ) -> Result<TranscribeResult> {
        self.streaming_transcribe_with_tail(samples, state, true)
    }

    /// Streaming transcription without processing tail frames.
    /// Only complete encoder chunks are processed. The remaining frames
    /// are deferred to the next call or `streaming_transcribe` (which
    /// processes everything including tail).
    pub fn streaming_transcribe_partial(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
    ) -> Result<TranscribeResult> {
        self.streaming_transcribe_with_tail(samples, state, false)
    }

    fn streaming_transcribe_with_tail(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        allow_tail: bool,
    ) -> Result<TranscribeResult> {
        let duration_seconds = samples.len() as f64 / MEL_SAMPLE_RATE as f64;
        let chunk_size = self.audio_encoder.chunk_size();
        let tokens_per_chunk = self.audio_encoder.tokens_per_chunk();

        let t_start = std::time::Instant::now();

        // Step 1: Compute mel
        let mel = self.mel_extractor.extract(samples, self.device)?;
        let total_mel_frames = mel.size()[1] as usize;

        // Determine how many frames to process.
        // - allow_tail=true (final): process all frames.
        // - allow_tail=false (partial): process up to the last complete chunk
        //   boundary, OR all frames if there's no tail (naturally aligned).
        let full_frames = (total_mel_frames / chunk_size) * chunk_size;
        let mel_frames_to_use = if allow_tail {
            total_mel_frames
        } else {
            full_frames
        };

        // Skip if no new frames to process.
        if mel_frames_to_use == 0 || (mel_frames_to_use <= state.cached_mel_frames && !allow_tail) {
            // Nothing new to encode. Return empty result.
            return Ok(TranscribeResult {
                text: String::new(),
                language: String::new(),
                raw_output: String::new(),
                duration_seconds,
            });
        }

        let mel = if mel_frames_to_use < total_mel_frames {
            mel.narrow(1, 0, mel_frames_to_use as i64)
        } else {
            mel
        };
        let total_mel_frames = mel_frames_to_use;

        // Step 2: Incremental encoding.
        //
        // cached_audio_embeds holds embeddings for frames [0..cached_mel_frames)
        // which are all complete chunks. We encode frames
        // [cached_mel_frames..total_mel_frames) (which may include the old
        // tail being re-encoded + new frames), concatenate, then update the
        // cache to the new complete-chunk boundary.
        let audio_embeds =
            if state.cached_mel_frames > 0 && state.cached_mel_frames < total_mel_frames {
                let new_frame_start = state.cached_mel_frames;
                let new_frame_count = total_mel_frames - new_frame_start;
                let new_mel = mel.narrow(1, new_frame_start as i64, new_frame_count as i64);
                let new_embeds = self.audio_encoder.forward(&new_mel);
                new_embeds.eval();

                let cached = state.cached_audio_embeds.as_ref().unwrap();
                let combined = Tensor::cat(&[cached.shallow_clone(), new_embeds], 0);
                combined.eval();

                // Update cache to the new complete-chunk boundary.
                let new_full_frames = (total_mel_frames / chunk_size) * chunk_size;
                let new_full_chunks = new_full_frames / chunk_size;
                let new_full_tokens = new_full_chunks * tokens_per_chunk;
                state.cached_audio_embeds = Some(combined.narrow(0, 0, new_full_tokens as i64));
                state.cached_mel_frames = new_full_frames;
                combined
            } else {
                let embeds = self.audio_encoder.forward(&mel);
                embeds.eval();
                let new_full_frames = (total_mel_frames / chunk_size) * chunk_size;
                let new_full_chunks = new_full_frames / chunk_size;
                let new_full_tokens = new_full_chunks * tokens_per_chunk;
                state.cached_audio_embeds = Some(embeds.narrow(0, 0, new_full_tokens as i64));
                state.cached_mel_frames = new_full_frames;
                embeds
            };
        let num_audio_tokens = audio_embeds.size()[0] as usize;
        let t_encode = t_start.elapsed();

        tracing::info!(
            "[streaming] audio_tokens={} (was_cached={}) delta={}",
            num_audio_tokens,
            state.audio_tokens_in_cache,
            num_audio_tokens as i64 - state.audio_tokens_in_cache as i64
        );

        let text_config = &self.config.thinker_config.text_config;

        let t_prefill_start = std::time::Instant::now();
        let logits = if state.kv_cache.is_none() {
            // First call: full prefill
            self.streaming_full_prefill(&audio_embeds, num_audio_tokens, state, text_config)?
        } else {
            // Incremental: take KV cache out of state to avoid double borrow
            let mut kv_cache = state.kv_cache.take().unwrap();
            let result = self.streaming_incremental_prefill(
                &audio_embeds,
                num_audio_tokens,
                &mut kv_cache,
                state,
                text_config,
            );
            state.kv_cache = Some(kv_cache);
            result?
        };
        let t_prefill = t_prefill_start.elapsed();

        // Step 4: Autoregressive decode
        let kv_cache = state.kv_cache.as_mut().unwrap();
        let seq_len = state.cache_seq_len;
        let max_new_tokens: usize = 256;
        let eos_token_ids = vec![ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID];

        // logits has shape (1, num_prefilled, vocab_size). We need the last one.
        let logits_seq_len = logits.size()[1];
        let mut next_logits = logits.narrow(1, logits_seq_len - 1, 1).squeeze_dim(1);
        let mut current_pos = seq_len;

        // When rollback_tokens > 0, the prefix text was already prefilled in
        // streaming_incremental_prefill. We start with those prefix IDs and
        // only decode new tokens.
        let prefix_len = state
            .last_generated_ids
            .len()
            .saturating_sub(state.rollback_tokens);
        let prefix_ids: Vec<i64> = if state.rollback_tokens > 0 && prefix_len > 0 {
            state.last_generated_ids[..prefix_len].to_vec()
        } else {
            Vec::new()
        };
        let mut generated_ids = prefix_ids.clone();

        for _ in 0..max_new_tokens {
            let next_token = next_logits.argmax(-1, false).int64_value(&[0]);
            if eos_token_ids.contains(&next_token) {
                break;
            }
            generated_ids.push(next_token);

            let next_input = Tensor::from_slice_i64(&[next_token]).to_device(state.device);
            let next_hidden = self.text_decoder.embed(&next_input).unsqueeze(0);
            let new_cos = state.cos.narrow(0, current_pos, 1);
            let new_sin = state.sin.narrow(0, current_pos, 1);

            next_logits =
                self.text_decoder
                    .forward(&next_hidden, &new_cos, &new_sin, kv_cache, None);
            next_logits = next_logits.squeeze_dim(1);
            next_logits.eval();
            current_pos += 1;
        }

        // Post-decode: detect and trim long repeated subsequences.
        // When rollback > 0, the decoder may copy chunks of the prefix.
        // This catches and removes such repetitions.
        trim_repeated_tail(&mut generated_ids);

        tracing::info!(
            "Streaming: generated {} tokens (prefix={}, new={})",
            generated_ids.len(),
            prefix_ids.len(),
            generated_ids.len() - prefix_ids.len()
        );

        // Save generated IDs for next call's prefix rollback.
        state.last_generated_ids = generated_ids.clone();

        let t_total = t_start.elapsed();
        tracing::info!(
            "[streaming] timing: encode={:.0}ms prefill={:.0}ms decode={:.0}ms total={:.0}ms",
            t_encode.as_millis(),
            t_prefill.as_millis(),
            (t_total - t_encode - t_prefill).as_millis(),
            t_total.as_millis()
        );

        // Cleanup
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::stream::synchronize();
            crate::backend::mlx::stream::clear_cache();
        }

        let raw_text = self.tokenizer.decode(&generated_ids)?;
        let (language_detected, transcription) =
            parse_asr_output(&raw_text, state.forced_language.is_some());

        // When language is forced, use the forced language name.
        let language = state.forced_language.clone().unwrap_or(language_detected);

        Ok(TranscribeResult {
            text: transcription,
            language,
            raw_output: raw_text,
            duration_seconds,
        })
    }

    /// First call: full prefill from scratch. Returns logits for all positions.
    fn streaming_full_prefill(
        &self,
        audio_embeds: &Tensor,
        num_audio_tokens: usize,
        state: &mut StreamingState,
        text_config: &crate::config::TextDecoderConfig,
    ) -> Result<Tensor> {
        let seq_len = PRE_AUDIO_TOKEN_COUNT
            + num_audio_tokens
            + POST_AUDIO_TOKENS.len()
            + state.language_prefix_ids.len();

        // Build hidden states: pre_audio + audio + post_audio + language_prefix
        let mut hidden_states = state.pre_audio_embeds.shallow_clone();

        // Inject audio embeddings
        for i in 0..num_audio_tokens {
            let embed = audio_embeds.get(i as i64);
            hidden_states = hidden_states.slice_scatter(
                &embed.unsqueeze(0).unsqueeze(0),
                1,
                (PRE_AUDIO_TOKEN_COUNT + i) as i64,
                (PRE_AUDIO_TOKEN_COUNT + i + 1) as i64,
                1,
            );
            if i % 32 == 31 {
                hidden_states.eval();
            }
        }

        // Append post-audio tokens
        let post_tensor = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let post_embeds = self.text_decoder.embed(&post_tensor).unsqueeze(0);
        hidden_states = Tensor::cat(&[hidden_states, post_embeds], 1);

        // Append language prefix tokens (e.g. "language English")
        if !state.language_prefix_ids.is_empty() {
            let lang_tensor =
                Tensor::from_slice_i64(&state.language_prefix_ids).to_device(state.device);
            let lang_embeds = self.text_decoder.embed(&lang_tensor).unsqueeze(0);
            hidden_states = Tensor::cat(&[hidden_states, lang_embeds], 1);
        }
        hidden_states.eval();

        // Cos/sin + mask for full prefill
        let cos = state.cos.narrow(0, 0, seq_len as i64);
        let sin = state.sin.narrow(0, 0, seq_len as i64);
        let mask = create_causal_mask(seq_len as i64, 0, state.device);

        let mut kv_cache = KvCache::new(text_config.num_hidden_layers);
        let logits =
            self.text_decoder
                .forward(&hidden_states, &cos, &sin, &mut kv_cache, Some(&mask));
        logits.eval();

        state.kv_cache = Some(kv_cache);
        // Track only complete-chunk token count (excluding tail) so that
        // old tail KV gets truncated when tail becomes a full chunk.
        let tpc = self.audio_encoder.tokens_per_chunk();
        let fchunks = num_audio_tokens / tpc;
        state.audio_tokens_in_cache = fchunks * tpc;
        state.cache_seq_len = seq_len as i64;

        Ok(logits)
    }

    /// Incremental call: reuse KV cache, only prefill new tokens.
    /// Returns logits for the last position.
    fn streaming_incremental_prefill(
        &self,
        audio_embeds: &Tensor,
        num_audio_tokens: usize,
        kv_cache: &mut KvCache,
        state: &mut StreamingState,
        text_config: &crate::config::TextDecoderConfig,
    ) -> Result<Tensor> {
        let old_audio = state.audio_tokens_in_cache;
        let _ = text_config;

        tracing::debug!(
            "[streaming-inc] old_audio={} new_audio={} cache_seq_len={}",
            old_audio,
            num_audio_tokens,
            state.cache_seq_len
        );

        // Always truncate to pre_audio + old_audio (complete chunks only).
        // This removes old tail KV + post-audio + generated text,
        // so we can re-prefill with correct embeddings.
        let truncate_len = (PRE_AUDIO_TOKEN_COUNT + old_audio) as i64;
        // Safety: clamp truncate_len to the actual KV cache length.
        let actual_cache_len = kv_cache.seq_len();
        if truncate_len > actual_cache_len {
            tracing::warn!(
                "[streaming-inc] truncate_len={} > actual_cache_len={}! Clamping. old_audio={} num_audio_tokens={} cache_seq_len={}",
                truncate_len, actual_cache_len, old_audio, num_audio_tokens, state.cache_seq_len
            );
        }
        let truncate_len = truncate_len.min(actual_cache_len);
        // Force materialize before truncate to avoid stale lazy graph.
        for layer in &kv_cache.layers {
            if let Some((k, v)) = layer {
                k.eval();
                v.eval();
            }
        }
        kv_cache.truncate(truncate_len);
        state.cache_seq_len = truncate_len;

        // Number of new audio tokens to prefill (new complete chunks + tail).
        let num_new_i64 = num_audio_tokens as i64 - old_audio as i64;
        if num_new_i64 <= 0 {
            // No new audio at all — just re-prefill post-audio + prefix.
            return self.prefill_post_audio(kv_cache, state, num_audio_tokens);
        }

        // Step 2+3: Prefill new audio tokens + post-audio tokens + prefix
        // text in a single forward call. This avoids potential numerical
        // differences from splitting the prefill into multiple calls.
        let num_new = num_audio_tokens - old_audio;
        let pos_start = (PRE_AUDIO_TOKEN_COUNT + old_audio) as i64;

        // Compute prefix text IDs (rollback strategy).
        let prefix_ids: Vec<i64> = if state.rollback_tokens > 0 {
            let last = &state.last_generated_ids;
            let prefix_len = last.len().saturating_sub(state.rollback_tokens);
            if prefix_len > 0 {
                last[..prefix_len].to_vec()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let total_new =
            num_new + POST_AUDIO_TOKENS.len() + state.language_prefix_ids.len() + prefix_ids.len();

        // Verify all KV cache layers have the same length before prefill.
        let cache_lens: Vec<i64> = kv_cache
            .layers
            .iter()
            .map(|l| l.as_ref().map(|(k, _)| k.size()[2]).unwrap_or(0))
            .collect();
        let max_len = *cache_lens.iter().max().unwrap_or(&0);
        let min_len = *cache_lens.iter().min().unwrap_or(&0);
        if max_len != min_len {
            tracing::warn!(
                "[streaming-inc] KV cache layer length mismatch! min={} max={} truncate_len={} total_new={}",
                min_len, max_len, truncate_len, total_new
            );
        }

        tracing::debug!(
            "[streaming-inc] truncate_len={} num_new={} pos_start={} total_new={} prefix={} lang={}",
            truncate_len,
            num_new,
            pos_start,
            total_new,
            prefix_ids.len(),
            state.language_prefix_ids.len()
        );

        // Build hidden states: new audio embeds + post-audio embeds + language prefix + rollback prefix text embeds
        let mut parts: Vec<Tensor> = Vec::with_capacity(total_new);
        for i in old_audio..num_audio_tokens {
            parts.push(audio_embeds.get(i as i64).unsqueeze(0).unsqueeze(0));
        }
        let post_tensor = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let post_embeds = self.text_decoder.embed(&post_tensor).unsqueeze(0);
        parts.push(post_embeds);
        if !state.language_prefix_ids.is_empty() {
            let lang_tensor =
                Tensor::from_slice_i64(&state.language_prefix_ids).to_device(state.device);
            let lang_embeds = self.text_decoder.embed(&lang_tensor).unsqueeze(0);
            parts.push(lang_embeds);
        }
        if !prefix_ids.is_empty() {
            let prefix_tensor = Tensor::from_slice_i64(&prefix_ids).to_device(state.device);
            let prefix_embeds = self.text_decoder.embed(&prefix_tensor).unsqueeze(0);
            parts.push(prefix_embeds);
        }
        let new_hidden = Tensor::cat(&parts, 1); // (1, total_new, dim)

        let cos = state.cos.narrow(0, pos_start, total_new as i64);
        let sin = state.sin.narrow(0, pos_start, total_new as i64);
        let mask = create_causal_mask(total_new as i64, truncate_len, state.device);

        // Pre-forward diagnostics.
        tracing::info!(
            "[streaming-inc] pre-forward: total_new={} truncate_len={} kv_len={} mask_dims={:?} hidden_dims={:?}",
            total_new,
            truncate_len,
            kv_cache.seq_len(),
            mask.size(),
            new_hidden.size()
        );

        let logits = self
            .text_decoder
            .forward(&new_hidden, &cos, &sin, kv_cache, Some(&mask));
        logits.eval();

        // Verify KV cache consistency after forward.
        let post_lens: Vec<i64> = kv_cache
            .layers
            .iter()
            .map(|l| l.as_ref().map(|(k, _)| k.size()[2]).unwrap_or(0))
            .collect();
        let post_max = *post_lens.iter().max().unwrap_or(&0);
        let post_min = *post_lens.iter().min().unwrap_or(&0);
        if post_max != post_min {
            tracing::warn!(
                "[streaming-inc] POST-FORWARD KV mismatch! min={} max={} truncate={} total_new={}",
                post_min,
                post_max,
                truncate_len,
                total_new
            );
        }

        state.cache_seq_len += total_new as i64;
        let tpc = self.audio_encoder.tokens_per_chunk();
        let fchunks = num_audio_tokens / tpc;
        state.audio_tokens_in_cache = fchunks * tpc;

        Ok(logits)
    }

    /// Diagnostic: compare incremental prefill logits against a fresh
    /// full prefill. Logs max abs diff per layer's K cache and the final
    /// logits. Does NOT modify `state`.
    #[allow(clippy::too_many_arguments)]
    fn verify_incremental_vs_fresh(
        &self,
        audio_embeds: &Tensor,
        num_audio_tokens: usize,
        state: &StreamingState,
        text_config: &crate::config::TextDecoderConfig,
        inc_logits: &Tensor,
    ) {
        // Build a fresh full prefill in a temporary state clone.
        let seq_len = PRE_AUDIO_TOKEN_COUNT + num_audio_tokens + POST_AUDIO_TOKENS.len();

        let mut hidden_states = state.pre_audio_embeds.shallow_clone();
        for i in 0..num_audio_tokens {
            let embed = audio_embeds.get(i as i64);
            hidden_states = hidden_states.slice_scatter(
                &embed.unsqueeze(0).unsqueeze(0),
                1,
                (PRE_AUDIO_TOKEN_COUNT + i) as i64,
                (PRE_AUDIO_TOKEN_COUNT + i + 1) as i64,
                1,
            );
            if i % 32 == 31 {
                hidden_states.eval();
            }
        }
        let post_tensor = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let post_embeds = self.text_decoder.embed(&post_tensor).unsqueeze(0);
        hidden_states = Tensor::cat(&[hidden_states, post_embeds], 1);
        hidden_states.eval();

        let cos = state.cos.narrow(0, 0, seq_len as i64);
        let sin = state.sin.narrow(0, 0, seq_len as i64);
        let mask = create_causal_mask(seq_len as i64, 0, state.device);

        let mut fresh_cache = KvCache::new(text_config.num_hidden_layers);
        let fresh_logits =
            self.text_decoder
                .forward(&hidden_states, &cos, &sin, &mut fresh_cache, Some(&mask));
        fresh_logits.eval();

        // Compare logits: incremental returned (1, post_len, vocab) for the
        // last segment only; fresh returned (1, seq_len, vocab). Compare the
        // last position of each.
        let inc_last = inc_logits
            .narrow(1, inc_logits.size()[1] - 1, 1)
            .squeeze_dim(1);
        let fresh_last = fresh_logits
            .narrow(1, fresh_logits.size()[1] - 1, 1)
            .squeeze_dim(1);
        let diff = (&inc_last - &fresh_last).abs();
        let max_diff = diff.max().f64_value(&[]);

        // Also compare per-layer K cache at the first truncate_len positions.
        let inc_cache = state.kv_cache.as_ref().unwrap();
        let mut layer_max_diffs: Vec<f64> = Vec::new();
        for i in 0..text_config.num_hidden_layers {
            let (inc_k, _inc_v) = inc_cache.get(i).unwrap();
            let (fresh_k, _fresh_v) = fresh_cache.get(i).unwrap();
            // Both should have the same seq_len now.
            let cmp_len = inc_k.size()[2].min(fresh_k.size()[2]);
            let inc_k_part = inc_k.narrow(2, 0, cmp_len);
            let fresh_k_part = fresh_k.narrow(2, 0, cmp_len);
            let kdiff = (&inc_k_part - &fresh_k_part).abs().max().f64_value(&[]);
            layer_max_diffs.push(kdiff);
        }
        let k_max = layer_max_diffs.iter().cloned().fold(0.0f64, f64::max);

        // Per-segment K cache diff on layer 0: pre_audio / old_audio / new_audio / post_audio.
        let (inc_k0, _) = inc_cache.get(0).unwrap();
        let (fresh_k0, _) = fresh_cache.get(0).unwrap();
        let seg_diff = |start: i64, end: i64| -> f64 {
            if end <= start {
                return 0.0;
            }
            let len = end - start;
            let inc_seg = inc_k0.narrow(2, start, len);
            let fresh_seg = fresh_k0.narrow(2, start, len);
            (&inc_seg - &fresh_seg).abs().max().f64_value(&[])
        };
        let pre_diff = seg_diff(0, PRE_AUDIO_TOKEN_COUNT as i64);
        let old_audio_diff = seg_diff(
            PRE_AUDIO_TOKEN_COUNT as i64,
            (PRE_AUDIO_TOKEN_COUNT + state.audio_tokens_in_cache) as i64,
        );
        let total_len = inc_k0.size()[2];
        let seg_total_diff = seg_diff(0, total_len);

        // argmax of last position
        let inc_argmax = inc_last.argmax(-1, false).int64_value(&[0]);
        let fresh_argmax = fresh_last.argmax(-1, false).int64_value(&[0]);

        tracing::warn!(
            "[verify] logits_max_diff={:.3e} k_cache_max_diff={:.3e} inc_argmax={} fresh_argmax={} match={}",
            max_diff,
            k_max,
            inc_argmax,
            fresh_argmax,
            inc_argmax == fresh_argmax
        );
        tracing::warn!(
            "[verify] layer0 K seg diffs: pre_audio={:.3e} old_audio={:.3e} total={:.3e} (cache_len={} fresh_len={})",
            pre_diff,
            old_audio_diff,
            seg_total_diff,
            total_len,
            fresh_k0.size()[2]
        );

        // Free the fresh computation graph.
        drop(fresh_cache);
        drop(fresh_logits);
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::stream::synchronize();
            crate::backend::mlx::stream::clear_cache();
        }
    }

    /// Diagnostic: compare truncated K cache against fresh full prefill K
    /// at positions 0..truncate_len. Run BEFORE any new prefill.
    fn verify_truncated_k(
        &self,
        audio_embeds: &Tensor,
        num_audio_tokens: usize,
        kv_cache: &KvCache,
        state: &StreamingState,
        text_config: &crate::config::TextDecoderConfig,
        truncate_len: i64,
    ) {
        let seq_len = PRE_AUDIO_TOKEN_COUNT + num_audio_tokens + POST_AUDIO_TOKENS.len();
        let mut hidden_states = state.pre_audio_embeds.shallow_clone();
        for i in 0..num_audio_tokens {
            let embed = audio_embeds.get(i as i64);
            hidden_states = hidden_states.slice_scatter(
                &embed.unsqueeze(0).unsqueeze(0),
                1,
                (PRE_AUDIO_TOKEN_COUNT + i) as i64,
                (PRE_AUDIO_TOKEN_COUNT + i + 1) as i64,
                1,
            );
            if i % 32 == 31 {
                hidden_states.eval();
            }
        }
        let post_tensor = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let post_embeds = self.text_decoder.embed(&post_tensor).unsqueeze(0);
        hidden_states = Tensor::cat(&[hidden_states, post_embeds], 1);
        hidden_states.eval();
        let cos = state.cos.narrow(0, 0, seq_len as i64);
        let sin = state.sin.narrow(0, 0, seq_len as i64);
        let mask = create_causal_mask(seq_len as i64, 0, state.device);
        let mut fresh_cache = KvCache::new(text_config.num_hidden_layers);
        let fresh_logits =
            self.text_decoder
                .forward(&hidden_states, &cos, &sin, &mut fresh_cache, Some(&mask));
        fresh_logits.eval();

        let (inc_k, _) = kv_cache.get(0).unwrap();
        let (fresh_k, _) = fresh_cache.get(0).unwrap();
        let inc_len = inc_k.size()[2];
        let fresh_len = fresh_k.size()[2];

        let seg_diff = |start: i64, end: i64| -> f64 {
            if end <= start || end > inc_len.min(fresh_len) {
                return -1.0;
            }
            let len = end - start;
            let inc_seg = inc_k.narrow(2, start, len);
            let fresh_seg = fresh_k.narrow(2, start, len);
            (&inc_seg - &fresh_seg).abs().max().f64_value(&[])
        };

        let pre_d = seg_diff(0, PRE_AUDIO_TOKEN_COUNT as i64);
        let old_d = seg_diff(PRE_AUDIO_TOKEN_COUNT as i64, truncate_len);
        let all_d = seg_diff(0, inc_len.min(fresh_len));

        // CRITICAL TEST: compare fresh prefill K at positions 9..truncate_len
        // against a SEPARATE fresh prefill using ONLY old_audio tokens (not
        // the full num_audio_tokens). If K values differ, it means K depends
        // on sequence length — which would invalidate incremental prefill.
        let old_audio_count = (truncate_len - PRE_AUDIO_TOKEN_COUNT as i64) as usize;
        let short_seq_len = PRE_AUDIO_TOKEN_COUNT + old_audio_count + POST_AUDIO_TOKENS.len();
        let mut short_hidden = state.pre_audio_embeds.shallow_clone();
        for i in 0..old_audio_count {
            let embed = audio_embeds.get(i as i64);
            short_hidden = short_hidden.slice_scatter(
                &embed.unsqueeze(0).unsqueeze(0),
                1,
                (PRE_AUDIO_TOKEN_COUNT + i) as i64,
                (PRE_AUDIO_TOKEN_COUNT + i + 1) as i64,
                1,
            );
        }
        let short_post = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let short_post_embeds = self.text_decoder.embed(&short_post).unsqueeze(0);
        short_hidden = Tensor::cat(&[short_hidden, short_post_embeds], 1);
        short_hidden.eval();
        let short_cos = state.cos.narrow(0, 0, short_seq_len as i64);
        let short_sin = state.sin.narrow(0, 0, short_seq_len as i64);
        let short_mask = create_causal_mask(short_seq_len as i64, 0, state.device);
        let mut short_cache = KvCache::new(text_config.num_hidden_layers);
        let short_logits = self.text_decoder.forward(
            &short_hidden,
            &short_cos,
            &short_sin,
            &mut short_cache,
            Some(&short_mask),
        );
        short_logits.eval();

        // Compare short fresh prefill K vs long fresh prefill K at positions 9..truncate_len.
        let (short_k, _) = short_cache.get(0).unwrap();
        let short_len = short_k.size()[2];
        let k_seg_diff = |start: i64, end: i64| -> f64 {
            let len = end - start;
            let s = short_k.narrow(2, start, len);
            let f = fresh_k.narrow(2, start, len);
            (&s - &f).abs().max().f64_value(&[])
        };
        let short_vs_long_pre = k_seg_diff(0, PRE_AUDIO_TOKEN_COUNT as i64);
        let short_vs_long_audio = k_seg_diff(
            PRE_AUDIO_TOKEN_COUNT as i64,
            truncate_len.min(short_len).min(fresh_len),
        );

        // Also compare truncated state K vs short fresh K (should match if
        // state K was correctly computed from the short prefill).
        let state_vs_short_audio = {
            let len = truncate_len.min(short_len).min(inc_len);
            let s = inc_k.narrow(
                2,
                PRE_AUDIO_TOKEN_COUNT as i64,
                len - PRE_AUDIO_TOKEN_COUNT as i64,
            );
            let f = short_k.narrow(
                2,
                PRE_AUDIO_TOKEN_COUNT as i64,
                len - PRE_AUDIO_TOKEN_COUNT as i64,
            );
            (&s - &f).abs().max().f64_value(&[])
        };

        tracing::warn!(
            "[verify-trunc] inc_k_len={} fresh_k_len={} trunc_len={} pre_diff={:.3e} old_audio_diff={:.3e} all_diff={:.3e}",
            inc_len,
            fresh_len,
            truncate_len,
            pre_d,
            old_d,
            all_d
        );
        tracing::warn!(
            "[verify-trunc] short_fresh_k_len={} short_vs_long pre={:.3e} audio={:.3e} | state_vs_short_audio={:.3e}",
            short_len,
            short_vs_long_pre,
            short_vs_long_audio,
            state_vs_short_audio
        );

        drop(fresh_cache);
        drop(fresh_logits);
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::stream::synchronize();
            crate::backend::mlx::stream::clear_cache();
        }
    }

    fn verify_post_decode(
        &self,
        audio_embeds: &Tensor,
        num_audio_tokens: usize,
        state: &StreamingState,
        text_config: &crate::config::TextDecoderConfig,
    ) {
        let seq_len = PRE_AUDIO_TOKEN_COUNT + num_audio_tokens + POST_AUDIO_TOKENS.len();
        let mut hidden_states = state.pre_audio_embeds.shallow_clone();
        for i in 0..num_audio_tokens {
            let embed = audio_embeds.get(i as i64);
            hidden_states = hidden_states.slice_scatter(
                &embed.unsqueeze(0).unsqueeze(0),
                1,
                (PRE_AUDIO_TOKEN_COUNT + i) as i64,
                (PRE_AUDIO_TOKEN_COUNT + i + 1) as i64,
                1,
            );
            if i % 32 == 31 {
                hidden_states.eval();
            }
        }
        let post_tensor = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let post_embeds = self.text_decoder.embed(&post_tensor).unsqueeze(0);
        hidden_states = Tensor::cat(&[hidden_states, post_embeds], 1);
        hidden_states.eval();
        let cos = state.cos.narrow(0, 0, seq_len as i64);
        let sin = state.sin.narrow(0, 0, seq_len as i64);
        let mask = create_causal_mask(seq_len as i64, 0, state.device);
        let mut fresh_cache = KvCache::new(text_config.num_hidden_layers);
        let fresh_logits =
            self.text_decoder
                .forward(&hidden_states, &cos, &sin, &mut fresh_cache, Some(&mask));
        fresh_logits.eval();

        let inc_cache = state.kv_cache.as_ref().unwrap();
        let (inc_k, _) = inc_cache.get(0).unwrap();
        let (fresh_k, _) = fresh_cache.get(0).unwrap();
        let inc_len = inc_k.size()[2];
        let fresh_len = fresh_k.size()[2];
        let cmp_len = (seq_len as i64).min(inc_len).min(fresh_len);
        let seg_diff = |start: i64, end: i64| -> f64 {
            if end <= start || end > cmp_len {
                return -1.0;
            }
            let len = end - start;
            let inc_seg = inc_k.narrow(2, start, len);
            let fresh_seg = fresh_k.narrow(2, start, len);
            (&inc_seg - &fresh_seg).abs().max().f64_value(&[])
        };
        let pre_d = seg_diff(0, PRE_AUDIO_TOKEN_COUNT as i64);
        let audio_d = seg_diff(
            PRE_AUDIO_TOKEN_COUNT as i64,
            (PRE_AUDIO_TOKEN_COUNT + num_audio_tokens) as i64,
        );
        let post_d = seg_diff(
            (PRE_AUDIO_TOKEN_COUNT + num_audio_tokens) as i64,
            seq_len as i64,
        );
        let all_prefill_d = seg_diff(0, cmp_len);
        tracing::warn!(
            "[verify-post-decode] inc_k_len={} fresh_k_len={} pre_diff={:.3e} audio_diff={:.3e} post_diff={:.3e} all_prefill_diff={:.3e}",
            inc_len,
            fresh_len,
            pre_d,
            audio_d,
            post_d,
            all_prefill_d
        );
        drop(fresh_cache);
        drop(fresh_logits);
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::stream::synchronize();
            crate::backend::mlx::stream::clear_cache();
        }
    }

    /// Prefill the post-audio tokens (audio_end, im_end, assistant, \n).
    /// Returns logits from the last position.
    fn prefill_post_audio(
        &self,
        kv_cache: &mut KvCache,
        state: &mut StreamingState,
        num_audio_tokens: usize,
    ) -> Result<Tensor> {
        // Build hidden: post_audio + language_prefix
        let post_tensor = Tensor::from_slice_i64(POST_AUDIO_TOKENS).to_device(state.device);
        let mut hidden_parts = vec![self.text_decoder.embed(&post_tensor).unsqueeze(0)];

        if !state.language_prefix_ids.is_empty() {
            let lang_tensor =
                Tensor::from_slice_i64(&state.language_prefix_ids).to_device(state.device);
            let lang_embeds = self.text_decoder.embed(&lang_tensor).unsqueeze(0);
            hidden_parts.push(lang_embeds);
        }

        let post_hidden = Tensor::cat(&hidden_parts, 1);
        let total_len = POST_AUDIO_TOKENS.len() + state.language_prefix_ids.len();

        let post_pos_start = (PRE_AUDIO_TOKEN_COUNT + num_audio_tokens) as i64;
        let cos = state.cos.narrow(0, post_pos_start, total_len as i64);
        let sin = state.sin.narrow(0, post_pos_start, total_len as i64);
        let mask = create_causal_mask(total_len as i64, state.cache_seq_len, state.device);

        let logits = self
            .text_decoder
            .forward(&post_hidden, &cos, &sin, kv_cache, Some(&mask));
        logits.eval();
        state.cache_seq_len += total_len as i64;
        let tpc = self.audio_encoder.tokens_per_chunk();
        let fchunks = num_audio_tokens / tpc;
        state.audio_tokens_in_cache = fchunks * tpc;

        Ok(logits)
    }
}

/// Detect and trim a long repeated subsequence at the end of `ids`.
///
/// If the last `len` tokens (len >= MIN_REPEAT) appear earlier in the
/// sequence (non-overlapping), truncate the repeated tail. This catches
/// the common case where the decoder copies a large chunk of the prefix.
fn trim_repeated_tail(ids: &mut Vec<i64>) {
    const MIN_REPEAT: usize = 6;
    let n = ids.len();
    if n < MIN_REPEAT * 2 {
        return;
    }
    // Try from longest possible repeat down to MIN_REPEAT.
    for len in (MIN_REPEAT..=n / 2).rev() {
        let tail = &ids[n - len..n];
        // Search for tail in ids[0..n-len], non-overlapping.
        for start in 0..=n - len - len {
            if ids[start..start + len] == *tail {
                ids.truncate(n - len);
                tracing::warn!(
                    "[streaming] trimmed {} repeated tokens (matched at pos {})",
                    len,
                    start
                );
                return;
            }
        }
    }
}

fn parse_asr_output(raw: &str, language_forced: bool) -> (String, String) {
    if language_forced {
        return ("forced".to_string(), raw.trim().to_string());
    }

    let raw = raw.trim();

    if let Some(rest) = raw.strip_prefix("language ") {
        if let Some(asr_pos) = rest.find("<asr_text>") {
            let lang = rest[..asr_pos].trim().to_string();
            let text = rest[asr_pos + "<asr_text>".len()..].trim().to_string();
            return (lang, text);
        }
        let mut lang_end = 0;
        for (i, c) in rest.char_indices() {
            if c.is_whitespace() || !c.is_alphabetic() {
                lang_end = i;
                break;
            }
            lang_end = i + c.len_utf8();
        }
        if lang_end > 0 {
            let lang = rest[..lang_end].to_string();
            let text = rest[lang_end..].trim().to_string();
            return (lang, text);
        }
    }

    ("unknown".to_string(), raw.to_string())
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
