//! Qwen3 ForcedAligner — non-autoregressive word/char timestamp prediction.
//!
//! Mirrors the Python `qwen_asr.Qwen3ForcedAligner` API:
//! 1. Tokenize transcript into alignment units (CJK chars / space words).
//! 2. Build prompt: `<|audio_start|>` + audio pads + `<|audio_end|>` +
//!    `word<timestamp><timestamp>...`
//! 3. Single bidirectional-capable full-sequence forward (same causal mask
//!    as the HF thinker; NAR because we read all timestamp positions at once).
//! 4. Argmax over the 5000-class timestamp head at `<timestamp>` positions.
//! 5. LIS-based `fix_timestamp`, then pair start/end per word.

use anyhow::{Context, Result};
use std::path::Path;

use crate::audio;
use crate::audio_encoder::AudioEncoder;
use crate::config::AsrConfig;
use crate::layers::compute_mrope_cos_sin;
use crate::mel::WhisperFeatureExtractor;
use crate::tensor::{Device, Tensor};
use crate::text_decoder::{create_causal_mask, KvCache, TextDecoder};
use crate::tokenizer::{
    AsrTokenizer, AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_START_TOKEN_ID, TIMESTAMP_TOKEN_ID,
};
use crate::weights;

const MEL_SAMPLE_RATE: u32 = 16000;

/// One aligned span (word or CJK character).
#[derive(Debug, Clone)]
pub struct ForcedAlignItem {
    pub text: String,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
}

/// Forced alignment result for one sample.
#[derive(Debug, Clone)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
}

/// ForcedAligner inference engine (loads Qwen3-ForcedAligner weights).
pub struct AlignInference {
    audio_encoder: AudioEncoder,
    text_decoder: TextDecoder,
    mel_extractor: WhisperFeatureExtractor,
    tokenizer: AsrTokenizer,
    config: AsrConfig,
    device: Device,
    timestamp_token_id: i64,
    timestamp_segment_time: f64,
}

impl AlignInference {
    /// Load ForcedAligner from a model directory.
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        tracing::info!("Loading ForcedAligner from {:?}", model_dir);

        let config =
            AsrConfig::from_file(&model_dir.join("config.json")).context("Failed to load config")?;

        let is_aligner = config
            .thinker_config
            .model_type
            .as_deref()
            .map(|t| t.contains("forced_aligner") || t.contains("aligner"))
            .unwrap_or(false)
            || config.thinker_config.classify_num.is_some();
        if !is_aligner {
            tracing::warn!(
                "Model does not look like a ForcedAligner (no classify_num / aligner model_type); \
                 continuing anyway"
            );
        }

        let all_weights =
            weights::load_model_weights(model_dir, device).context("Failed to load weights")?;
        tracing::info!("Loaded {} weight tensors", all_weights.len());

        let audio_encoder = AudioEncoder::load(
            &all_weights,
            "thinker.audio_tower",
            &config.thinker_config.audio_config,
            device,
        )
        .context("Failed to load audio encoder")?;

        // ForcedAligner uses untied lm_head of shape (classify_num, hidden).
        // Ensure tie_word_embeddings is false so we load thinker.lm_head.
        let mut text_config = config.thinker_config.text_config.clone();
        if config.thinker_config.classify_num.is_some() {
            text_config.tie_word_embeddings = false;
        }

        let text_decoder = TextDecoder::load(&all_weights, "thinker.model", &text_config)
            .context("Failed to load text decoder")?;

        let tokenizer = AsrTokenizer::from_dir(model_dir).context("Failed to load tokenizer")?;

        let mel_extractor = WhisperFeatureExtractor::new(
            400,
            160,
            config.thinker_config.audio_config.num_mel_bins,
            MEL_SAMPLE_RATE,
            device,
        );

        let timestamp_token_id = if config.timestamp_token_id != 0 {
            config.timestamp_token_id
        } else {
            TIMESTAMP_TOKEN_ID
        };

        tracing::info!(
            "ForcedAligner loaded (timestamp_token_id={}, segment_time={}ms, classify_num={:?})",
            timestamp_token_id,
            config.timestamp_segment_time,
            config.thinker_config.classify_num
        );

        Ok(Self {
            audio_encoder,
            text_decoder,
            mel_extractor,
            tokenizer,
            timestamp_token_id,
            timestamp_segment_time: config.timestamp_segment_time,
            config,
            device,
        })
    }

    /// Align transcript text to an audio file.
    pub fn align(
        &self,
        audio_path: &str,
        text: &str,
        language: &str,
    ) -> Result<ForcedAlignResult> {
        let samples = audio::load_audio(audio_path, MEL_SAMPLE_RATE)?;
        self.align_samples(&samples, text, language)
    }

    /// Align transcript text to raw 16 kHz mono f32 samples.
    pub fn align_samples(
        &self,
        samples: &[f32],
        text: &str,
        language: &str,
    ) -> Result<ForcedAlignResult> {
        let (word_list, _) = encode_timestamp(text, language)?;
        if word_list.is_empty() {
            return Ok(ForcedAlignResult { items: Vec::new() });
        }

        let mel = self.mel_extractor.extract(samples, self.device)?;
        let audio_embeds = self.audio_encoder.forward(&mel);
        audio_embeds.eval();
        let num_audio_tokens = audio_embeds.size()[0] as usize;
        tracing::info!(
            "Align: {} words, {} audio tokens",
            word_list.len(),
            num_audio_tokens
        );

        let (input_ids, audio_start, audio_end) =
            self.build_align_prompt(num_audio_tokens, &word_list)?;
        let seq_len = input_ids.len();

        let pre_tensor = Tensor::from_slice_i64(&input_ids[..audio_start]).to_device(self.device);
        let post_tensor = Tensor::from_slice_i64(&input_ids[audio_end..]).to_device(self.device);
        let pre_embeds = self.text_decoder.embed(&pre_tensor).unsqueeze(0);
        let post_embeds = self.text_decoder.embed(&post_tensor).unsqueeze(0);
        let hidden_states = Tensor::cat(&[pre_embeds, audio_embeds.unsqueeze(0), post_embeds], 1);
        hidden_states.eval();

        let text_config = &self.config.thinker_config.text_config;
        let model_dtype = self.text_decoder.dtype();
        let all_positions: Vec<i64> = (0..seq_len as i64).collect();
        let all_pos_ids: [Vec<i64>; 3] =
            [all_positions.clone(), all_positions.clone(), all_positions];
        let (cos, sin) = compute_mrope_cos_sin(
            &all_pos_ids,
            text_config.head_dim,
            text_config.rope_theta,
            &text_config.mrope_section(),
            text_config.mrope_interleaved(),
            self.device,
        );
        let cos = cos.to_dtype(model_dtype);
        let sin = sin.to_dtype(model_dtype);

        // Match HF thinker: causal mask over the full sequence, single forward.
        let mask = create_causal_mask(seq_len as i64, 0, model_dtype, self.device);
        let mut kv_cache = KvCache::new(text_config.num_hidden_layers);
        let logits =
            self.text_decoder
                .forward(&hidden_states, &cos, &sin, &mut kv_cache, Some(&mask));
        logits.eval();

        // logits: (1, seq, classify_num) → argmax over last dim → (1, seq)
        let pred = logits.argmax(-1, false);
        pred.eval();
        let pred_ids = pred.squeeze_dim(0).to_vec_i64();
        if pred_ids.len() != seq_len {
            anyhow::bail!(
                "Unexpected argmax length: got {}, expected {}",
                pred_ids.len(),
                seq_len
            );
        }

        let mut timestamp_ms: Vec<i64> = Vec::with_capacity(word_list.len() * 2);
        for (i, &tok) in input_ids.iter().enumerate() {
            if tok == self.timestamp_token_id {
                let class = pred_ids[i].max(0);
                timestamp_ms.push((class as f64 * self.timestamp_segment_time).round() as i64);
            }
        }

        if timestamp_ms.len() != word_list.len() * 2 {
            anyhow::bail!(
                "Timestamp count mismatch: got {} values for {} words (expected {})",
                timestamp_ms.len(),
                word_list.len(),
                word_list.len() * 2
            );
        }

        let fixed = fix_timestamp(&timestamp_ms);
        let mut items = Vec::with_capacity(word_list.len());
        for (i, word) in word_list.iter().enumerate() {
            let start = fixed[i * 2] as f64 / 1000.0;
            let end = fixed[i * 2 + 1] as f64 / 1000.0;
            items.push(ForcedAlignItem {
                text: word.clone(),
                start_time: (start * 1000.0).round() / 1000.0,
                end_time: (end * 1000.0).round() / 1000.0,
            });
        }

        drop(kv_cache);
        drop(hidden_states);
        drop(audio_embeds);
        drop(mel);
        #[cfg(feature = "mlx")]
        {
            crate::backend::mlx::stream::synchronize();
            crate::backend::mlx::stream::clear_cache();
        }

        Ok(ForcedAlignResult { items })
    }

    fn build_align_prompt(
        &self,
        num_audio_tokens: usize,
        word_list: &[String],
    ) -> Result<(Vec<i64>, usize, usize)> {
        // <|audio_start|> + pads + <|audio_end|> + word<ts><ts>...
        let mut tokens: Vec<i64> = vec![AUDIO_START_TOKEN_ID];
        let audio_start = tokens.len();
        for _ in 0..num_audio_tokens {
            tokens.push(AUDIO_PAD_TOKEN_ID);
        }
        let audio_end = tokens.len();
        tokens.push(AUDIO_END_TOKEN_ID);

        for word in word_list {
            tokens.extend(self.tokenizer.encode(word)?);
            tokens.push(self.timestamp_token_id);
            tokens.push(self.timestamp_token_id);
        }

        Ok((tokens, audio_start, audio_end))
    }
}

// ---------------------------------------------------------------------------
// Text tokenization (port of Qwen3ForceAlignProcessor)
// ---------------------------------------------------------------------------

fn is_kept_char(ch: char) -> bool {
    if ch.is_alphabetic() || ch.is_numeric() {
        return true;
    }
    // Keep apostrophe / punctuation so aligned segments match ASR text.
    // Whitespace still dropped via split_whitespace upstream.
    matches!(
        ch,
        '\''
            | '\u{2019}' // ’
            | '\u{2018}' // ‘
            | '.'
            | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | '-'
            | '"'
            | '\u{201c}' // “
            | '\u{201d}' // ”
            | '('
            | ')'
            | '['
            | ']'
            | '…'
            | '—'
            | '–'
            | '。'
            | '，'
            | '！'
            | '？'
            | '；'
            | '：'
            | '、'
            | '（'
            | '）'
            | '【'
            | '】'
            | '《'
            | '》'
            | '「'
            | '」'
            | '『'
            | '』'
            | '·'
            | '～'
            | '~'
            | '%'
            | '/'
            | '\\'
            | '&'
            | '+'
            | '='
            | '*'
            | '#'
            | '@'
    )
}

fn clean_token(token: &str) -> String {
    token.chars().filter(|&c| is_kept_char(c)).collect()
}

/// Split CJK ideographs AND CJK/fullwidth punctuation into single units;
/// keep Latin runs (incl. attached ASCII punct / apostrophes) together.
fn is_cjk_unit_char(ch: char) -> bool {
    is_cjk_char(ch)
        || matches!(
            ch,
            '。' | '，' | '！' | '？' | '；' | '：' | '、' | '（' | '）' | '【' | '】'
                | '《' | '》' | '「' | '」' | '『' | '』' | '·' | '…' | '—' | '～'
        )
}

fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

fn split_segment_with_chinese(seg: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in seg.chars() {
        if is_cjk_unit_char(ch) {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

fn tokenize_space_lang(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for seg in text.split_whitespace() {
        let cleaned = clean_token(seg);
        if cleaned.is_empty() {
            continue;
        }
        tokens.extend(split_segment_with_chinese(&cleaned));
    }
    tokens
}

/// Encode transcript into alignment units + the aligner input text string.
/// Japanese/Korean fall back to space/CJK tokenization (no nagisa/soynlp).
pub fn encode_timestamp(text: &str, language: &str) -> Result<(Vec<String>, String)> {
    let lang = language.to_lowercase();
    let word_list = match lang.as_str() {
        "japanese" | "korean" => {
            tracing::warn!(
                "language={}: using CJK/space tokenization (no dedicated ja/ko tokenizer)",
                language
            );
            tokenize_space_lang(text)
        }
        _ => tokenize_space_lang(text),
    };

    // Match Python: "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
    let mut input_text = String::from("<|audio_start|><|audio_pad|><|audio_end|>");
    for w in &word_list {
        input_text.push_str(w);
        input_text.push_str("<timestamp><timestamp>");
    }

    Ok((word_list, input_text))
}

/// Longest non-decreasing subsequence repair (port of Python `fix_timestamp`).
pub fn fix_timestamp(data: &[i64]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dp = vec![1usize; n];
    let mut parent = vec![-1isize; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j as isize;
            }
        }
    }

    let max_length = *dp.iter().max().unwrap();
    let max_idx = dp.iter().position(|&x| x == max_length).unwrap();
    let mut lis_indices = Vec::new();
    let mut idx = max_idx as isize;
    while idx != -1 {
        lis_indices.push(idx as usize);
        idx = parent[idx as usize];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for &i in &lis_indices {
        is_normal[i] = true;
    }

    let mut result: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let mut i = 0;
    while i < n {
        if !is_normal[i] {
            let mut j = i;
            while j < n && !is_normal[j] {
                j += 1;
            }
            let anomaly_count = j - i;

            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            if anomaly_count <= 2 {
                for k in i..j {
                    result[k] = match (left_val, right_val) {
                        (None, Some(r)) => r,
                        (Some(l), None) => l,
                        (Some(l), Some(r)) => {
                            if (k as isize - (i as isize - 1)) <= (j as isize - k as isize) {
                                l
                            } else {
                                r
                            }
                        }
                        (None, None) => result[k],
                    };
                }
            } else {
                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let step = (r - l) / (anomaly_count as f64 + 1.0);
                        for k in i..j {
                            result[k] = l + step * ((k - i + 1) as f64);
                        }
                    }
                    (Some(l), None) => {
                        for k in i..j {
                            result[k] = l;
                        }
                    }
                    (None, Some(r)) => {
                        for k in i..j {
                            result[k] = r;
                        }
                    }
                    (None, None) => {}
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    result.into_iter().map(|x| x as i64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_english() {
        let (words, _) = encode_timestamp("Hello world Python test", "English").unwrap();
        assert_eq!(words, vec!["Hello", "world", "Python", "test"]);
    }

    #[test]
    fn tokenize_english_keeps_punct_and_apostrophe() {
        let (words, _) =
            encode_timestamp("It's day five, growing.", "English").unwrap();
        assert_eq!(words, vec!["It's", "day", "five,", "growing."]);
    }

    #[test]
    fn tokenize_chinese() {
        let (words, _) =
            encode_timestamp("甚至出现交易几乎停滞的情况。", "Chinese").unwrap();
        assert_eq!(
            words,
            vec![
                "甚", "至", "出", "现", "交", "易", "几", "乎", "停", "滞", "的", "情", "况",
                "。"
            ]
        );
    }

    #[test]
    fn fix_timestamp_monotonic() {
        let data = vec![0, 80, 160, 240];
        assert_eq!(fix_timestamp(&data), data);
    }

    #[test]
    fn fix_timestamp_repairs_glitch() {
        let data = vec![0, 80, 1000, 240, 320];
        let fixed = fix_timestamp(&data);
        // 1000 breaks LIS; should be repaired toward neighbors
        assert!(fixed.windows(2).all(|w| w[0] <= w[1]));
    }
}
