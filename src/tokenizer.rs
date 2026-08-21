use anyhow::Result;
use std::path::Path;

pub struct AsrTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl AsrTokenizer {
    /// Load tokenizer from model directory.
    /// Expects either tokenizer.json or vocab.json + merges.txt
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            return Ok(Self { tokenizer });
        }

        // If tokenizer.json doesn't exist, generate it from vocab.json and merges.txt
        anyhow::bail!(
            "tokenizer.json not found in {:?}. \
             Please generate it using: \
             python -c \"from transformers import AutoTokenizer; \
             tok = AutoTokenizer.from_pretrained('{}', trust_remote_code=True); \
             tok.backend_tokenizer.save('{}/tokenizer.json')\"",
            model_dir,
            model_dir.display(),
            model_dir.display()
        );
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        let text = self
            .tokenizer
            .decode(&u32_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }

    /// Token IDs that must never be *generated* during ASR decoding.
    ///
    /// The added-token block of the Qwen vocab is entirely control markup
    /// (chat-template markers, vision/box/audio/tts tokens, blanks, tool
    /// markers). None of it is legitimate transcription output, but greedy
    /// argmax occasionally lands on one mid-decode (observed: `<|im_start|>`
    /// → the model continues a whole new chat turn, producing
    /// `assistant<asr_text>…` garbage that must be stripped downstream, and
    /// in streaming it poisons the next call's rollback prefix).
    ///
    /// Kept legal: the two EOS tokens and `<asr_text>` (part of the model's
    /// normal `language X<asr_text>…` output format).
    pub fn banned_output_ids(&self) -> Vec<i64> {
        const OUTPUT_LEGAL: [&str; 3] = ["<|endoftext|>", "<|im_end|>", "<asr_text>"];
        self.tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, tok)| !OUTPUT_LEGAL.contains(&tok.content.as_str()))
            .map(|(id, _)| *id as i64)
            .collect()
    }
}

// Special token IDs for Qwen3-ASR
pub const IM_START_TOKEN_ID: i64 = 151644;
pub const IM_END_TOKEN_ID: i64 = 151645;
pub const ENDOFTEXT_TOKEN_ID: i64 = 151643;
pub const AUDIO_START_TOKEN_ID: i64 = 151669;
pub const AUDIO_END_TOKEN_ID: i64 = 151670;
pub const AUDIO_PAD_TOKEN_ID: i64 = 151676;
pub const ASR_TEXT_TOKEN_ID: i64 = 151704;
/// ForcedAligner `<timestamp>` token.
pub const TIMESTAMP_TOKEN_ID: i64 = 151705;

#[cfg(test)]
mod tests {
    use super::*;

    /// Requires a local model dir (tokenizer.json); skips otherwise.
    #[test]
    fn banned_output_ids_excludes_eos_and_asr_text() {
        let model_dir = std::path::Path::new("Qwen3-ASR-0.6B");
        if !model_dir.join("tokenizer.json").exists() {
            eprintln!("skipping: no local model dir");
            return;
        }
        let tok = AsrTokenizer::from_dir(model_dir).unwrap();
        let banned = tok.banned_output_ids();

        assert!(!banned.is_empty(), "expected the added-token block to be banned");
        for legal in [
            ENDOFTEXT_TOKEN_ID,
            IM_END_TOKEN_ID,
            ASR_TEXT_TOKEN_ID,
        ] {
            assert!(
                !banned.contains(&legal),
                "token {legal} must stay legal in output"
            );
        }
        for must_ban in [
            IM_START_TOKEN_ID,
            AUDIO_START_TOKEN_ID,
            AUDIO_END_TOKEN_ID,
            AUDIO_PAD_TOKEN_ID,
        ] {
            assert!(banned.contains(&must_ban), "token {must_ban} must be banned");
        }
        // Banned ids all live in the added-token block (>= im_start id).
        assert!(banned.iter().all(|&id| id >= IM_START_TOKEN_ID));
    }
}
