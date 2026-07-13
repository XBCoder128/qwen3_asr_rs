use crate::tensor::{DType, Device, Tensor};
use anyhow::Result;
use std::collections::HashMap;

use crate::config::TextDecoderConfig;
use crate::layers::{RmsNorm, TextDecoderLayer};
use crate::weights::get_weight;

/// Capacity growth granularity for the preallocated KV buffers.
const KV_CACHE_BLOCK: i64 = 256;

/// Per-layer preallocated K/V buffers (mlx-lm KVCache pattern).
///
/// `k`/`v` have shape (B, n_kv_heads, capacity, head_dim) with
/// capacity >= offset; only the first `offset` positions are valid.
struct LayerKv {
    k: Tensor,
    v: Tensor,
    offset: i64,
}

/// KV cache for autoregressive generation.
///
/// Instead of `cat([past, new])` on every step (which copies the whole
/// cache per layer per token), buffers are preallocated in blocks of
/// KV_CACHE_BLOCK positions and new entries are written in place via
/// `slice_scatter` (MLX donates the buffer when it holds the only
/// reference, so the write is a true in-place update). Truncation is O(1):
/// it only rewinds `offset`, leaving stale data beyond it to be
/// overwritten by the next prefill.
pub struct KvCache {
    layers: Vec<Option<LayerKv>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(None);
        }
        Self { layers }
    }

    /// Valid-length views of a layer's K/V (for diagnostics/verification).
    pub fn get(&self, layer: usize) -> Option<(Tensor, Tensor)> {
        self.layers[layer].as_ref().map(|e| {
            (
                e.k.narrow(2, 0, e.offset),
                e.v.narrow(2, 0, e.offset),
            )
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn seq_len(&self) -> i64 {
        self.layers[0].as_ref().map(|e| e.offset).unwrap_or(0)
    }

    /// Append `k_new`/`v_new` (B, n_kv_heads, S, head_dim) at the current
    /// offset and return valid-length views covering all positions.
    pub fn update_and_fetch(
        &mut self,
        layer: usize,
        k_new: Tensor,
        v_new: Tensor,
    ) -> (Tensor, Tensor) {
        let s = k_new.size()[2];
        let round_up = |n: i64| -> i64 { (n + KV_CACHE_BLOCK - 1) / KV_CACHE_BLOCK * KV_CACHE_BLOCK };

        let entry = match self.layers[layer].take() {
            None => {
                let shape = k_new.size(); // (B, H, S, D)
                let cap = round_up(s);
                let buf_shape = [shape[0], shape[1], cap, shape[3]];
                let kb = Tensor::zeros(&buf_shape, k_new.kind(), k_new.device());
                let vb = Tensor::zeros(&buf_shape, v_new.kind(), v_new.device());
                let kb = kb.slice_scatter(&k_new, 2, 0, s, 1);
                let vb = vb.slice_scatter(&v_new, 2, 0, s, 1);
                LayerKv { k: kb, v: vb, offset: s }
            }
            Some(mut e) => {
                let cap = e.k.size()[2];
                if e.offset + s > cap {
                    // Grow by whole blocks: one copy per KV_CACHE_BLOCK
                    // appended tokens (amortized O(1) per token).
                    let new_cap = round_up(e.offset + s);
                    let shape = e.k.size();
                    let pad_shape = [shape[0], shape[1], new_cap - cap, shape[3]];
                    let kpad = Tensor::zeros(&pad_shape, e.k.kind(), e.k.device());
                    let vpad = Tensor::zeros(&pad_shape, e.v.kind(), e.v.device());
                    e.k = Tensor::cat(&[e.k, kpad], 2);
                    e.v = Tensor::cat(&[e.v, vpad], 2);
                }
                e.k = e.k.slice_scatter(&k_new, 2, e.offset, e.offset + s, 1);
                e.v = e.v.slice_scatter(&v_new, 2, e.offset, e.offset + s, 1);
                e.offset += s;
                e
            }
        };

        let out = (
            entry.k.narrow(2, 0, entry.offset),
            entry.v.narrow(2, 0, entry.offset),
        );
        self.layers[layer] = Some(entry);
        out
    }

    /// Truncate the KV cache to `new_seq_len` positions. O(1): only the
    /// offset is rewound; stale data past it is overwritten by later writes.
    pub fn truncate(&mut self, new_seq_len: i64) {
        for layer in &mut self.layers {
            if let Some(e) = layer {
                e.offset = e.offset.min(new_seq_len);
            }
        }
    }

    /// Materialize all cache buffers (breaks the lazy graph chain).
    pub fn eval(&self) {
        for layer in self.layers.iter().flatten() {
            layer.k.eval();
            layer.v.eval();
        }
    }
}

/// Qwen3 Text Decoder model.
pub struct TextDecoder {
    embed_tokens: Tensor,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head_weight_t: Tensor, // Pre-transposed for matmul
    config: TextDecoderConfig,
}

impl TextDecoder {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &TextDecoderConfig,
    ) -> Result<Self> {
        let embed_tokens = get_weight(weights, &format!("{}.embed_tokens", prefix), "weight")?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = TextDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, &format!("{}.norm", prefix), config.rms_norm_eps)?;

        let lm_head_key = format!("{}", prefix.replace(".model", ".lm_head"));
        let lm_head_weight = if config.tie_word_embeddings {
            embed_tokens.shallow_clone()
        } else {
            get_weight(weights, &lm_head_key, "weight")?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_weight_t: lm_head_weight.tr(), // Pre-transpose at load time
            config: config.clone(),
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Tensor {
        Tensor::embedding(&self.embed_tokens, input_ids)
    }

    /// The dtype the model weights are stored in (bf16 for MLX, f32 for tch).
    /// All activations, masks and RoPE tables must use this dtype to avoid
    /// implicit weight up-casting on every forward pass.
    pub fn dtype(&self) -> DType {
        self.embed_tokens.kind()
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let mut hidden = hidden_states.shallow_clone();

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, cos, sin, kv_cache, i, mask);
        }

        let hidden = self.norm.forward(&hidden);
        hidden.matmul(&self.lm_head_weight_t)
    }

    pub fn config(&self) -> &TextDecoderConfig {
        &self.config
    }
}

/// Create a causal attention mask.
pub fn create_causal_mask(seq_len: i64, past_len: i64, dtype: DType, device: Device) -> Tensor {
    let total_len = past_len + seq_len;
    let mask = Tensor::full(&[seq_len, total_len], f64::NEG_INFINITY, dtype, device);
    let mask = mask.triu(past_len + 1);
    mask.unsqueeze(0).unsqueeze(0)
}
