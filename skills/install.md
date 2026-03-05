# Qwen3 ASR Skill — Installation Guide

Install the Qwen3 ASR skill for voice transcription using Qwen3-ASR-0.6B.

## Prerequisites

- `curl` (for downloading)
- `unzip` (for extraction)
- `bash` (shell)
- `pip` with `huggingface_hub` and `transformers` (for model download and tokenizer generation)

## Quick Install (Recommended)

```bash
SKILL_DIR="${HOME}/.openclaw/skills/audio_asr"
mkdir -p "$SKILL_DIR"

# Clone the repo
git clone --depth 1 https://github.com/second-state/qwen3_asr_rs.git /tmp/qwen3-asr-repo
cp -r /tmp/qwen3-asr-repo/skills/* "$SKILL_DIR"
rm -rf /tmp/qwen3-asr-repo

# Download platform-specific release and model
"${SKILL_DIR}/bootstrap.sh"
```

After installation, verify it works:

```bash
~/.openclaw/skills/audio_asr/scripts/asr \
  ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B \
  /path/to/audio.wav
```

## Manual Installation

If the automatic download fails, manually install the components:

1. Go to https://github.com/second-state/qwen3_asr_rs/releases/latest
2. Download the zip for your platform:
   - `asr-linux-x86_64.zip` (Linux x86_64 CPU — includes libtorch)
   - `asr-linux-x86_64-cuda.zip` (Linux x86_64 CUDA — includes libtorch)
   - `asr-linux-aarch64.zip` (Linux ARM64 CPU — includes libtorch)
   - `asr-linux-aarch64-cuda.zip` (Linux ARM64 CUDA / Jetson — includes libtorch)
   - `asr-macos-aarch64.zip` (macOS Apple Silicon — includes mlx.metallib)
3. Extract the zip and copy contents to the scripts directory:
   ```bash
   SCRIPTS=~/.openclaw/skills/audio_asr/scripts
   mkdir -p "$SCRIPTS"
   unzip asr-<platform>.zip
   cp -r asr-<platform>/* "$SCRIPTS/"
   chmod +x "$SCRIPTS/asr"
   ```
4. Download model:
   ```bash
   huggingface-cli download Qwen/Qwen3-ASR-0.6B \
     --local-dir ~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B
   ```
5. Generate `tokenizer.json`:
   ```bash
   python3 -c "
   from transformers import AutoTokenizer
   import os
   path = os.path.expanduser('~/.openclaw/skills/audio_asr/scripts/models/Qwen3-ASR-0.6B')
   tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
   tok.backend_tokenizer.save(f'{path}/tokenizer.json')
   print(f'Saved {path}/tokenizer.json')
   "
   ```

## Troubleshooting

### Download Failed

Check network connectivity:

```bash
curl -I "https://github.com/second-state/qwen3_asr_rs/releases/latest"
```

### Unsupported Platform

Check your platform:

```bash
echo "OS: $(uname -s), Arch: $(uname -m)"
```

Supported: Linux (x86_64, aarch64) and macOS (Apple Silicon arm64).
