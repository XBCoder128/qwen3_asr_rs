#!/bin/bash
# Bootstrap script for Qwen3 ASR skill
# Downloads platform-specific release (binary + runtime deps) and model

set -e

REPO="second-state/qwen3_asr_rs"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SKILL_DIR}/scripts"
MODELS_DIR="${SCRIPTS_DIR}/models"

detect_platform() {
    local os arch

    case "$(uname -s)" in
    Linux*) os="linux" ;;
    Darwin*) os="darwin" ;;
    *)
        echo "Error: Unsupported operating system: $(uname -s)" >&2
        exit 1
        ;;
    esac

    case "$(uname -m)" in
    x86_64 | amd64) arch="x86_64" ;;
    aarch64 | arm64) arch="aarch64" ;;
    *)
        echo "Error: Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
    esac

    echo "${os}-${arch}"
}

has_nvidia_gpu() {
    command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null
}

get_asset_name() {
    local platform="$1"

    case "$platform" in
    linux-x86_64)
        echo "asr-linux-x86_64"
        ;;
    linux-aarch64)
        if has_nvidia_gpu; then
            echo "asr-linux-aarch64-cuda"
        else
            echo "asr-linux-aarch64"
        fi
        ;;
    darwin-aarch64)
        echo "asr-macos-aarch64"
        ;;
    *)
        echo "Error: Unsupported platform: ${platform}" >&2
        exit 1
        ;;
    esac
}

download_release() {
    local asset_name="$1"
    local zip_name="${asset_name}.zip"

    echo "=== Downloading release (${asset_name}) ===" >&2

    mkdir -p "${SCRIPTS_DIR}"

    # Get download URL from latest release
    local api_url="https://api.github.com/repos/${REPO}/releases/latest"
    local download_url
    download_url=$(curl -sL "$api_url" | grep -o "https://github.com/${REPO}/releases/download/[^\"]*/${zip_name}" | head -1)

    if [ -z "$download_url" ]; then
        echo "Error: Could not find release asset ${zip_name}" >&2
        echo "Check https://github.com/${REPO}/releases for available downloads." >&2
        exit 1
    fi

    local temp_dir
    temp_dir=$(mktemp -d)

    echo "Fetching from: ${download_url}" >&2
    curl -sL -o "${temp_dir}/${zip_name}" "$download_url"

    echo "Extracting release..." >&2
    unzip -q "${temp_dir}/${zip_name}" -d "${temp_dir}"

    # Copy binary
    cp "${temp_dir}/${asset_name}/asr" "${SCRIPTS_DIR}/asr"
    chmod +x "${SCRIPTS_DIR}/asr"

    # Copy mlx.metallib if present (macOS MLX backend)
    if [ -f "${temp_dir}/${asset_name}/mlx.metallib" ]; then
        cp "${temp_dir}/${asset_name}/mlx.metallib" "${SCRIPTS_DIR}/mlx.metallib"
    fi

    # Copy libtorch if present (Linux tch backend)
    if [ -d "${temp_dir}/${asset_name}/libtorch" ]; then
        rm -rf "${SCRIPTS_DIR}/libtorch"
        cp -r "${temp_dir}/${asset_name}/libtorch" "${SCRIPTS_DIR}/libtorch"
    fi

    # Copy pre-built tokenizers from release assets
    if [ -d "${temp_dir}/${asset_name}/tokenizers" ]; then
        rm -rf "${SCRIPTS_DIR}/tokenizers"
        cp -r "${temp_dir}/${asset_name}/tokenizers" "${SCRIPTS_DIR}/tokenizers"
    fi

    rm -rf "$temp_dir"
    echo "Release installed to ${SCRIPTS_DIR}" >&2
}

download_models() {
    echo "=== Downloading models ===" >&2

    local model="Qwen3-ASR-0.6B"
    local model_dir="${MODELS_DIR}/${model}"
    local base_url="https://huggingface.co/Qwen/${model}/resolve/main"

    if [ -d "$model_dir" ] && [ -f "${model_dir}/config.json" ]; then
        echo "${model} already downloaded, skipping." >&2
    else
        mkdir -p "${model_dir}"

        local files="config.json model.safetensors"
        echo "Downloading ${model} from HuggingFace..." >&2
        for f in $files; do
            if [ -f "${model_dir}/${f}" ]; then
                echo "${f} already exists — skipping." >&2
            else
                echo "  Downloading ${f}..." >&2
                curl -fSL -o "${model_dir}/${f}" "${base_url}/${f}"
            fi
        done
    fi

    # Copy pre-built tokenizer from release assets
    if [ -f "${model_dir}/tokenizer.json" ]; then
        echo "tokenizer.json already exists — skipping." >&2
    else
        local src="${SCRIPTS_DIR}/tokenizers/tokenizer-0.6B.json"
        if [ ! -f "$src" ]; then
            echo "Error: Pre-built tokenizer not found at ${src}" >&2
            exit 1
        fi
        echo "Copying pre-built tokenizer..." >&2
        cp "$src" "${model_dir}/tokenizer.json"
    fi

    echo "Models installed to ${MODELS_DIR}" >&2
}

main() {
    local platform
    platform=$(detect_platform)
    echo "Detected platform: ${platform}" >&2

    local asset_name
    asset_name=$(get_asset_name "$platform")
    echo "Asset: ${asset_name}" >&2

    download_release "$asset_name"
    download_models

    echo "" >&2
    echo "=== Installation complete ===" >&2
    echo "Installed files:" >&2
    ls -1 "${SCRIPTS_DIR}" | grep -v '^\.' >&2
}

main "$@"
