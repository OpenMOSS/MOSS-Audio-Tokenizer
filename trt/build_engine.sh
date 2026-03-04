#!/usr/bin/env bash
#
# Build TensorRT engines from ONNX models for MOSS Audio Tokenizer.
#
# ⚠️  WE DO NOT PROVIDE PRE-BUILT TRT ENGINES.
# TRT engines are tied to your specific GPU architecture and TensorRT version.
# You must build them yourself from the ONNX models we provide.
#
# Download ONNX models first:
#   huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX \
#       --local-dir weights/MOSS-Audio-Tokenizer-ONNX
#
# Prerequisites:
#   - TensorRT SDK installed (trtexec available in PATH)
#   - pip install tensorrt cuda-python
#
# Usage:
#   bash build_engine.sh <encoder.onnx> <decoder.onnx> <output_dir> [--fp8]
#
# Example:
#   bash build_engine.sh \
#       weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx \
#       weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx \
#       weights/MOSS-Audio-Tokenizer-TRT
#
# ═══════════════════════════════════════════════════════════════════════════
#  ⚠️  IMPORTANT: maxShapes AND MAXIMUM AUDIO LENGTH
# ═══════════════════════════════════════════════════════════════════════════
#
#  MOSS Audio Tokenizer operates at 24kHz with a downsample rate of 1920.
#  The relationship between shapes and audio duration is:
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  ENCODER: input_values shape = (1, 1, num_samples)                 │
#  │           num_samples = audio_seconds × 24000                      │
#  │           num_samples must be a multiple of 1920                   │
#  │                                                                     │
#  │  DECODER: audio_codes shape = (32, 1, num_frames)                  │
#  │           num_frames = num_samples / 1920                          │
#  │           audio_seconds = num_frames × 0.08                        │
#  └─────────────────────────────────────────────────────────────────────┘
#
#  The --maxShapes flag sets the MAXIMUM input your engine can handle.
#  Inputs exceeding maxShapes will cause a runtime error.
#
#  ┌────────────────────────────────────────────────────────────────────────┐
#  │  Encoder maxShapes          │  Max Audio Duration                     │
#  │────────────────────────────│─────────────────────────────────────────│
#  │  input_values:1x1x480000   │  20 seconds  (480000 / 24000)          │
#  │  input_values:1x1x960000   │  40 seconds  (960000 / 24000)  ← DEFAULT │
#  │  input_values:1x1x1440000  │  60 seconds  (1440000 / 24000)         │
#  │  input_values:1x1x2400000  │  100 seconds (2400000 / 24000)         │
#  │  input_values:1x1x7200000  │  300 seconds (7200000 / 24000)         │
#  │────────────────────────────│─────────────────────────────────────────│
#  │  Decoder maxShapes          │  Max Audio Duration                     │
#  │────────────────────────────│─────────────────────────────────────────│
#  │  audio_codes:32x1x250      │  20 seconds  (250 × 0.08)              │
#  │  audio_codes:32x1x500      │  40 seconds  (500 × 0.08)   ← DEFAULT  │
#  │  audio_codes:32x1x750      │  60 seconds  (750 × 0.08)              │
#  │  audio_codes:32x1x1250     │  100 seconds (1250 × 0.08)             │
#  │  audio_codes:32x1x3750     │  300 seconds (3750 × 0.08)             │
#  └────────────────────────────────────────────────────────────────────────┘
#
#  Larger maxShapes → longer build time, more GPU memory for the engine.
#  Choose the smallest maxShapes that covers your use case.
#
#  To change the max audio length, edit MAX_AUDIO_SECONDS below.
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configurable: max audio length in seconds ────────────────────────────
MAX_AUDIO_SECONDS=40

# ── Derived shapes ───────────────────────────────────────────────────────
SAMPLE_RATE=24000
DOWNSAMPLE_RATE=1920

MAX_SAMPLES=$((MAX_AUDIO_SECONDS * SAMPLE_RATE))
MAX_FRAMES=$((MAX_SAMPLES / DOWNSAMPLE_RATE))

# Round MAX_SAMPLES up to nearest multiple of DOWNSAMPLE_RATE
MAX_SAMPLES=$(( ((MAX_SAMPLES + DOWNSAMPLE_RATE - 1) / DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE ))

OPT_SECONDS=8
OPT_SAMPLES=$((OPT_SECONDS * SAMPLE_RATE))
OPT_SAMPLES=$(( ((OPT_SAMPLES + DOWNSAMPLE_RATE - 1) / DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE ))
OPT_FRAMES=$((OPT_SAMPLES / DOWNSAMPLE_RATE))

if [ $# -lt 3 ]; then
    echo "Usage: $0 <encoder.onnx> <decoder.onnx> <output_dir> [--fp8]"
    echo ""
    echo "  encoder.onnx   Path to the encoder ONNX model"
    echo "  decoder.onnx   Path to the decoder ONNX model"
    echo "  output_dir     Directory to save .engine files"
    echo "  --fp8          (optional) Use FP8 precision instead of FP16"
    echo ""
    echo "Current MAX_AUDIO_SECONDS=${MAX_AUDIO_SECONDS} → max ${MAX_SAMPLES} samples / ${MAX_FRAMES} frames"
    echo "Edit MAX_AUDIO_SECONDS in this script to change the limit."
    exit 1
fi

ENCODER_ONNX="$1"
DECODER_ONNX="$2"
OUTPUT_DIR="$3"
PRECISION="--fp16"

if [ "${4:-}" = "--fp8" ]; then
    PRECISION="--fp8"
fi

mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════════════════"
echo "  MOSS Audio Tokenizer — TensorRT Engine Builder"
echo "═══════════════════════════════════════════════════════════════"
echo "  Max audio duration:  ${MAX_AUDIO_SECONDS} seconds"
echo "  Encoder maxShapes:   input_values:1x1x${MAX_SAMPLES}"
echo "  Decoder maxShapes:   audio_codes:32x1x${MAX_FRAMES}"
echo "  Precision:           ${PRECISION}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "[1/2] Building encoder engine..."
trtexec \
    --onnx="${ENCODER_ONNX}" \
    --saveEngine="${OUTPUT_DIR}/encoder.engine" \
    ${PRECISION} \
    --minShapes=input_values:1x1x${DOWNSAMPLE_RATE},n_quantizers:1 \
    --optShapes=input_values:1x1x${OPT_SAMPLES},n_quantizers:1 \
    --maxShapes=input_values:1x1x${MAX_SAMPLES},n_quantizers:1 \
    --workspace=4096

echo ""
echo "[2/2] Building decoder engine..."
trtexec \
    --onnx="${DECODER_ONNX}" \
    --saveEngine="${OUTPUT_DIR}/decoder.engine" \
    ${PRECISION} \
    --minShapes=audio_codes:32x1x1,n_quantizers:1 \
    --optShapes=audio_codes:32x1x${OPT_FRAMES},n_quantizers:1 \
    --maxShapes=audio_codes:32x1x${MAX_FRAMES},n_quantizers:1 \
    --workspace=4096

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done! Engines saved to:"
echo "    ${OUTPUT_DIR}/encoder.engine"
echo "    ${OUTPUT_DIR}/decoder.engine"
echo ""
echo "  Max supported audio: ${MAX_AUDIO_SECONDS} seconds"
echo "  To change: edit MAX_AUDIO_SECONDS in this script and rebuild."
echo "═══════════════════════════════════════════════════════════════"
