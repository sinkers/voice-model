#!/usr/bin/env bash
# preprocess.sh — Convert raw recordings to Piper training format
# Output: 22050Hz, mono, 16-bit PCM WAV → data/wavs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RAW_DIR="$PROJECT_DIR/data/raw"
WAV_DIR="$PROJECT_DIR/data/wavs"

mkdir -p "$WAV_DIR"

count=0
errors=0

echo "Converting raw recordings → $WAV_DIR"
echo "Format: 22050Hz mono 16-bit WAV"
echo ""

for f in "$RAW_DIR"/*.wav; do
    [ -f "$f" ] || { echo "No WAV files found in $RAW_DIR"; exit 1; }
    filename="$(basename "$f")"

    echo -n "  $filename ... "

    if ffmpeg -y -i "$f" \
        -ar 22050 \
        -ac 1 \
        -sample_fmt s16 \
        "$WAV_DIR/$filename" \
        -loglevel error 2>/dev/null; then
        echo "✅"
        ((count++)) || true
    else
        echo "❌ FAILED"
        ((errors++)) || true
    fi
done

echo ""
echo "Done: $count converted, $errors errors"
echo "Output: $WAV_DIR"
