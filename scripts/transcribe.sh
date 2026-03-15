#!/usr/bin/env bash
# transcribe.sh — Auto-transcribe wavs using whisper.cpp
# Appends new entries to metadata.csv (LJSpeech format)
# Skips files already in metadata.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WAV_DIR="$PROJECT_DIR/data/wavs"
METADATA="$PROJECT_DIR/metadata.csv"
WHISPER_CLI="/opt/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL="/opt/whisper.cpp/models/ggml-base.en.bin"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -x "$WHISPER_CLI" ]; then
    echo "ERROR: whisper-cli not found at $WHISPER_CLI"
    echo "       Update WHISPER_CLI in this script if your path differs."
    exit 1
fi

if [ ! -f "$WHISPER_MODEL" ]; then
    echo "ERROR: Whisper model not found at $WHISPER_MODEL"
    exit 1
fi

# Create metadata.csv header if it doesn't exist
if [ ! -f "$METADATA" ]; then
    echo "filename|transcription|normalized_transcription" > "$METADATA"
fi

# Build a list of already-transcribed files
already_done=$(awk -F'|' 'NR>1 {print $1}' "$METADATA" | sed 's|wavs/||')

count=0
skipped=0

echo "Transcribing WAV files → $METADATA"
echo ""

for f in "$WAV_DIR"/*.wav; do
    [ -f "$f" ] || { echo "No WAV files found in $WAV_DIR"; exit 1; }
    filename="$(basename "$f" .wav)"

    # Skip if already transcribed
    if echo "$already_done" | grep -qx "$filename"; then
        echo "  $filename — already transcribed, skipping"
        ((skipped++)) || true
        continue
    fi

    echo -n "  $filename ... "

    # Run whisper — output just the text
    transcript=$("$WHISPER_CLI" \
        -m "$WHISPER_MODEL" \
        -f "$f" \
        --output-txt \
        --no-timestamps \
        -l en \
        2>/dev/null | tr -d '\n' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')

    if [ -n "$transcript" ]; then
        echo "wavs/$filename|$transcript|$transcript" >> "$METADATA"
        echo "✅  $transcript"
        ((count++)) || true
    else
        echo "⚠️  empty transcript — check file"
    fi
done

echo ""
echo "Done: $count transcribed, $skipped already done"
echo "Metadata: $METADATA"
echo ""
echo "Next: Review $METADATA and correct any transcription errors"
echo "      Then run training with: piper-train (see README)"
