#!/usr/bin/env python3
"""
Interactive voice recording script for TTS training data collection.

Usage:
    python3 record.py [--sentences ../data/sentences.txt] [--out ../data/raw]

Controls:
    Enter       → start recording / stop recording
    s           → skip sentence
    q           → quit
    r           → re-record last sentence
"""

import os
import sys
import subprocess
import argparse
import signal
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_SENTENCES = Path(__file__).parent.parent / "data" / "reading-script.txt"
DEFAULT_OUT       = Path(__file__).parent.parent / "data" / "raw"
MIC_DEVICE        = "default"   # or "hw:2,0" for webcam card 2 explicitly
SAMPLE_RATE       = 22050       # 22050Hz — Piper native format
CHANNELS          = 1           # mono
# ──────────────────────────────────────────────────────────────────────────────

def find_webcam_device():
    """Try to auto-detect the webcam ALSA device by name."""
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "cam" in line.lower() or "webcam" in line.lower() or "usb" in line.lower():
                # Extract card number
                parts = line.split("card ")
                if len(parts) > 1:
                    card_num = parts[1].split(":")[0].strip()
                    print(f"  Auto-detected mic: card {card_num} ({line.strip()})")
                    return f"hw:{card_num},0"
    except Exception:
        pass
    print("  Could not auto-detect webcam mic — using 'default'")
    return "default"


def record_clip(output_path: Path, device: str) -> subprocess.Popen:
    """Start recording. Returns the process handle."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "alsa",
        "-i", device,
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-sample_fmt", "s16",
        str(output_path)
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc


def stop_recording(proc: subprocess.Popen):
    """Gracefully stop ffmpeg recording."""
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def load_sentences(path: Path) -> list[str]:
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return lines


def main():
    parser = argparse.ArgumentParser(description="TTS recording session")
    parser.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default=None, help="ALSA device (e.g. hw:2,0)")
    parser.add_argument("--start", type=int, default=0, help="Start from sentence index")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences_path = Path(args.sentences)
    if not sentences_path.exists():
        print(f"ERROR: Sentences file not found: {sentences_path}")
        sys.exit(1)

    sentences = load_sentences(sentences_path)
    total = len(sentences)
    print(f"\n  Loaded {total} sentences from {sentences_path.name}")

    device = args.device or find_webcam_device()
    print(f"  Recording device : {device}")
    print(f"  Output directory : {out_dir}")
    print(f"  Sample rate      : {SAMPLE_RATE}Hz mono\n")
    print("─" * 60)
    print("  Controls: [Enter] start/stop  [s] skip  [r] redo  [q] quit")
    print("─" * 60)

    i = args.start
    recorded = 0
    skipped = 0

    while i < total:
        sentence = sentences[i]
        clip_name = f"line_{i+1:04d}"
        wav_path = out_dir / f"{clip_name}.wav"

        print(f"\n[{i+1}/{total}] {sentence}")

        if wav_path.exists():
            print(f"  (exists: {clip_name}.wav — press r to re-record, Enter to skip)")

        print("  >> Press Enter to start recording...", end="", flush=True)
        key = input()

        if key.lower() == "q":
            print("\n  Quitting. Session complete.")
            break
        elif key.lower() == "s":
            skipped += 1
            i += 1
            continue
        elif key.lower() == "r" and i > 0:
            i -= 1
            continue

        # Start recording — brief countdown so mic stabilises before speech
        for i in (3, 2, 1):
            print(f"  {i}...", end=" ", flush=True)
            time.sleep(0.6)
        print("  🔴 Recording... (press Enter to stop)", end="", flush=True)
        proc = record_clip(wav_path, device)
        input()
        stop_recording(proc)

        # Check file was created
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            duration = wav_path.stat().st_size / (SAMPLE_RATE * 2)  # rough estimate
            print(f"  ✅ Saved {clip_name}.wav (~{duration:.1f}s)")
            recorded += 1
            i += 1
        else:
            print("  ⚠️  Recording seems empty. Try again (press Enter) or [s] to skip.")

    print(f"\n{'─'*60}")
    print(f"  Session done — {recorded} recorded, {skipped} skipped")
    print(f"  Output: {out_dir}")
    print(f"  Next: run scripts/preprocess.sh to convert to training format")
    print(f"        run scripts/transcribe.sh to auto-transcribe clips\n")


if __name__ == "__main__":
    main()
