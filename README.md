# Voice Model Training Project

A base project for training a custom TTS voice model using your own voice recordings.

## Goal
Train a Piper-compatible TTS voice model using Andrew's voice, suitable for use with the OpenClaw voice assistant.

---

## Project Structure

```
voice-model/
├── data/
│   ├── raw/        # Original recordings (untouched)
│   ├── wavs/       # Processed 22050Hz mono WAV files (training-ready)
│   └── clips/      # Short segmented clips (if splitting long recordings)
├── scripts/        # Preprocessing, segmentation, training scripts
├── output/         # Trained model files
├── notes/          # Transcripts, session notes, todo lists
├── metadata.csv    # LJSpeech-format transcript index
└── README.md       # This file
```

---

## Setup & Known Fixes

Before anything works, run the setup script:

```bash
bash scripts/setup.sh
```

This applies all required patches documented below. Safe to re-run.

### Fix 1 — Telegram IPv6 / media download failure

**Symptom:** Telegram shows "failed to download media" when sending audio files. OpenClaw logs show:
```
MediaFetchError: Failed to fetch media from https://api.telegram.org/file/...: TypeError: fetch failed
```

**Root cause:** The Pi's DNS resolves `api.telegram.org` to an IPv6 address (`2001:67c:4e8:f004::9`), but IPv6 is not configured on this network. Node.js's built-in `fetch` (undici) uses the system resolver and times out. `curl` works because it has its own IPv4 fallback.

**Fix:** Pin `api.telegram.org` to its IPv4 address in `/etc/hosts`:
```
149.154.166.110 api.telegram.org
```

Also set in OpenClaw config (`~/.openclaw/openclaw.json`) under `channels.telegram`:
```json
"network": {
  "autoSelectFamily": false,
  "dnsResultOrder": "ipv4first"
}
```

Note: The `autoSelectFamily` config only fixes the grammY polling connection. The `/etc/hosts` entry is required to fix the media download fetch path.

**Verify:**
```bash
getent ahosts api.telegram.org   # should show IPv4 only
node -e "fetch('https://api.telegram.org').then(r=>console.log('OK',r.status)).catch(e=>console.log('FAIL',e.cause?.code))"
```

### Fix 2 — OpenClaw media inbound path

**Symptom:** Voice training capture hook fails to find audio files after download.

**Root cause:** OpenClaw saves inbound media to `~/.openclaw/media/inbound/`, not `/tmp/openclaw-media/` as initially assumed from the source code.

**Fix:** Hook `handler.ts` updated to use correct path:
```typescript
const MEDIA_CACHE = join(homedir(), ".openclaw", "media", "inbound");
```

No script needed — already fixed in the hook source.

---

## Training Format: LJSpeech

Piper (and most open TTS engines) use the **LJSpeech** format:

- Audio: 22050 Hz, mono, 16-bit WAV files in `data/wavs/`
- Metadata: `metadata.csv` in the format:
  ```
  filename|transcription|normalized_transcription
  ```
  Example:
  ```
  wavs/line_001|Hello, my name is Andrew.|Hello, my name is Andrew.
  ```

---

## Recommended Approach

### Phase 1 – Record
- Record yourself reading **500–2000 sentences** (more = better quality)
- Good sources: common phrases, news articles, book excerpts
- Target: clean room, consistent mic position, 16kHz+ sample rate
- Avoid: background noise, mouth sounds, long pauses

### Phase 2 – Process
- Convert to 22050Hz mono WAV (script provided in `scripts/`)
- Segment long recordings into short clips (2–10 seconds each)
- Transcribe each clip accurately

### Phase 3 – Train
- Use **Piper Training** (cloud recommended for speed) or local on a GPU machine
- Piper training repo: https://github.com/rhasspy/piper-train

### Phase 4 – Deploy
- Drop the trained `.onnx` model into `/opt/voice-assistant/voices/`
- Update `voice_assistant.py` to reference the new model

---

## Tools

| Tool | Purpose |
|------|---------|
| `ffmpeg` | Audio conversion and preprocessing |
| `whisper.cpp` | Auto-transcription of recordings |
| `piper-train` | Model training |
| `piper` | Inference / testing |

---

## Resources

- Piper TTS: https://github.com/rhasspy/piper
- Piper Training: https://github.com/rhasspy/piper-train
- LJSpeech format: https://keithito.com/LJ-Speech-Dataset/
- Coqui XTTS (alternative, higher quality): https://github.com/coqui-ai/TTS

---

## Status

- [ ] Project scaffolded
- [ ] Recording script prepared
- [ ] Sample sentences collected
- [ ] Recordings captured
- [ ] Audio preprocessed
- [ ] Transcriptions completed
- [ ] Training run
- [ ] Model deployed to voice assistant
