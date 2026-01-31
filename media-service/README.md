# Media Service

Python service that handles RTP audio streams, implements VAD, barge-in, and integrates with AI providers (ASR, LLM, TTS).

## Features

- RTP packet handling (PCMU/ulaw, 20ms frames)
- ulaw ↔ PCM16 codec conversion
- Simple jitter buffer for packet reordering
- Energy-based VAD (Voice Activity Detection)
- Barge-in: stop TTS when speech is detected
- Pluggable AI providers (ASR, LLM, TTS)
- Multiple operation modes: echo, tts_only, full

## Configuration

Environment variables:

- `MEDIA_HOST`: Host to bind to (default: `127.0.0.1`)
- `MEDIA_PORT_BASE`: Base port for sessions (default: `40000`)
- `MEDIA_MODE`: Operation mode - `echo`, `tts_only`, or `full` (default: `full`)
- `ASR_PROVIDER`: ASR provider - `mock` or real provider (default: `mock`)
- `LLM_PROVIDER`: LLM provider - `mock` or `openai_compat` (default: `mock`)
- `TTS_PROVIDER`: TTS provider - `mock` or `http` (default: `mock`)
- `SYSTEM_PROMPT`: System prompt for LLM (short string; default: "You are a helpful assistant.")
- `SYSTEM_PROMPT_FILE`: Path to a prompt file (recommended for long prompts). If set, it overrides `SYSTEM_PROMPT`.
- `RESPONSE_LANGUAGE`: If set, forces the assistant to respond in this language (e.g. `French`, `français`, `fr`).
- `GREETING_TEXT`: Initial greeting spoken in `tts_only` / `full` mode (default: "Hello! I'm ready to help. How can I assist you today?")

### `.env` auto-loading

This service will automatically load environment variables from:

- repo root: `.env`
- or `media-service/.env`

Process environment variables still take precedence (they are **not** overwritten).

Tip: for long system prompts, put the text in a file under `media-service/prompts/` and set `SYSTEM_PROMPT_FILE`.

### OpenAI-compatible LLM provider

Set `LLM_PROVIDER=openai_compat` to use an OpenAI-compatible Chat Completions endpoint.

Additional environment variables:

- `LLM_BASE_URL`: Base URL for the OpenAI-compatible server (default: `https://api.openai.com`)
  - Examples: `https://api.openai.com`, `https://api.openai.com/v1`, `http://localhost:11434/v1`
- `LLM_API_KEY`: API key (optional for some local servers; required for OpenAI)
- `LLM_MODEL`: Model name (default: `gpt-4o-mini`)
- `LLM_TIMEOUT_MS`: Request timeout in milliseconds (default: `30000`)

### TTS_PROVIDER=http

Env vars:

- `TTS_BASE_URL`: Base URL for a TTS HTTP server (required). Defaults to using an OpenAI-style endpoint at `/v1/audio/speech`.
- `TTS_API_KEY`: Bearer token (optional).
- `TTS_VOICE`: Voice name (default: `alloy`).
- `TTS_TIMEOUT_MS`: Request timeout (default: `30000`).

Optional:

- `TTS_MODEL`: Model name sent in the request (default: `tts-1`).
- `TTS_RESPONSE_FORMAT`: `pcm` (raw PCM16, some providers) or `wav` (e.g. Orpheus-FastAPI). Default: `pcm`. WAV is auto-detected too.
- `TTS_SOURCE_SAMPLE_RATE`: Sample rate of returned PCM when `response_format="pcm"` (default: `24000`). The service resamples to 8kHz internally.

## Installation

```bash
cd media-service
pip install -r requirements.txt
```

## Running

```bash
# From media-service directory
python -m src.media_service
```

Example (OpenAI-compatible):

```bash
set LLM_PROVIDER=openai_compat
set LLM_BASE_URL=https://api.openai.com
set LLM_API_KEY=YOUR_KEY
set LLM_MODEL=gpt-4o-mini
set LLM_TIMEOUT_MS=30000
python -m src.media_service
```

Or put the same values in `media-service/.env` and just run:

```bash
python -m src.media_service
```

Or set PYTHONPATH:

```bash
export PYTHONPATH=$PWD
python src/media_service.py
```

## Operation Modes

### Echo Mode
- Simply echoes audio back (for validation)
- No AI processing

### TTS-Only Mode
- Sends initial greeting via TTS
- No ASR/LLM processing

### Full Mode
- Full conversational loop: VAD → ASR → LLM → TTS
- Barge-in support

## Architecture

- `rtp_handler.py`: RTP packet parsing and creation
- `audio_codec.py`: ulaw ↔ PCM16 conversion
- `vad.py`: Energy-based VAD
- `ai_providers.py`: Abstract interfaces and implementations (mock + real providers)
- `media_service.py`: Main service and session management

## Testing

Start in echo mode to validate audio path:

```bash
MEDIA_MODE=echo python -m src.media_service
```

Then test with a call - you should hear your voice echoed back.
