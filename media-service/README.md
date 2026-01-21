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
- `LLM_PROVIDER`: LLM provider - `mock` or real provider (default: `mock`)
- `TTS_PROVIDER`: TTS provider - `mock` or real provider (default: `mock`)
- `SYSTEM_PROMPT`: System prompt for LLM (default: "You are a helpful assistant.")

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
