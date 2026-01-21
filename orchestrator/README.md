# ARI Orchestrator

Node.js/TypeScript service that manages call lifecycle via Asterisk ARI (Asterisk REST Interface).

## Features

- Connects to ARI WebSocket for the `voicebot` application
- Handles `StasisStart` events: answers channel, creates bridge, creates ExternalMedia channel
- Handles `StasisEnd`/hangup events: cleanup bridge and media session (idempotent)
- Structured logging with correlation IDs
- Port allocation for media service endpoints
- Graceful shutdown

## Configuration

Environment variables:

- `ARI_URL`: ARI HTTP server URL (default: `http://localhost:8088`)
- `ARI_USER`: ARI username (default: `voicebot`)
- `ARI_PASS`: ARI password (default: `voicebot_pass`)
- `ARI_APP`: Stasis application name (default: `voicebot`)
- `MEDIA_HOST`: Media service host (default: `127.0.0.1`)
- `MEDIA_PORT_BASE`: Base port for media service (default: `40000`)
- `AUDIO_FORMAT`: Audio format (default: `ulaw`)
- `LOG_LEVEL`: Log level (default: `info`)

## Installation

```bash
cd orchestrator
npm install
npm run build
```

## Running

```bash
# Production
npm start

# Development (with ts-node)
npm run dev

# Watch mode (rebuild on changes)
npm run watch
```

## Logs

The orchestrator logs structured JSON with correlation IDs:
- `call_id`: Unique call identifier
- `asterisk_channel_id`: Asterisk channel ID
- `bridge_id`: Bridge ID
- `external_channel_id`: ExternalMedia channel ID
- `media_port`: Allocated media port

Example log entry:
```json
{
  "timestamp": "2024-01-20T20:42:38.791Z",
  "level": "info",
  "message": "StasisStart event received",
  "call_id": "1234567890.123",
  "asterisk_channel_id": "1234567890.123",
  "correlation_id": "call-1234567890.123",
  "caller": "1001"
}
```
