# Bombardilo Crocodallo

VoIP system that bridges Asterisk to real-time voice AI using ARI + External Media (RTP). Full duplex conversational loop with VAD, streaming ASR, LLM, and streaming TTS.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌─────────────┐
│   Softphone │─────▶│   Asterisk   │─────▶│ Orchestrator │─────▶│    Media    │
│  (SIP Call) │      │  (PJSIP/ARI) │      │   (Node.js)  │      │  Service    │
└─────────────┘      └──────────────┘      └──────────────┘      └─────────────┘
                            │                       │                    │
                            │                       │                    │
                            └───────────────────────┴────────────────────┘
                                         RTP (ulaw, 8kHz)
```

### Components

1. **Asterisk**: Handles SIP calls, uses PJSIP and ARI
2. **ARI Orchestrator**: Node.js/TypeScript service managing call lifecycle via ARI WebSocket
3. **Media Service**: Python RTP endpoint handling audio processing, VAD, barge-in, ASR/TTS integration
4. **AI Integration**: Pluggable ASR, LLM, TTS providers (mock implementations included)

## Quick Start

### Prerequisites

- Docker and Docker Compose (for containerized setup)
- OR: Node.js 20+, Python 3.11+, Asterisk 18+ (for native setup)

### Option 1: Docker Compose (Recommended)

1. **Start all services**:
   ```bash
   docker-compose up -d
   ```

2. **Check logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Verify services**:
   - Asterisk: `docker-compose exec asterisk asterisk -rvvv`
   - Orchestrator: Check logs for "ARI Orchestrator ready"
   - Media Service: `curl http://localhost:5000/health`

### Option 2: Native Installation

#### 1. Install Asterisk

**Ubuntu 22.04**:
```bash
sudo apt-get update
sudo apt-get install -y asterisk

# Copy configuration files
sudo cp asterisk/*.conf /etc/asterisk/

# Restart Asterisk
sudo systemctl restart asterisk
```

**Verify Asterisk**:
```bash
sudo asterisk -rvvv
# In CLI: pjsip show endpoints
```

#### 2. Start Orchestrator

```bash
cd orchestrator
npm install
npm run build
npm start
```

Environment variables (optional, defaults shown):
```bash
export ARI_URL=http://localhost:8088
export ARI_USER=voicebot
export ARI_PASS=voicebot_pass
export MEDIA_API_URL=http://localhost:5000
export MEDIA_MODE=full  # or "echo", "tts_only"
```

#### 3. Start Media Service

```bash
cd media-service
pip install -r requirements.txt
python -m src.media_service
```

Environment variables (optional, defaults shown):
```bash
export MEDIA_HOST=127.0.0.1
export MEDIA_PORT_BASE=40000
export MEDIA_API_PORT=5000
export MEDIA_MODE=full  # or "echo", "tts_only"
export ASR_PROVIDER=mock
export LLM_PROVIDER=mock
export TTS_PROVIDER=mock
export SYSTEM_PROMPT="You are a helpful assistant."
```

## Testing with Softphone

### 1. Install Softphone

Recommended: [MicroSIP](https://www.microsip.org/) (Windows), [X-Lite](https://www.counterpath.com/x-lite/) (cross-platform), or [Zoiper](https://www.zoiper.com/) (cross-platform).

### 2. Configure MicroSIP

#### Account Setup

1. **Open MicroSIP** and go to **Account** → **Add Account** (or right-click the tray icon → **Settings**)

2. **Basic Registration Settings**:
   - **Server**: `127.0.0.1:5060` (or your Asterisk IP if running remotely)
   - **Username**: `softphone`
   - **Password**: `softphone_pass`
   - **Domain**: `127.0.0.1` (or leave empty if using IP address)

3. **Click "OK"** to save the account

#### Codec Configuration (Important)

The system uses **G.711 u-law (ulaw)** at 8 kHz. Configure codecs in MicroSIP:

1. Go to **Settings** (or right-click tray icon → **Settings**)
2. Navigate to the **Codecs** section
3. **Move codecs between lists**:
   - **Activate**: `G.711 u-law` (should be in "Codecs activés")
   - **Deactivate**: All other codecs (move to "Codecs disponibles")
   - **Priority**: Ensure `G.711 u-law` is at the top of the active list
4. **Recommended settings**:
   - ✅ **EC (Echo Cancellation)**: Enabled (checked)
   - ❌ **VAD (Voice Activity Detection)**: Disabled (unchecked) - let Asterisk handle this
   - ❌ **Opus 2ch**: Disabled (unchecked)
   - ❌ **Forcer le codec des appels entrants**: Disabled (unchecked)

#### Audio Device Settings

1. In **Settings** → **General**:
   - **Speaker**: Select your preferred output device
   - **Microphone**: Select your preferred input device
   - **Ringtone device**: Select your preferred device
   - ✅ **Software sound level adjustment**: Recommended (helps with audio levels)

2. **Optional but recommended**:
   - ✅ **Microphone amplification**: Enable if your mic is too quiet
   - Test audio levels before making calls

#### Network Settings

For local testing (Asterisk on same machine):
- **Port source**: `0` (auto-assign)
- **RTP Ports**: `0 - 0` (auto-assign)
- **rport**: ✅ Enabled (checked)
- **DNS SRV**: ❌ Disabled (unchecked) - not needed for IP-based registration
- **STUN Server**: Leave empty (not needed for localhost)

If Asterisk is on a different machine:
- Ensure firewall allows UDP port `5060` (SIP) and RTP ports `10000-20000`
- Use the Asterisk machine's IP address instead of `127.0.0.1`

#### Call Settings

1. **Automatic Answer**: Set to **"Bouton de contrôle"** (Control button) - manual answer recommended for testing
2. **Call Forwarding**: Set to **"Non"** (No)
3. **Reject Incoming Calls**: Set to **"Bouton de contrôle"** (Control button)

### 3. Verify Registration

1. **Check registration status**:
   - MicroSIP tray icon should show registered status
   - Green indicator = registered successfully
   - Red indicator = registration failed

2. **Verify in Asterisk** (if you have CLI access):
   ```bash
   # In Asterisk CLI
   pjsip show endpoints
   pjsip show aors
   ```
   You should see `softphone` endpoint registered.

### 4. Place a Call

1. **Dial any number** (e.g., `1000`, `1234`, or any digits)
2. The call will route to the `voicebot` Stasis application
3. **Expected behavior**:
   - **Echo mode**: You should hear your voice echoed back
   - **TTS-only mode**: You should hear: "Hello! I'm ready to help. How can I assist you today?"
   - **Full mode**: You should hear the greeting, then can have a conversation

### 5. Troubleshooting MicroSIP

**Registration fails**:
- Verify Asterisk is running: `docker-compose ps` or `systemctl status asterisk`
- Check credentials match `asterisk/pjsip.conf`
- Verify SIP port 5060 is accessible: `netstat -an | findstr 5060` (Windows) or `netstat -tuln | grep 5060` (Linux)
- Check MicroSIP logs: Right-click tray icon → **Log**

**No audio**:
- Verify codec: Ensure `G.711 u-law` is active and prioritized
- Check audio devices are selected correctly
- Test with echo mode first to validate audio path
- Check Windows firewall allows RTP ports (10000-20000 UDP)

**One-way audio**:
- Check codec negotiation: Both sides must agree on ulaw
- Verify RTP ports are not blocked by firewall
- Check Asterisk logs: `docker-compose logs asterisk` or `/var/log/asterisk/messages`

**"No Answer" or Call Not Connecting**:
- **Check if orchestrator is running**:
  ```bash
  # Docker
  docker-compose ps orchestrator
  docker-compose logs orchestrator
  
  # Native
  # Check if Node.js process is running
  ```
- **Verify orchestrator connected to ARI**:
  - Look for log message: `"ARI Orchestrator ready"` in orchestrator logs
  - Look for log message: `"Stasis application started"` in orchestrator logs
- **Check ARI is accessible**:
  ```bash
  # Test ARI endpoint
  curl http://localhost:8088/ari/api-docs/resources.json
  # Should return JSON, not an error
  ```
- **Verify ARI credentials match**:
  - Check `asterisk/ari.conf` has user `voicebot` with password `voicebot_pass`
  - Check orchestrator environment variables match
- **Check Asterisk is routing calls correctly**:
  ```bash
  # In Asterisk CLI (asterisk -rvvv)
  pjsip show endpoints
  # Should show softphone registered
  
  # When you make a call, check Asterisk CLI for:
  # - "Incoming call to XXXX"
  # - Stasis application being invoked
  ```
- **Verify Stasis application name matches**:
  - `extensions.conf` uses: `Stasis(voicebot)`
  - Orchestrator `ARI_APP` should be: `voicebot`
  - These must match exactly
- **Check for errors in orchestrator logs**:
  - Look for `"Error handling StasisStart"` messages
  - Look for connection errors: `"Failed to start ARI Orchestrator"`
  - Look for media service errors: `"Failed to create media session"`
- **Quick test - verify services are running**:
  ```bash
  # Docker
  docker-compose ps  # All services should be "Up"
  
  # Native - check processes
  # Asterisk: systemctl status asterisk (Linux) or check process
  # Orchestrator: Check Node.js process
  # Media Service: Check Python process or curl http://localhost:5000/health
  ```

## Operation Modes

### Echo Mode (Audio Validation)

Test audio path without AI processing:

```bash
# Set in orchestrator
export MEDIA_MODE=echo

# Set in media-service
export MEDIA_MODE=echo
```

When you speak, you should hear your voice echoed back. This validates:
- RTP path is working
- Codec conversion (ulaw ↔ PCM16) is correct
- Audio latency is acceptable

### TTS-Only Mode

Test TTS without ASR/LLM:

```bash
export MEDIA_MODE=tts_only
```

You'll hear the greeting, but no conversation.

### Full Mode (Default)

Full conversational loop:

```bash
export MEDIA_MODE=full
```

- VAD detects speech
- ASR transcribes speech
- LLM generates response
- TTS synthesizes response
- Barge-in: speech during TTS stops TTS and resumes ASR

## Configuration

### Asterisk

Configuration files in `asterisk/`:
- `http.conf`: ARI HTTP server (port 8088, localhost)
- `ari.conf`: ARI user credentials
- `extensions.conf`: Call routing to Stasis app
- `pjsip.conf`: SIP endpoints (softphone, trunk placeholder)

**Security**: ARI is bound to `127.0.0.1:8088` by default. For production:
- Bind to private network interface if needed
- Use reverse proxy with authentication
- Configure firewall rules

### Orchestrator

Environment variables:
- `ARI_URL`: ARI HTTP server URL
- `ARI_USER`: ARI username
- `ARI_PASS`: ARI password
- `ARI_APP`: Stasis application name
- `MEDIA_API_URL`: Media service HTTP API URL
- `MEDIA_MODE`: Operation mode (`echo`, `tts_only`, `full`)
- `LOG_LEVEL`: Log level (`debug`, `info`, `warn`, `error`)

### Media Service

Environment variables:
- `MEDIA_HOST`: Host to bind to
- `MEDIA_PORT_BASE`: Base port for RTP sessions
- `MEDIA_API_PORT`: HTTP API port
- `MEDIA_MODE`: Operation mode
- `ASR_PROVIDER`: ASR provider (`mock` or real provider)
- `LLM_PROVIDER`: LLM provider (`mock` or real provider)
- `TTS_PROVIDER`: TTS provider (`mock` or real provider)
- `SYSTEM_PROMPT`: System prompt for LLM

## AI Providers

### Mock Providers (Default)

No external keys required. Good for testing:
- **ASR**: Returns predefined transcriptions
- **LLM**: Returns simple mock responses
- **TTS**: Generates tone audio

### Real Providers

To add real providers, implement the interfaces in `media-service/src/ai_providers.py`:
- `ASRClient`: Streaming ASR interface
- `LLMClient`: LLM interface
- `TTSClient`: Streaming TTS interface

Example (pseudo-code):
```python
class OpenAIASRClient(ASRClient):
    async def start_stream(self) -> str:
        # Create OpenAI Whisper stream
        ...
    
    async def send_audio(self, stream_id: str, audio: bytes) -> None:
        # Send audio to OpenAI
        ...
```

Then update `create_asr_client()`, `create_llm_client()`, `create_tts_client()` in `ai_providers.py`.

## Troubleshooting

### One-Way Audio

**Symptoms**: You can hear the AI, but it can't hear you (or vice versa).

**Causes**:
1. **RTP path issue**: Check firewall rules, NAT traversal
2. **Codec mismatch**: Verify ulaw is negotiated
3. **Port allocation**: Check media service ports are accessible

**Solutions**:
- Check Asterisk RTP ports: `core show settings` (look for `rtpstart`/`rtpend`)
- Verify media service is receiving packets: Check logs for "RTP packet received"
- Test with echo mode first to validate audio path
- Check firewall: `sudo ufw allow 10000:20000/udp` (RTP range)

### Codec Mismatch

**Symptoms**: No audio, or garbled audio.

**Solutions**:
- Verify `pjsip.conf` allows `ulaw`: `allow=ulaw`
- Check Asterisk codec negotiation: `pjsip show endpoints`
- Ensure media service is using ulaw: `AUDIO_FORMAT=ulaw`

### NAT Issues

**Symptoms**: Works locally but not from external network.

**Solutions**:
- Configure Asterisk NAT settings in `pjsip.conf`:
  ```
  [transport-udp]
  type=transport
  external_media_address=YOUR_PUBLIC_IP
  external_signaling_address=YOUR_PUBLIC_IP
  ```
- Use STUN/TURN for complex NAT scenarios
- Consider using WebRTC gateway for browser-based clients

### Port Issues

**Symptoms**: Calls fail to connect, or media service errors.

**Solutions**:
- Check port availability: `netstat -tuln | grep 5060` (SIP), `netstat -tuln | grep 8088` (ARI)
- Verify RTP port range: `10000-20000` (Asterisk), `40000+` (Media Service)
- Check Docker port mappings if using containers

### ARI Connection Failed

**Symptoms**: Orchestrator can't connect to ARI.

**Solutions**:
- Verify `http.conf` is loaded: `ari show status` in Asterisk CLI
- Check ARI credentials in `ari.conf` match orchestrator env vars
- Verify ARI is bound to correct interface: `bindaddr=127.0.0.1` in `http.conf`
- Test ARI endpoint: `curl http://localhost:8088/ari/api-docs/resources.json` (should return JSON)

### Media Service Not Creating Sessions

**Symptoms**: Calls connect but no audio processing.

**Solutions**:
- Check media service HTTP API: `curl http://localhost:5000/health`
- Verify orchestrator can reach media service: Check `MEDIA_API_URL` env var
- Check media service logs for session creation errors
- Verify port allocation: Check `MEDIA_PORT_BASE` is not conflicting

### VAD Not Detecting Speech

**Symptoms**: AI doesn't respond to speech.

**Solutions**:
- Check VAD threshold: Adjust `energy_threshold` in `vad.py` (lower = more sensitive)
- Verify audio levels: Check if audio is too quiet
- Test with echo mode to verify audio is reaching media service
- Check logs for VAD decisions: Look for "Speech detected" messages

### Barge-In Not Working

**Symptoms**: TTS continues playing when you speak.

**Solutions**:
- Verify VAD is running: Check logs for VAD activity
- Check TTS stop logic: Ensure `tts_active` flag is checked
- Verify ASR stream restart: Check logs for "Barge-in detected"

## Validation Steps

### 1. Audio Path Validation (Echo Mode)

```bash
# Set both services to echo mode
export MEDIA_MODE=echo

# Place a call
# Speak into phone
# You should hear your voice echoed back
```

### 2. TTS Validation (TTS-Only Mode)

```bash
export MEDIA_MODE=tts_only

# Place a call
# You should hear: "Hello! I'm ready to help. How can I assist you today?"
```

### 3. Full Loop Validation (Full Mode)

```bash
export MEDIA_MODE=full

# Place a call
# Wait for greeting
# Say: "Hello"
# You should hear a response
# Try barge-in: speak while AI is talking
```

## Logs and Monitoring

### Structured Logs

All services log structured JSON with correlation IDs:
- `call_id`: Unique call identifier
- `asterisk_channel_id`: Asterisk channel ID
- `bridge_id`: Bridge ID
- `external_channel_id`: ExternalMedia channel ID
- `session_id`: Media session ID
- `media_port`: Allocated media port

### Log Locations

- **Asterisk**: `docker-compose logs asterisk` or `/var/log/asterisk/`
- **Orchestrator**: `docker-compose logs orchestrator` or console output
- **Media Service**: `docker-compose logs media-service` or console output

### Metrics

Basic metrics available via logs:
- Call count: Count of "StasisStart" events
- Session count: Active sessions in media service
- Port usage: Allocated ports vs. available

## Development

### Building

```bash
# Orchestrator
cd orchestrator
npm install
npm run build

# Media Service
cd media-service
pip install -r requirements.txt
```

### Running in Development

```bash
# Orchestrator (with watch)
cd orchestrator
npm run watch  # In one terminal
npm run dev    # In another terminal

# Media Service (direct)
cd media-service
python -m src.media_service
```

### Adding Real AI Providers

1. Implement interface in `media-service/src/ai_providers.py`
2. Add provider creation logic in factory functions
3. Set environment variables: `ASR_PROVIDER=your_provider`
4. Add API keys via environment variables

## Security Notes

1. **ARI**: Bound to localhost by default. For production, use reverse proxy with auth.
2. **SIP**: Bound to all interfaces. Consider firewall rules and SIP over TLS.
3. **RTP**: Open port range. Consider RTP encryption (SRTP) for production.
4. **Credentials**: Change default passwords in `ari.conf` and `pjsip.conf`.
5. **API Keys**: Store AI provider keys securely (use secrets management).

## Next Steps

- [ ] Add real ASR provider (e.g., Deepgram, AssemblyAI)
- [ ] Add real LLM provider (e.g., OpenAI, Anthropic)
- [ ] Add real TTS provider (e.g., ElevenLabs, Azure TTS)
- [ ] Implement SRTP for encrypted RTP
- [ ] Add metrics collection (Prometheus)
- [ ] Add health checks and monitoring
- [ ] Production hardening (error recovery, rate limiting)

## License

MIT

## Support

For issues, check:
1. Troubleshooting section above
2. Service logs
3. Asterisk CLI: `asterisk -rvvv`
