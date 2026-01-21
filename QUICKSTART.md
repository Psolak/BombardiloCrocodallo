# Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- OR: Node.js 20+, Python 3.11+, Asterisk 18+ (for native setup)

## Docker Compose (Easiest)

1. **Start all services**:
   ```bash
   docker-compose up -d
   ```

2. **Check services are running**:
   ```bash
   docker-compose ps
   ```

3. **View logs**:
   ```bash
   # All services
   docker-compose logs -f
   
   # Individual service
   docker-compose logs -f asterisk
   docker-compose logs -f orchestrator
   docker-compose logs -f media-service
   ```

4. **Verify services**:
   ```bash
   # Check Asterisk
   docker-compose exec asterisk asterisk -rvvv
   # In CLI: pjsip show endpoints
   
   # Check Media Service API
   curl http://localhost:5000/health
   # Should return: {"status": "healthy"}
   ```

## Configure Softphone

1. **Install a softphone**:
   - Windows: [MicroSIP](https://www.microsip.org/)
   - Mac/Linux: [X-Lite](https://www.counterpath.com/x-lite/) or [Zoiper](https://www.zoiper.com/)

2. **Register with Asterisk**:
   - Server: `127.0.0.1:5060`
   - Username: `softphone`
   - Password: `softphone_pass`
   - Domain: `127.0.0.1`

3. **Place a test call**:
   - Dial any number (e.g., `1000`)
   - You should hear the greeting (in `full` or `tts_only` mode)

## Testing Modes

### 1. Echo Mode (Audio Validation)

Test that audio path is working:

```bash
# Stop services
docker-compose down

# Edit docker-compose.yml, set:
#   MEDIA_MODE=echo  (in both orchestrator and media-service)

# Or use environment override
docker-compose run -e MEDIA_MODE=echo orchestrator
docker-compose run -e MEDIA_MODE=echo media-service

# Start services
docker-compose up -d

# Place a call and speak - you should hear your voice echoed back
```

### 2. TTS-Only Mode

Test TTS without ASR/LLM:

```bash
# Set MEDIA_MODE=tts_only in docker-compose.yml
docker-compose up -d

# Place a call - you'll hear the greeting
```

### 3. Full Mode (Default)

Full conversational AI:

```bash
# MEDIA_MODE=full (default)
docker-compose up -d

# Place a call
# Wait for greeting
# Say: "Hello"
# AI should respond
# Try barge-in: speak while AI is talking
```

## Native Setup (Without Docker)

### 1. Install Asterisk

**Ubuntu 22.04**:
```bash
sudo apt-get update
sudo apt-get install -y asterisk
sudo cp asterisk/*.conf /etc/asterisk/
sudo systemctl restart asterisk
```

### 2. Start Orchestrator

```bash
cd orchestrator
npm install
npm run build

# Set environment variables (optional)
export ARI_URL=http://localhost:8088
export ARI_USER=voicebot
export ARI_PASS=voicebot_pass
export MEDIA_API_URL=http://localhost:5000
export MEDIA_MODE=full

npm start
```

### 3. Start Media Service

```bash
cd media-service
pip install -r requirements.txt

# Set environment variables (optional)
export MEDIA_HOST=127.0.0.1
export MEDIA_PORT_BASE=40000
export MEDIA_API_PORT=5000
export MEDIA_MODE=full
export ASR_PROVIDER=mock
export LLM_PROVIDER=mock
export TTS_PROVIDER=mock

python -m src.media_service
```

## Troubleshooting

### Services Won't Start

```bash
# Check if ports are in use
netstat -tuln | grep 5060  # SIP
netstat -tuln | grep 8088  # ARI
netstat -tuln | grep 5000  # Media API

# Check Docker logs
docker-compose logs asterisk
docker-compose logs orchestrator
docker-compose logs media-service
```

### No Audio

1. **Check echo mode first**:
   ```bash
   MEDIA_MODE=echo docker-compose up -d
   ```

2. **Check RTP ports**:
   ```bash
   # Asterisk RTP range: 10000-20000
   # Media service: 40000+
   sudo ufw allow 10000:20000/udp
   sudo ufw allow 40000:41000/udp
   ```

3. **Check codec negotiation**:
   ```bash
   docker-compose exec asterisk asterisk -rvvv
   # In CLI: pjsip show endpoints
   ```

### ARI Connection Failed

```bash
# Test ARI endpoint
curl http://localhost:8088/ari/api-docs/resources.json

# Check ARI credentials
docker-compose exec asterisk cat /etc/asterisk/ari.conf
```

### Media Service Not Responding

```bash
# Test health endpoint
curl http://localhost:5000/health

# Check logs
docker-compose logs media-service

# Verify port is accessible
netstat -tuln | grep 5000
```

## Next Steps

- See [README.md](README.md) for full documentation
- Configure real AI providers (see `media-service/src/ai_providers.py`)
- Add production security (firewall, reverse proxy, SRTP)
