# Asterisk Configuration

This directory contains minimal working Asterisk configurations for the Bombardilo Crocodallo voice AI system.

## Files

- `http.conf`: Enables ARI HTTP server on port 8088 (bound to localhost)
- `ari.conf`: Creates ARI user "voicebot" for orchestrator access
- `extensions.conf`: Routes inbound calls to Stasis application "voicebot"
- `pjsip.conf`: Defines local test endpoint and trunk placeholder

## Installation

### Option 1: Docker (Recommended for MVP)

See the main `docker-compose.yml` in the project root.

### Option 2: Native Installation (Ubuntu 22.04)

```bash
# Install Asterisk
sudo apt-get update
sudo apt-get install -y asterisk

# Copy configuration files
sudo cp asterisk/*.conf /etc/asterisk/

# Restart Asterisk
sudo systemctl restart asterisk

# Check status
sudo systemctl status asterisk
sudo asterisk -rvvv  # Connect to CLI
```

## Security Notes

1. **ARI Binding**: ARI is bound to `127.0.0.1:8088` by default. For production:
   - Bind to a private network interface if needed
   - Use a reverse proxy (nginx/traefik) with authentication
   - Consider firewall rules to restrict access

2. **ARI Password**: Change the default password in `ari.conf` for production.

3. **SIP Port**: SIP is bound to `0.0.0.0:5060` (all interfaces). For production:
   - Consider binding to specific interfaces
   - Use firewall rules to restrict access
   - Consider using SIP over TLS (SIPS)

4. **Firewall Example**:
   ```bash
   # Allow SIP (UDP 5060)
   sudo ufw allow 5060/udp
   
   # Allow RTP range (10000-20000)
   sudo ufw allow 10000:20000/udp
   
   # ARI should only be accessible from localhost (already bound to 127.0.0.1)
   ```

## Testing with Softphone

1. Install a softphone (e.g., X-Lite, Zoiper, MicroSIP)
2. Configure registration:
   - Server: `127.0.0.1:5060` (or your Asterisk IP)
   - Username: `softphone`
   - Password: `softphone_pass`
   - Domain: `127.0.0.1`
3. Register the softphone
4. Dial any number (e.g., `1000`) - it will route to the voicebot Stasis application

## Verification

```bash
# Connect to Asterisk CLI
sudo asterisk -rvvv

# Check PJSIP endpoints
pjsip show endpoints

# Check ARI status
ari show status

# Monitor calls
core show channels
```

## Troubleshooting

- **SIP registration fails**: Check firewall, verify credentials in `pjsip.conf`
- **ARI connection fails**: Verify `http.conf` and `ari.conf`, check if port 8088 is accessible
- **No audio**: Check RTP ports (10000-20000), verify codec negotiation (ulaw)
