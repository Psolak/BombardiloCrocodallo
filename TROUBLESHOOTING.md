# Troubleshooting "No Answer" Issue

## Quick Diagnostic Steps

### 1. Verify Services Are Running

```powershell
# Check all services
docker-compose ps

# Check orchestrator logs
docker-compose logs orchestrator --tail 20

# Check Asterisk logs
docker-compose logs asterisk --tail 20
```

### 2. Verify Orchestrator is Connected

Look for these log messages in orchestrator:
- ✅ `"ARI Orchestrator ready"` - Good!
- ✅ `"Stasis application started"` - Good!
- ❌ `"Failed to start ARI Orchestrator"` - Problem!

### 3. Check Softphone Registration

```powershell
# Check if softphone is registered
docker-compose exec asterisk asterisk -rx "pjsip show endpoints"

# Look for "softphone" endpoint with "Qualified" status
# If it shows "NonQual", the softphone is NOT registered
```

**In MicroSIP:**
- Tray icon should be **green** = registered
- Tray icon is **red** = not registered
- Right-click tray icon → **Log** to see registration errors

### 4. Test ARI Connection

```powershell
# Test if ARI is accessible
curl http://localhost:8088/ari/api-docs/resources.json

# Should return JSON, not an error
```

### 5. Monitor Calls in Real-Time

**Option A: Watch logs while making a call**
```powershell
# Terminal 1: Watch orchestrator
docker-compose logs -f orchestrator

# Terminal 2: Watch Asterisk
docker-compose logs -f asterisk

# Terminal 3: Make a call from MicroSIP
# Watch for "StasisStart" events in orchestrator logs
```

**Option B: Use Asterisk CLI**
```powershell
# Connect to Asterisk CLI
docker-compose exec asterisk asterisk -rvvv

# In CLI, you'll see real-time call events
# Make a call and watch for:
# - "Incoming call to XXXX"
# - Stasis application invocation
```

### 6. Common Issues and Fixes

#### Issue: Softphone Not Registered
**Symptoms:**
- MicroSIP tray icon is red
- `pjsip show endpoints` shows "NonQual"

**Fixes:**
1. Double-check credentials in MicroSIP:
   - Server: `127.0.0.1:5060`
   - Username: `softphone`
   - Password: `softphone_pass`
   - Domain: `127.0.0.1`
2. Check Asterisk is running: `docker-compose ps asterisk`
3. Check SIP port is accessible: `netstat -an | findstr 5060`
4. Check MicroSIP logs: Right-click tray → Log

#### Issue: Orchestrator Not Receiving Calls
**Symptoms:**
- Call connects but immediately says "no answer"
- No "StasisStart" events in orchestrator logs

**Fixes:**
1. Verify Stasis app name matches:
   - `extensions.conf`: `Stasis(voicebot)`
   - Orchestrator `ARI_APP`: `voicebot`
2. Check dialplan is loaded:
   ```powershell
   docker-compose exec asterisk asterisk -rx "dialplan show inbound"
   ```
3. Verify ARI is enabled:
   ```powershell
   docker-compose exec asterisk asterisk -rx "ari show status"
   ```
4. Check orchestrator can reach Asterisk:
   - In Docker: Uses service name `asterisk:8088`
   - Native: Uses `localhost:8088`

#### Issue: Media Service Not Available
**Symptoms:**
- Call connects but no audio
- Orchestrator logs show: `"Failed to create media session"`

**Fixes:**
1. Check media service is running: `docker-compose ps media-service`
2. Test media service API:
   ```powershell
   curl http://localhost:5000/health
   ```
3. Check media service logs:
   ```powershell
   docker-compose logs media-service --tail 20
   ```

### 7. Step-by-Step Test Call Procedure

1. **Start all services:**
   ```powershell
   docker-compose up -d
   ```

2. **Verify orchestrator is ready:**
   ```powershell
   docker-compose logs orchestrator | Select-String "ready"
   # Should see: "ARI Orchestrator ready"
   ```

3. **Check softphone registration:**
   - MicroSIP tray icon should be green
   - Or: `docker-compose exec asterisk asterisk -rx "pjsip show endpoints"`

4. **Start monitoring:**
   ```powershell
   # In one terminal
   docker-compose logs -f orchestrator
   ```

5. **Make a test call:**
   - Dial `1000` from MicroSIP
   - Watch for log messages:
     - `"StasisStart event received"` ✅
     - `"Channel answered"` ✅
     - `"Media session created"` ✅
     - `"Channels added to bridge"` ✅

6. **If no logs appear:**
   - Check Asterisk logs: `docker-compose logs asterisk --tail 50`
   - Check if call reached Asterisk
   - Verify dialplan routing

### 8. Windows-Specific Issues

**Firewall blocking RTP:**
```powershell
# Allow RTP ports (10000-20000 UDP)
New-NetFirewallRule -DisplayName "Asterisk RTP" -Direction Inbound -Protocol UDP -LocalPort 10000-20000 -Action Allow
```

**Port conflicts:**
```powershell
# Check if port 5060 is in use
netstat -an | findstr 5060

# Check if port 8088 is in use
netstat -an | findstr 8088
```

### 9. Debug Mode

Enable debug logging:
```powershell
# Set LOG_LEVEL=debug in docker-compose.yml orchestrator service
# Or set environment variable
$env:LOG_LEVEL="debug"
docker-compose up -d orchestrator
```

### 10. Still Not Working?

Collect diagnostic information:
```powershell
# Save all logs
docker-compose logs > all-logs.txt

# Check service status
docker-compose ps > services-status.txt

# Check Asterisk configuration
docker-compose exec asterisk asterisk -rx "core show settings" > asterisk-settings.txt
docker-compose exec asterisk asterisk -rx "pjsip show endpoints" > pjsip-endpoints.txt
docker-compose exec asterisk asterisk -rx "dialplan show inbound" > dialplan.txt
```
