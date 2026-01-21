import * as ari from 'ari-client';
import { createLogger, format, transports } from 'winston';
import { CallSession } from './call-session';

// Configuration from environment variables
const config = {
  ari: {
    url: process.env.ARI_URL || 'http://localhost:8088',
    username: process.env.ARI_USER || 'voicebot',
    password: process.env.ARI_PASS || 'voicebot_pass',
    app: process.env.ARI_APP || 'voicebot',
  },
  media: {
    host: process.env.MEDIA_HOST || '127.0.0.1',
    ip: process.env.MEDIA_IP || process.env.MEDIA_HOST || '127.0.0.1', // Use IP for RTP if provided
    portBase: parseInt(process.env.MEDIA_PORT_BASE || '40000', 10),
    apiPort: parseInt(process.env.MEDIA_API_PORT || '5000', 10),
    apiUrl: process.env.MEDIA_API_URL || `http://${process.env.MEDIA_HOST || '127.0.0.1'}:${process.env.MEDIA_API_PORT || '5000'}`,
    audioFormat: process.env.AUDIO_FORMAT || 'ulaw',
    mode: process.env.MEDIA_MODE || 'full', // "echo", "tts_only", "full"
  },
  logLevel: process.env.LOG_LEVEL || 'info',
};

// Structured logger with correlation ID support
const logger = createLogger({
  level: config.logLevel,
  format: format.combine(
    format.timestamp(),
    format.errors({ stack: true }),
    format.json()
  ),
  defaultMeta: { service: 'orchestrator' },
  transports: [
    new transports.Console({
      format: format.combine(
        format.colorize(),
        format.printf(({ timestamp, level, message, ...meta }) => {
          const metaStr = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : '';
          return `${timestamp} [${level}]: ${message} ${metaStr}`;
        })
      ),
    }),
  ],
});

// Note: Port allocation is now handled by the media service via HTTP API

// Active call sessions
const activeSessions = new Map<string, CallSession>();

// Store ARI client globally to keep connection alive
let ariClient: any = null;

async function main() {
  logger.info('Starting ARI Orchestrator', { config: { ...config, ari: { ...config.ari, password: '***' } } });

  try {
    // Connect to ARI
    ariClient = await ari.connect(config.ari.url, config.ari.username, config.ari.password);

    logger.info('Connected to ARI', { app: config.ari.app });

    // Start the Stasis application
    await ariClient.start(config.ari.app);
    logger.info('Stasis application started', { app: config.ari.app });

    // Start listening for Stasis events
    ariClient.on('StasisStart', async (event: any, channel: any) => {
      const callId = event.channel.id;
      const channelName = event.channel.name || '';
      
      // Handle StasisStart events from ExternalMedia (UnicastRTP) channels
      // These are created by us - we need to track them but not create new sessions
      if (channelName.startsWith('UnicastRTP/')) {
        logger.debug('StasisStart from ExternalMedia channel (expected)', {
          call_id: callId,
          channel_name: channelName,
        });
        // Don't return - we want to answer this channel and add it to the bridge
        // But we need to find the existing session that created this channel
        // For now, just log it and let it continue - the channel should already be in the session
        return;
      }

      const correlationId = `call-${callId}`;

      logger.info('StasisStart event received', {
        call_id: callId,
        asterisk_channel_id: callId,
        correlation_id: correlationId,
        caller: event.channel.caller?.number || 'unknown',
        channel_name: channelName,
      });

      try {
        // Get channel state before answering
        const channelState = event.channel.state;
        logger.debug('Channel state in StasisStart', { 
          call_id: callId, 
          state: channelState,
          channel_name: event.channel.name 
        });
        
        // Answer the channel - it should be in Ringing state for incoming calls
        // For channels created via ARI, they might be in Down state and can't be answered
        if (channelState === 'Ringing' || channelState === 'Ring') {
          await channel.answer();
          logger.info('Channel answered', { call_id: callId, previous_state: channelState });
        } else if (channelState === 'Up') {
          logger.info('Channel already answered', { call_id: callId });
        } else {
          logger.warn('Channel in unexpected state, attempting to answer anyway', { 
            call_id: callId, 
            state: channelState 
          });
          try {
            await channel.answer();
            logger.info('Channel answered despite unexpected state', { call_id: callId });
          } catch (answerError) {
            logger.error('Failed to answer channel', { 
              call_id: callId, 
              state: channelState,
              error: answerError instanceof Error ? answerError.message : String(answerError)
            });
            // Continue anyway - might be a test channel or already answered
          }
        }

        // Get channel information to determine RTP address
        // For ExternalMedia, Asterisk will send RTP to the media service
        // The media service will learn Asterisk's source address from incoming packets
        // We just need to allocate a port and tell Asterisk where to send
        
        // Create media session via HTTP API
        // Note: remote_addr will be learned from first RTP packet
        const mediaApiUrl = `${config.media.apiUrl}/sessions`;
        const mediaResponse = await fetch(mediaApiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: callId,
            remote_host: '127.0.0.1', // Will be updated from first RTP packet
            remote_port: 10000, // Placeholder, will be learned from RTP
            mode: config.media.mode,
          }),
        });

        if (!mediaResponse.ok) {
          throw new Error(`Failed to create media session: ${mediaResponse.statusText}`);
        }

        const mediaData = await mediaResponse.json() as { local_port: number; session_id: string; status: string };
        const mediaPort = mediaData.local_port;
        logger.info('Media session created', {
          call_id: callId,
          media_port: mediaPort,
          session_id: mediaData.session_id,
        });

        // Create a mixing bridge
        const bridge = await ariClient.bridges.create({ type: 'mixing' });
        logger.info('Bridge created', {
          call_id: callId,
          bridge_id: bridge.id,
        });

        // Create ExternalMedia channel targeting the Media Service
        // Note: external_host must be in format "host:port" (no rtp:// prefix)
        // Use IP address instead of hostname for better reliability
        // direction: "both" enables bidirectional audio (required for echo)
        const externalHost = `${config.media.ip}:${mediaPort}`;
        const externalChannel = await ariClient.channels.externalMedia({
          app: config.ari.app,
          external_host: externalHost,
          format: config.media.audioFormat,
          direction: 'both', // Enable bidirectional audio
          transport: 'udp', // Use UDP for RTP
          encapsulation: 'rtp', // RTP encapsulation
        });

        logger.info('ExternalMedia channel created', {
          call_id: callId,
          external_channel_id: externalChannel.id,
          media_port: mediaPort,
          external_host: externalHost,
          using_ip: config.media.ip !== config.media.host,
        });

        // Wait a moment for the ExternalMedia channel to be ready
        await new Promise(resolve => setTimeout(resolve, 500));

        // ExternalMedia channels should automatically start when added to bridge
        // But let's verify the channel state and try to start it explicitly
        try {
          const channelInfo = await ariClient.channels.get({ channelId: externalChannel.id });
          logger.info('ExternalMedia channel state before bridge', {
            call_id: callId,
            channel_id: externalChannel.id,
            state: channelInfo.state,
            name: channelInfo.name,
          });
          
          // Try to answer/start the ExternalMedia channel if it's not already up
          if (channelInfo.state !== 'Up') {
            try {
              await externalChannel.answer();
              logger.info('Answered ExternalMedia channel', {
                call_id: callId,
                channel_id: externalChannel.id,
              });
            } catch (answerError) {
              logger.debug('Could not answer ExternalMedia channel (may not be needed)', {
                call_id: callId,
                error: answerError instanceof Error ? answerError.message : String(answerError),
              });
            }
          }
        } catch (error) {
          logger.warn('Could not get ExternalMedia channel info', {
            call_id: callId,
            error: error instanceof Error ? error.message : String(error),
          });
        }

        // Add both channels to the bridge
        try {
          await bridge.addChannel({ channel: channel.id });
          logger.debug('SIP channel added to bridge', { call_id: callId });
        } catch (error) {
          logger.error('Failed to add SIP channel to bridge', {
            call_id: callId,
            error: error instanceof Error ? error.message : String(error),
          });
          throw error;
        }

        try {
          await bridge.addChannel({ channel: externalChannel.id });
          logger.debug('ExternalMedia channel added to bridge', { call_id: callId });
        } catch (error) {
          logger.error('Failed to add ExternalMedia channel to bridge', {
            call_id: callId,
            external_channel_id: externalChannel.id,
            error: error instanceof Error ? error.message : String(error),
          });
          throw error;
        }

        logger.info('Channels added to bridge', {
          call_id: callId,
          bridge_id: bridge.id,
          asterisk_channel_id: channel.id,
          external_channel_id: externalChannel.id,
        });

        // Verify both channels are in the bridge and check their states
        try {
          const bridgeInfo = await ariClient.bridges.get({ bridgeId: bridge.id });
          logger.info('Bridge info after adding channels', {
            call_id: callId,
            bridge_id: bridge.id,
            bridge_type: bridgeInfo.bridge_type,
            bridge_class: bridgeInfo.bridge_class,
            channels: bridgeInfo.channels || [],
          });
          
          // Check ExternalMedia channel state one more time and get channel variables
          const extChannelInfo = await ariClient.channels.get({ channelId: externalChannel.id });
          logger.info('ExternalMedia channel state after bridge', {
            call_id: callId,
            channel_id: externalChannel.id,
            state: extChannelInfo.state,
            name: extChannelInfo.name,
          });
          
          // Check RTP-related channel variables and update media service with Asterisk's RTP address
          let asteriskRtpAddress: string | null = null;
          let asteriskRtpPort: number | null = null;
          try {
            const channelVars = await ariClient.channels.getChannelVar({
              channelId: externalChannel.id,
              variable: 'UNICASTRTP_LOCAL_ADDRESS'
            });
            asteriskRtpAddress = channelVars.value || null;
            logger.info('ExternalMedia RTP variables', {
              call_id: callId,
              channel_id: externalChannel.id,
              UNICASTRTP_LOCAL_ADDRESS: asteriskRtpAddress || 'not set',
            });
            
            const channelVarsPort = await ariClient.channels.getChannelVar({
              channelId: externalChannel.id,
              variable: 'UNICASTRTP_LOCAL_PORT'
            });
            asteriskRtpPort = channelVarsPort.value ? parseInt(channelVarsPort.value, 10) : null;
            logger.info('ExternalMedia RTP port', {
              call_id: callId,
              channel_id: externalChannel.id,
              UNICASTRTP_LOCAL_PORT: asteriskRtpPort || 'not set',
            });
            
            // Update media service with Asterisk's RTP address so it can send RTP back
            logger.debug('Checking if RTP update needed', {
              call_id: callId,
              asteriskRtpAddress,
              asteriskRtpPort,
              hasAddress: !!asteriskRtpAddress,
              hasPort: !!asteriskRtpPort,
            });
            if (asteriskRtpAddress && asteriskRtpPort) {
              logger.info('Attempting to update media service RTP address', {
                call_id: callId,
                api_url: `${config.media.apiUrl}/sessions/${callId}/rtp-address`,
                remote_host: asteriskRtpAddress,
                remote_port: asteriskRtpPort,
              });
              try {
                const updateResponse = await fetch(`${config.media.apiUrl}/sessions/${callId}/rtp-address`, {
                  method: 'PUT',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    remote_host: asteriskRtpAddress,
                    remote_port: asteriskRtpPort,
                  }),
                });
                if (updateResponse.ok) {
                  logger.info('Updated media service with Asterisk RTP address', {
                    call_id: callId,
                    rtp_address: `${asteriskRtpAddress}:${asteriskRtpPort}`,
                  });
                } else {
                  logger.warn('Failed to update media service RTP address', {
                    call_id: callId,
                    status: updateResponse.status,
                  });
                }
              } catch (updateError) {
                logger.warn('Could not update media service RTP address', {
                  call_id: callId,
                  error: updateError instanceof Error ? updateError.message : String(updateError),
                  stack: updateError instanceof Error ? updateError.stack : undefined,
                });
              }
            }
          } catch (varError) {
            logger.debug('Could not get RTP channel variables', {
              call_id: callId,
              error: varError instanceof Error ? varError.message : String(varError),
            });
          }
        } catch (error) {
          logger.warn('Could not get bridge or channel info', {
            call_id: callId,
            error: error instanceof Error ? error.message : String(error),
          });
        }

        // Create and store call session FIRST to prevent race conditions
        const session = new CallSession({
          callId,
          asteriskChannelId: channel.id,
          bridgeId: bridge.id,
          externalChannelId: externalChannel.id,
          mediaPort,
          channel,
          bridge,
          externalChannel,
          client: ariClient,
        });

        activeSessions.set(callId, session);
        
        logger.info('Call session stored', {
          call_id: callId,
          asterisk_channel_id: channel.id,
          bridge_id: bridge.id,
          external_channel_id: externalChannel.id,
        });

        // Handle channel hangup
        channel.on('ChannelHangupRequest', async () => {
          logger.info('Channel hangup requested', { call_id: callId });
          await cleanupSession(callId);
        });

        // Handle external channel hangup
        externalChannel.on('ChannelHangupRequest', async () => {
          logger.info('External channel hangup requested', { call_id: callId });
          await cleanupSession(callId);
        });

        // Handle StasisEnd - but only if session still exists (not already cleaned up)
        channel.on('StasisEnd', async () => {
          logger.info('StasisEnd event received', { call_id: callId });
          // Only cleanup if session still exists (might have been cleaned up already)
          if (activeSessions.has(callId)) {
            await cleanupSession(callId);
          } else {
            logger.debug('StasisEnd received but session already cleaned up', { call_id: callId });
          }
        });

      } catch (error) {
        logger.error('Error handling StasisStart', {
          call_id: callId,
          error: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : undefined,
        });
        try {
          await channel.hangup();
        } catch (hangupError) {
          logger.error('Error hanging up channel', { call_id: callId, error: hangupError });
        }
      }
    });

    logger.info('ARI Orchestrator ready', { app: config.ari.app });

    // Keep the process alive - the ARI client connection should keep the event loop alive
    // But we'll also add error handling for the connection
    ariClient.on('error', (error: Error) => {
      logger.error('ARI client error', {
        error: error.message,
        stack: error.stack,
      });
    });

    // Handle connection close
    ariClient.on('close', () => {
      logger.warn('ARI connection closed, exiting');
      process.exit(1);
    });

  } catch (error) {
    logger.error('Failed to start ARI Orchestrator', {
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });
    process.exit(1);
  }
}

async function cleanupSession(callId: string) {
  const session = activeSessions.get(callId);
  if (!session) {
    // Session already cleaned up or doesn't exist - this is normal if multiple events fire
    logger.debug('Cleanup called but session not found', { call_id: callId });
    return;
  }

  // Remove from active sessions immediately to prevent duplicate cleanup
  activeSessions.delete(callId);
  
  logger.debug('Starting session cleanup', { call_id: callId });

  logger.info('Cleaning up session', {
    call_id: callId,
    asterisk_channel_id: session.asteriskChannelId,
    bridge_id: session.bridgeId,
    external_channel_id: session.externalChannelId,
    media_port: session.mediaPort,
  });

  try {
    // Remove channels from bridge (idempotent)
    try {
      await session.bridge.removeChannel({ channel: session.asteriskChannelId });
    } catch (error) {
      logger.warn('Error removing channel from bridge (may already be removed)', {
        call_id: callId,
        error: error instanceof Error ? error.message : String(error),
      });
    }

    try {
      await session.bridge.removeChannel({ channel: session.externalChannelId });
    } catch (error) {
      logger.warn('Error removing external channel from bridge (may already be removed)', {
        call_id: callId,
        error: error instanceof Error ? error.message : String(error),
      });
    }

    // Destroy bridge (idempotent)
    try {
      await session.bridge.destroy();
      logger.info('Bridge destroyed', { call_id: callId, bridge_id: session.bridgeId });
    } catch (error) {
      logger.warn('Error destroying bridge (may already be destroyed)', {
        call_id: callId,
        error: error instanceof Error ? error.message : String(error),
      });
    }

    // Hangup channels (idempotent)
    try {
      await session.externalChannel.hangup();
    } catch (error) {
      logger.warn('Error hanging up external channel (may already be hung up)', {
        call_id: callId,
        error: error instanceof Error ? error.message : String(error),
      });
    }

    // Stop media session via HTTP API
    try {
      const mediaApiUrl = `${config.media.apiUrl}/sessions`;
      const mediaResponse = await fetch(mediaApiUrl, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: callId,
        }),
      });

      if (mediaResponse.ok) {
        logger.info('Media session stopped', { call_id: callId, media_port: session.mediaPort });
      } else {
        logger.warn('Error stopping media session (may already be stopped)', {
          call_id: callId,
          status: mediaResponse.status,
        });
      }
    } catch (error) {
      logger.warn('Error calling media service API (may be unavailable)', {
        call_id: callId,
        error: error instanceof Error ? error.message : String(error),
      });
    }

    logger.info('Session cleanup completed', { call_id: callId });
  } catch (error) {
    logger.error('Error during session cleanup', {
      call_id: callId,
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  // Cleanup all active sessions
  const sessionIds = Array.from(activeSessions.keys());
  for (const callId of sessionIds) {
    await cleanupSession(callId);
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  const sessionIds = Array.from(activeSessions.keys());
  for (const callId of sessionIds) {
    await cleanupSession(callId);
  }
  process.exit(0);
});

// Start the orchestrator
main().catch((error) => {
  logger.error('Fatal error', {
    error: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined,
  });
  process.exit(1);
});
