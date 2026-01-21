import * as ari from 'ari-client';

export interface CallSessionData {
  callId: string;
  asteriskChannelId: string;
  bridgeId: string;
  externalChannelId: string;
  mediaPort: number;
  channel: ari.Channel;
  bridge: ari.Bridge;
  externalChannel: ari.Channel;
  client: ari.Client;
}

export class CallSession {
  public readonly callId: string;
  public readonly asteriskChannelId: string;
  public readonly bridgeId: string;
  public readonly externalChannelId: string;
  public readonly mediaPort: number;
  public readonly channel: ari.Channel;
  public readonly bridge: ari.Bridge;
  public readonly externalChannel: ari.Channel;
  public readonly client: ari.Client;

  constructor(data: CallSessionData) {
    this.callId = data.callId;
    this.asteriskChannelId = data.asteriskChannelId;
    this.bridgeId = data.bridgeId;
    this.externalChannelId = data.externalChannelId;
    this.mediaPort = data.mediaPort;
    this.channel = data.channel;
    this.bridge = data.bridge;
    this.externalChannel = data.externalChannel;
    this.client = data.client;
  }
}
