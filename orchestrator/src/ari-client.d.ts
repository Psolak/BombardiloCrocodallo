declare module 'ari-client' {
  export interface Channel {
    id: string;
    answer(): Promise<void>;
    hangup(): Promise<void>;
    on(event: string, callback: (...args: any[]) => void): void;
  }

  export interface Bridge {
    id: string;
    addChannel(params: { channel: string }): Promise<void>;
    removeChannel(params: { channel: string }): Promise<void>;
    destroy(): Promise<void>;
  }

  export interface Client {
    bridges: {
      create(params: { type: string }): Promise<Bridge>;
      get(params: { bridgeId: string }): Promise<any>;
    };
    channels: {
      externalMedia(params: {
        app: string;
        external_host: string;
        format: string;
        direction?: string;
        transport?: string;
        encapsulation?: string;
      }): Promise<Channel>;
      get(params: { channelId: string }): Promise<any>;
      getChannelVar(params: { channelId: string; variable: string }): Promise<{ value: string }>;
    };
    start(app: string): Promise<void>;
    on(event: string, callback: (...args: any[]) => void): void;
  }

  export interface StasisStartEvent {
    channel: {
      id: string;
      caller?: {
        number?: string;
      };
    };
  }

  export function connect(
    url: string,
    username: string,
    password: string
  ): Promise<Client>;
}
