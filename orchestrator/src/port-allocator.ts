/**
 * Simple port allocator for media service endpoints.
 * Allocates ports sequentially from a base port.
 */
export class PortAllocator {
  private basePort: number;
  private allocatedPorts: Set<number>;
  private nextPort: number;
  private maxPorts: number = 1000; // Allow up to 1000 concurrent calls

  constructor(basePort: number) {
    this.basePort = basePort;
    this.allocatedPorts = new Set();
    this.nextPort = basePort;
  }

  /**
   * Allocate a port for a new call.
   * Returns the allocated port number.
   */
  allocate(): number {
    // Find the next available port
    let attempts = 0;
    while (attempts < this.maxPorts) {
      const port = this.nextPort;
      this.nextPort = (this.nextPort + 1) % (this.basePort + this.maxPorts);
      
      if (!this.allocatedPorts.has(port)) {
        this.allocatedPorts.add(port);
        return port;
      }
      
      attempts++;
    }

    throw new Error('No available ports (all ports allocated)');
  }

  /**
   * Free a port when a call ends.
   */
  free(port: number): void {
    this.allocatedPorts.delete(port);
  }

  /**
   * Get the number of currently allocated ports.
   */
  getActiveCount(): number {
    return this.allocatedPorts.size;
  }
}
