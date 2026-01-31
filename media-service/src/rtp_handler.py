"""
RTP packet handling for PCMU (ulaw) audio.
Handles RTP header parsing, sequence numbers, timestamps, and SSRC.
"""

import struct
from typing import Optional, Tuple


class RTPPacket:
    """Represents an RTP packet with header and payload."""
    
    def __init__(self, data: bytes):
        """Parse RTP packet from raw bytes."""
        if len(data) < 12:
            raise ValueError("RTP packet too short")
        
        # Parse RTP header (first 12 bytes)
        header = struct.unpack('!BBHII', data[:12])
        self.version = (header[0] >> 6) & 0x3
        self.padding = (header[0] >> 5) & 0x1
        self.extension = (header[0] >> 4) & 0x1
        self.cc = header[0] & 0xF
        self.marker = (header[1] >> 7) & 0x1
        self.pt = header[1] & 0x7F
        self.sequence = header[2]
        self.timestamp = header[3]
        self.ssrc = header[4]
        
        # Payload starts after header (12 bytes) + CSRC (4 bytes * cc)
        payload_offset = 12 + (self.cc * 4)
        self.payload = data[payload_offset:]
    
    def to_bytes(self) -> bytes:
        """Serialize RTP packet to bytes."""
        header = struct.pack('!BBHII',
            (self.version << 6) | (self.padding << 5) | (self.extension << 4) | self.cc,
            (self.marker << 7) | self.pt,
            self.sequence,
            self.timestamp,
            self.ssrc
        )
        # For simplicity, assume no CSRC
        return header + self.payload


class RTPHandler:
    """Handles RTP packet encoding/decoding and jitter buffer."""
    
    def __init__(self, ssrc: int, sample_rate: int = 8000, frame_duration_ms: int = 20):
        """
        Initialize RTP handler.
        
        Args:
            ssrc: Synchronization source identifier
            sample_rate: Audio sample rate (Hz)
            frame_duration_ms: Frame duration in milliseconds
        """
        self.ssrc = ssrc
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = (sample_rate * frame_duration_ms) // 1000
        self.timestamp_increment = self.samples_per_frame
        
        # Sequence number and timestamp tracking
        self.sequence_number = 0
        self.timestamp = 0
        
        # Jitter buffer: simple ordered buffer
        self.jitter_buffer: dict[int, bytes] = {}
        self.expected_sequence = None
        self.max_jitter_buffer_size = 10  # Max 10 packets in buffer
    
    def parse_packet(self, data: bytes) -> Optional[Tuple[int, int, bytes]]:
        """
        Parse incoming RTP packet.
        
        Returns:
            Tuple of (sequence_number, payload_type, payload) or None if packet should be dropped
        """
        try:
            packet = RTPPacket(data)
            
            # Initialize expected sequence on first packet
            if self.expected_sequence is None:
                self.expected_sequence = packet.sequence
            
            # Simple jitter buffer: store out-of-order packets
            seq_diff = (packet.sequence - self.expected_sequence) % 65536
            
            if seq_diff == 0:
                # In order, return immediately
                self.expected_sequence = (packet.sequence + 1) % 65536
                # Check if we can now return buffered packets
                while self.expected_sequence in self.jitter_buffer:
                    blob = self.jitter_buffer.pop(self.expected_sequence)
                    self.expected_sequence = (self.expected_sequence + 1) % 65536
                    if not blob:
                        continue
                    pt = blob[0] & 0x7F
                    payload = blob[1:]
                    # Return buffered packet first, then current (simple behavior).
                    return (packet.sequence, pt, payload)
                return (packet.sequence, packet.pt, packet.payload)
            elif seq_diff < self.max_jitter_buffer_size:
                # Out of order, buffer it
                # Store payload with PT so we can decode correctly later.
                # We pack as bytes: pt (1 byte) + payload.
                self.jitter_buffer[packet.sequence] = bytes([packet.pt & 0x7F]) + packet.payload
                # Clean up old entries
                if len(self.jitter_buffer) > self.max_jitter_buffer_size:
                    oldest = min(self.jitter_buffer.keys())
                    del self.jitter_buffer[oldest]
                return None
            else:
                # Too far out of order, drop it
                return None
                
        except Exception as e:
            # Invalid packet, drop it
            return None
    
    def create_packet(self, payload: bytes, marker: int = 0) -> bytes:
        """
        Create RTP packet from payload.
        
        Args:
            payload: Audio payload (ulaw encoded)
            marker: RTP marker bit (1 for end of talk spurt, 0 otherwise)
        """
        # Build RTP header directly (don't instantiate RTPPacket with empty bytes).
        # RFC 3550 fixed header is 12 bytes: V/P/X/CC, M/PT, sequence, timestamp, SSRC.
        version = 2
        padding = 0
        extension = 0
        cc = 0
        pt = 0  # PCMU payload type

        b0 = (version << 6) | (padding << 5) | (extension << 4) | cc
        b1 = ((marker & 0x1) << 7) | (pt & 0x7F)
        header = struct.pack("!BBHII", b0, b1, self.sequence_number, self.timestamp, self.ssrc)

        # Update sequence and timestamp for next packet
        self.sequence_number = (self.sequence_number + 1) % 65536
        self.timestamp = (self.timestamp + self.timestamp_increment) % (2**32)

        return header + payload
