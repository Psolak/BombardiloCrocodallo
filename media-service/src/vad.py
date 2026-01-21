"""
Voice Activity Detection (VAD) using energy-based approach.
Simple and effective for MVP.
"""

import numpy as np
from typing import Optional


class EnergyVAD:
    """Energy-based Voice Activity Detection."""
    
    def __init__(self, sample_rate: int = 8000, frame_duration_ms: int = 20, 
                 energy_threshold: float = 0.01, hangover_frames: int = 3):
        """
        Initialize energy-based VAD.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            frame_duration_ms: Frame duration in milliseconds
            energy_threshold: Energy threshold for speech detection (0.0-1.0)
            hangover_frames: Number of frames to keep speech active after energy drops
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.hangover_frames = hangover_frames
        
        self.hangover_counter = 0
        self.is_speech = False
        
        # Adaptive threshold: learn background noise level
        self.background_energy = None
        self.adaptation_rate = 0.1
    
    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process audio frame and detect speech.
        
        Args:
            audio_frame: PCM16 audio samples (numpy array)
        
        Returns:
            True if speech detected, False otherwise
        """
        # Calculate frame energy (RMS)
        if len(audio_frame) == 0:
            return False
        
        # Normalize to [-1, 1] range
        normalized = audio_frame.astype(np.float32) / 32768.0
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(normalized ** 2))
        
        # Adaptive threshold: update background energy
        if self.background_energy is None:
            self.background_energy = energy
        else:
            # Only update if current energy is below threshold (background noise)
            if energy < self.energy_threshold:
                self.background_energy = (
                    (1 - self.adaptation_rate) * self.background_energy +
                    self.adaptation_rate * energy
                )
        
        # Use adaptive threshold (background + fixed threshold)
        threshold = max(self.energy_threshold, self.background_energy * 2.0)
        
        # Detect speech
        if energy > threshold:
            self.is_speech = True
            self.hangover_counter = self.hangover_frames
        else:
            if self.hangover_counter > 0:
                self.hangover_counter -= 1
            else:
                self.is_speech = False
        
        return self.is_speech
    
    def reset(self):
        """Reset VAD state."""
        self.hangover_counter = 0
        self.is_speech = False
        self.background_energy = None
