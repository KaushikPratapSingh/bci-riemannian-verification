import socket
import struct
import time
import numpy as np

# Networking Config (Local loopback)
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

def run_simulator():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    sampling_rate = 250
    interval = 1.0 / sampling_rate  # Exactly 0.004 seconds (4000 microseconds)
    packet_id = 0
    
    print(f"🚀 Virtual ESP32-S3 Active. Broadcasting 14-byte stream to {UDP_IP}:{UDP_PORT}...")
    print("Press Ctrl+C to terminate.")
    
    start_time = time.time()
    next_sample_time = time.time()
    
    while True:
        current_time = time.time()
        
        # Enforce strict 250Hz hardware-timed loop
        if current_time >= next_sample_time:
            next_sample_time += interval
            
            elapsed = current_time - start_time
            
            # --- SYNTHESIZE EEG SIGNAL MIX ---
            # Simulate shifting states: Rest for first 15s, Engagement after 15s
            is_engaged = (elapsed % 30) > 15 
            
            # Base 10Hz Alpha wave (stronger during rest, suppressed during engagement)
            alpha_amplitude = 150.0 if not is_engaged else 30.0
            alpha = alpha_amplitude * np.sin(2 * np.pi * 10.0 * elapsed)
            
            # Base 20Hz Beta wave (stronger during active engagement)
            beta_amplitude = 40.0 if not is_engaged else 120.0
            beta = beta_amplitude * np.sin(2 * np.pi * 20.0 * elapsed)
            
            # Add random ambient dry-electrode noise
            noise = np.random.normal(0, 15.0, 4)
            
            # Center the ADC readings around a baseline virtual offset (e.g., 2048)
            ch1 = int(2048 + alpha + beta * 0.8 + noise[0])
            ch2 = int(2048 + alpha * 0.9 + beta * 1.1 + noise[1])
            ch3 = int(2048 + alpha * 0.3 + beta * 1.5 + noise[2])
            ch4 = int(2048 + alpha * 0.2 + beta * 1.4 + noise[3])
            
            # Microsecond hardware timestamp simulation
            hw_timestamp_us = int((time.time() - start_time) * 1_000_000) & 0xFFFFFFFF
            status_footer = 0x01 if is_engaged else 0x00  # State flag
            
            # --- PACK INTO THE EXACT 14-BYTE SPEC ---
            # < = Little Endian, B = uint8, h = int16, I = uint32
            payload = struct.pack("<BhhhhIB", 
                                  packet_id, 
                                  ch1, ch2, ch3, ch4, 
                                  hw_timestamp_us, 
                                  status_footer)
            
            # Broadcast the packet
            sock.sendto(payload, (UDP_IP, UDP_PORT))
            
            # Increment and wrap packet ID at 255 (uint8 limit)
            packet_id = (packet_id + 1) % 256
            
        # Give the CPU breathing room, exactly like a low-power microcontroller sleep mode
        time.sleep(0.0005)

if __name__ == "__main__":
    try:
        run_simulator()
    except KeyboardInterrupt:
        print("\n🛑 Simulator stopped.")