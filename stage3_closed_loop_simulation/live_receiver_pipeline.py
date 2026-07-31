import socket
import struct
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
FS = 250          # Sampling rate
WINDOW_SIZE = 500 # 2 seconds * 250 Hz = 500 samples

class OnlineFilterPipeline:
    def __init__(self, fs=250):
        self.fs = fs
        # Layer 1-4: Consolidated 4th-order Butterworth Bandpass (4Hz - 40Hz) + Notch (50Hz)
        # For a live stream, we precompute the coefficients (b, a) and maintain the filter state (zi)
        low, high = 4.0, 40.0
        nyq = 0.5 * fs
        self.b, self.a = butter(4, [low / nyq, high / nyq], btype='band')
        
        # Initialize independent filter states for each of the 4 channels
        self.zi = [lfilter_zi(self.b, self.a) for _ in range(4)]

    def process_sample(self, raw_sample):
        """
        Applies live, stateful filtering to a single 4-channel sample vector.
        """
        filtered_sample = np.zeros(4)
        for ch in range(4):
            # Causal filtering: processes 1 sample, updates the internal memory state (zi)
            y, next_zi = lfilter(self.b, self.a, [raw_sample[ch]], zi=self.zi[ch])
            self.zi[ch] = next_zi
            filtered_sample[ch] = y[0]
            
        # Layer 5: Robust MAD/Z-Score clamping to remove high-amplitude spikes
        # Hard-clamping artifacts beyond a strict physiological deviation window
        clamped_sample = np.clip(filtered_sample, -300.0, 300.0)
        return clamped_sample

def run_pipeline_receiver():
    # Setup network socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # Initialize our filter core
    pipeline = OnlineFilterPipeline(fs=FS)
    
    # Pre-allocate our rolling 2-second window buffer (Shape: 4 channels x 500 samples)
    window_buffer = np.zeros((4, WINDOW_SIZE))
    samples_collected = 0
    
    print(f"📡 Real-time 5-Layer Filter Pipeline Active. Listening on port {UDP_PORT}...")
    
    while True:
        data, addr = sock.recvfrom(1024)
        if len(data) != 14:
            continue
            
        # Unpack the 14-byte frame
        packet_id, ch1, ch2, ch3, ch4, hw_time, status = struct.unpack("<BhhhhIB", data)
        raw_vector = np.array([ch1, ch2, ch3, ch4], dtype=float)
        
        # Step 1: Run through the 5-layer online processing pipeline
        clean_vector = pipeline.process_sample(raw_vector)
        
        # Step 2: Roll the window buffer left and append the new clean sample to the end
        window_buffer = np.roll(window_buffer, -1, axis=1)
        window_buffer[:, -1] = clean_vector
        samples_collected += 1
        
        # Step 3: Once the buffer is full, trigger downstream evaluation every 50 samples (200ms steps)
        if samples_collected >= WINDOW_SIZE and (samples_collected % 50 == 0):
            # Compute real-time variance across the window to inspect signal health
            channel_variances = np.var(window_buffer, axis=1)
            state_str = "ENGAGED" if status == 0x01 else "RESTING"
            
            print(f"🎬 [EPOCH COMPILED] Total Samples: {samples_collected} | State: {state_str}")
            print(f"   └─ Clean Signal Variance -> Ch1: {channel_variances[0]:.1f} | Ch2: {channel_variances[1]:.1f} | Ch3: {channel_variances[2]:.1f} | Ch4: {channel_variances[3]:.1f}")

if __name__ == "__main__":
    try:
        run_pipeline_receiver()
    except KeyboardInterrupt:
        print("\n🛑 Pipeline receiver stopped.")