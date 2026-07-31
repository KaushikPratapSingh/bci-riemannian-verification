import pylsl
import numpy as np
import time

def run_lsl_pipeline(stream_name='EEG_Device'):
    print(f"Searching for stream: {stream_name}...")
    
    # 1. Resolve the stream
    streams = pylsl.resolve_stream('name', stream_name)
    
    # 2. Create an inlet
    inlet = pylsl.StreamInlet(streams[0])
    print("Stream connected. Waiting for data...")
    
    # Configuration
    sampling_rate = 250  # Matches your EEG device
    window_size = 125    # 0.5 seconds of data
    data_buffer = np.empty((0, 8)) # Assuming 8 EEG channels
    
    try:
        while True:
            # 3. Pull chunks of data
            chunk, timestamps = inlet.pull_chunk(max_samples=100)
            
            if chunk:
                data_buffer = np.vstack((data_buffer, chunk))
                
                # 4. If buffer is full enough, process it
                if len(data_buffer) >= window_size:
                    # Extract the window
                    current_window = data_buffer[:window_size, :]
                    
                    # --- YOUR MODEL INFERENCE GOES HERE ---
                    # prediction = model.predict(current_window)
                    # print(f"Prediction: {prediction}")
                    
                    # Slide the window (keep last 50% for overlap)
                    overlap = 50
                    data_buffer = data_buffer[window_size - overlap:, :]
            
            time.sleep(0.001) # Yield CPU
            
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")

if __name__ == "__main__":
    # Ensure you have liblsl installed via `pip install pylsl`
    run_lsl_pipeline()