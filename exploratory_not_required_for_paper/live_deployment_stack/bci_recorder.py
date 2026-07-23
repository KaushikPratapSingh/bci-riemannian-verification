import socket
import json
import csv
import os
import time

# --- NETWORK CONFIGURATION ---
# Note: UDP allows multiple listeners if configured, but since our brain engine 
# streams to a single port, this recorder will act as your dedicated data harvester.
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# --- RECORDING CONFIGURATION ---
OUTPUT_DIR = "recorded_sessions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
session_filename = os.path.join(OUTPUT_DIR, f"bci_session_{int(time.time())}.csv")

print("💾 BCI Telemetry Recorder Initialized.")
print(f"📄 Target file: {session_filename}")
print("Connecting to streaming pipeline...")

# Initialize UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Define CSV Headers matching standard BCI dataset repositories
csv_headers = ["Timestamp", "State_Ground_Truth", "CH1_Voltage", "CH2_Voltage", "CH3_Voltage", "CH4_Voltage"]

sample_count = 0

try:
    with open(session_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)  # Write header row
        
        print("\n🔴 RECORDING STARTED. Press [ Ctrl + C ] safely in this window to stop.\n")
        
        while True:
            # Receive telemetry packet
            data, _ = sock.recvfrom(4096)
            packet = json.loads(data.decode('utf-8'))
            
            # Extract metrics
            ts = packet["timestamp"]
            ground_truth = packet["state_ground_truth"]
            eeg_channels = packet["eeg"]  # Array of 4 floats
            
            # Construct row payload
            row = [ts, ground_truth] + eeg_channels
            writer.writerow(row)
            
            sample_count += 1
            if sample_count % 250 == 0:
                print(f"📑 Archived {sample_count} samples (~{sample_count // 250} seconds of continuous EEG)...", end="\r")

except KeyboardInterrupt:
    print(f"\n\n🛑 Recording halted safely by user.")
    print(f"📊 Total samples written to disk: {sample_count}")
    print(f"📂 File saved successfully at: {session_filename}")