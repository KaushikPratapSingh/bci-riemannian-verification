"""
BCI Telemetry Recorder v2
==========================
Improvements over v1 (Gemini):
  1. SEQUENCE GAP DETECTION: reads the 'seq' field and logs any dropped
     UDP packets so you know if your system is falling behind.
  2. ARTIFACT STATE: records raw artifact_state (BASELINE_REST,
     COGNITIVE_LOAD, DISTRACTION_BLINK, DISTRACTION_MUSCLE) as a separate
     column alongside the binary ground truth label -- useful for analysis.
  3. PACKET LOSS SUMMARY printed at the end.

Run this in Terminal 2 while virtual_brain_v2.py is running in Terminal 1.
"""

import socket
import json
import csv
import os
import time

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

OUTPUT_DIR      = "recorded_sessions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
session_ts      = int(time.time())
session_file    = os.path.join(OUTPUT_DIR, f"bci_session_{session_ts}.csv")

print("💾 BCI Recorder v2 initialized.")
print(f"📄 Output: {session_file}")
print("Waiting for virtual brain stream...")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

CSV_HEADERS = ["seq", "timestamp", "state_ground_truth", "artifact_state",
               "CH1_uV", "CH2_uV", "CH3_uV", "CH4_uV"]

sample_count  = 0
dropped_count = 0
expected_seq  = None

try:
    with open(session_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

        print("\n🔴 RECORDING. Press Ctrl+C to stop.\n")
        while True:
            data, _ = sock.recvfrom(4096)
            pkt = json.loads(data.decode())

            seq            = pkt.get("seq", 0)
            ts             = pkt["timestamp"]
            ground_truth   = pkt["state_ground_truth"]
            artifact_state = pkt.get("artifact_state", ground_truth)
            eeg            = pkt["eeg"]

            # Sequence gap detection
            if expected_seq is not None and seq != expected_seq:
                gap = seq - expected_seq
                dropped_count += gap
                print(f"  ⚠️  Dropped {gap} packets (seq {expected_seq}→{seq})")
            expected_seq = seq + 1

            writer.writerow([seq, ts, ground_truth, artifact_state] + eeg)
            sample_count += 1

            if sample_count % 250 == 0:
                duration_s = sample_count / 250
                print(f"  📑 {sample_count:,} samples  ({duration_s:.0f}s)  "
                      f"dropped={dropped_count}", end="\r")

except KeyboardInterrupt:
    print(f"\n\n🛑 Recording stopped.")
    print(f"  Samples written : {sample_count:,}")
    print(f"  Packets dropped : {dropped_count}")
    print(f"  Packet loss     : {dropped_count/(sample_count+dropped_count)*100:.2f}%")
    print(f"  Duration        : {sample_count/250:.1f}s")
    print(f"  File            : {session_file}")