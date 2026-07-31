import socket
import json
import numpy as np
import time
import threading
import pygame
import sys
from scipy.signal import butter, lfilter

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
FS = 250
WINDOW_SIZE = 500
STEP_SIZE = 25
CHANNELS = 4

# --- GLOBAL DATA PIPES ---
raw_buffer = []
latest_score = 0.0
latest_state = "CALIBRATING"
filtered_wave_history = np.zeros((200, CHANNELS))  # 200 points for display window
data_lock = threading.Lock()

# --- DSP FILTER SETUP ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

b_band, a_band = butter_bandpass(4.0, 40.0, FS, order=5)

def live_filter(data_chunk):
    return lfilter(b_band, a_band, data_chunk, axis=0)

# --- MATHEMATICAL COVARIANCE & ML ---
def compute_covariance(window_data):
    centered = window_data - np.mean(window_data, axis=0)
    cov = np.dot(centered.T, centered) / (window_data.shape[0] - 1)
    cov += np.eye(CHANNELS) * 1e-6
    return cov

def tangent_space_vector(cov):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    log_cov = np.dot(vecs, np.dot(np.diag(np.log(vals)), vecs.T))
    return log_cov[np.triu_indices(CHANNELS)]

class RidgeRegressionModel:
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.weights = None
    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y
    def predict(self, X):
        return np.clip(X @ self.weights, 0.0, 100.0)

model = RidgeRegressionModel()

# --- NETWORK MULTI-THREADED DATA RECEIVER ---
def udp_network_thread():
    global raw_buffer, latest_score, latest_state, filtered_wave_history
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    local_raw = []
    calibration_features = []
    calibration_labels = []
    
    calibration_end_time = time.time() + 20.0
    is_calibrated = False
    
    print("📡 Network Thread Online. Reading socket pipeline data...")
    
    while True:
        try:
            data, _ = sock.recvfrom(4096)
            packet = json.loads(data.decode('utf-8'))
            eeg_sample = packet["eeg"]
            ground_truth = packet["state_ground_truth"]
            
            local_raw.append(eeg_sample)
            
            # Keep rolling display history updating sample-by-sample
            if len(local_raw) >= 10:
                recent_chunk = np.array(local_raw[-10:])
                filt_chunk = live_filter(recent_chunk)
                with data_lock:
                    filtered_wave_history = np.roll(filtered_wave_history, -1, axis=0)
                    filtered_wave_history[-1, :] = filt_chunk[-1, :]
            
            # Handle full optimization windows
            if len(local_raw) >= WINDOW_SIZE:
                window_np = np.array(local_raw[-WINDOW_SIZE:])
                filtered_np = live_filter(window_np)
                
                cov = compute_covariance(filtered_np)
                feat = tangent_space_vector(cov)
                
                if not is_calibrated:
                    latest_state = f"CALIBRATING ({max(0, int(calibration_end_time - time.time()))}s)"
                    calibration_features.append(feat)
                    target = 100.0 if ground_truth == "ENGAGED" else 0.0
                    calibration_labels.append(target)
                    
                    if time.time() >= calibration_end_time:
                        X = np.array(calibration_features)
                        y = np.array(calibration_labels)
                        model.fit(X, y)
                        is_calibrated = True
                        print("🏆 Riemannian Matrix Model trained and deployed.")
                else:
                    pred = model.predict(feat.reshape(1, -1))[0]
                    with data_lock:
                        latest_score = pred
                        latest_state = ground_truth
                
                local_raw = local_raw[-WINDOW_SIZE+STEP_SIZE:]
                
        except Exception as e:
            print(f"Extraction Error: {e}")

# --- PYGAME ENGINE RUNTIME ---
def run_dashboard():
    pygame.init()
    screen = pygame.display.set_mode((900, 600))
    pygame.display.set_caption("⚡ Live Riemannian BCI Application Suite")
    clock = pygame.time.Clock()
    
    font_large = pygame.font.SysFont("Courier", 26, bold=True)
    font_med = pygame.font.SysFont("Courier", 16, bold=True)
    font_small = pygame.font.SysFont("Courier", 12)
    
    threading.Thread(target=udp_network_thread, daemon=True).start()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        screen.fill((15, 18, 26))
        
        # Safe thread-safe state variable capture
        with data_lock:
            score = latest_score
            state = latest_state
            waves = np.copy(filtered_wave_history)
            
        # PANEL 1: STATUS
        pygame.draw.rect(screen, (30, 36, 50), (30, 20, 840, 70), border_radius=8)
        status_color = (0, 210, 255) if "CALIBRATING" in state else ((50, 220, 120) if state == "ENGAGED" else (240, 80, 80))
        lbl_status = font_large.render(f"SYSTEM STATUS: {state}", True, status_color)
        screen.blit(lbl_status, (50, 38))
        
        # PANEL 2: GAUGE
        pygame.draw.rect(screen, (24, 29, 41), (30, 110, 400, 450), border_radius=8)
        screen.blit(font_med.render("🔮 DECODED ATTENTION GAUGE", True, (200, 210, 230)), (50, 130))
        screen.blit(font_large.render(f"{score:05.1f} / 100", True, (255, 255, 255)), (140, 200))
        
        pygame.draw.rect(screen, (40, 50, 70), (80, 280, 300, 35), border_radius=5)
        bar_width = int((score / 100.0) * 300)
        if bar_width > 0:
            fill_color = (int(score * 2.55), int(255 - (score * 2.55)), 100)
            pygame.draw.rect(screen, tuple(np.clip(fill_color, 0, 255)), (80, 280, bar_width, 35), border_radius=5)
            
        if score > 80.0:
            screen.blit(font_med.render("🔥 ABOVE CRITICAL THRESHOLD", True, (255, 215, 0)), (70, 350))
            
        # PANEL 3: FREQUENCY OSCILLOSCOPE
        pygame.draw.rect(screen, (24, 29, 41), (460, 110, 410, 450), border_radius=8)
        screen.blit(font_med.render("📡 FREQUENCY SCOPE (4-40Hz)", True, (200, 210, 230)), (480, 130))
        
        for ch in range(CHANNELS):
            ch_y_center = 200 + (ch * 90)
            pygame.draw.line(screen, (40, 50, 70), (480, ch_y_center), (850, ch_y_center), 1)
            screen.blit(font_small.render(f"CH {ch+1}", True, (100, 120, 150)), (480, ch_y_center - 35))
            
            points = []
            for x_idx in range(len(waves)):
                x_pos = 480 + (x_idx * 1.85)
                y_pos = ch_y_center - int(waves[x_idx, ch] * 8)
                points.append((x_pos, y_pos))
                
            if len(points) > 1:
                pygame.draw.lines(screen, (0, 180, 255 - ch*40), False, points, 2)

        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_dashboard()