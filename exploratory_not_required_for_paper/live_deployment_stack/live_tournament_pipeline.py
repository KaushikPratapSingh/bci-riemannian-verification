import socket
import json
import numpy as np
import time
from scipy.signal import butter, lfilter

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
FS = 250          # Sampling rate from virtual brain
WINDOW_SIZE = 500 # 2-second moving window for covariance calculation
STEP_SIZE = 25    # Update metrics every 100ms
CHANNELS = 4

# --- 5-LAYER BUTTERWORTH FILTER SETUP ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=4.0, highcut=40.0, fs=250, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data, axis=0)

# --- RIEMANNIAN COVARIANCE MATH ---
def compute_covariance(window_data):
    # Center the data
    centered = window_data - np.mean(window_data, axis=0)
    # Compute regularized sample covariance matrix
    cov = np.dot(centered.T, centered) / (window_data.shape[0] - 1)
    cov += np.eye(CHANNELS) * 1e-6 # Regularization for matrix stability
    return cov

def matrix_logarithm(matrix):
    vals, vecs = np.linalg.eigh(matrix)
    # Ensure eigenvalues are strictly positive for log calculation
    vals = np.maximum(vals, 1e-9)
    return np.dot(vecs, np.dot(np.diag(np.log(vals)), vecs.T))

def tangent_space_vector(cov, reference_matrix=None):
    # Project covariance matrix to Tangent Space (Riemannian optimization)
    if reference_matrix is None:
        log_cov = matrix_logarithm(cov)
    else:
        # Vectorize using basic matrix log projection
        log_cov = matrix_logarithm(cov)
    
    # Extract upper triangular features to eliminate redundancy
    idx = np.triu_indices(CHANNELS)
    return log_cov[idx]

# --- RIDGE REGRESSION FROM SCRATCH ---
class MiniRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        
    def fit(self, X, y):
        # Normal Equation: w = (X^T * X + alpha * I)^-1 * X^T * y
        X_T = X.T
        num_features = X.shape[1]
        self.weights = np.linalg.inv(np.dot(X_T, X) + self.alpha * np.eye(num_features)).dot(X_T).dot(y)
        
    def predict(self, X):
        # Clip outputs bounded between 0 and 100
        scores = np.dot(X, self.weights)
        return np.clip(scores, 0.0, 100.0)

# --- LIVE ENGINE EXECUTION ---
def run_pipeline():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    raw_buffer = []
    
    # ----------------------------------------------------
    # PHASE 1: AUTOMATIC REAL-WORLD CALIBRATION
    # ----------------------------------------------------
    print("\n🛰️  [PHASE 1: CALIBRATION] Listening for Virtual Brain...")
    calibration_features = []
    calibration_labels = []
    
    start_time = time.time()
    calibration_duration = 30 # Collect data for 30 seconds
    
    print(f"🧬 Collecting baseline neural signatures. Keep simulator running for {calibration_duration}s...")
    
    while time.time() - start_time < calibration_duration:
        data, addr = sock.recvfrom(4096)
        packet = json.loads(data.decode('utf-8'))
        raw_buffer.append(packet["eeg"])
        
        # When we have enough data to form a window, extract a feature vector
        if len(raw_buffer) >= WINDOW_SIZE:
            recent_data = np.array(raw_buffer[-WINDOW_SIZE:])
            filtered_data = bandpass_filter(recent_data)
            cov = compute_covariance(filtered_data)
            feature_vector = tangent_space_vector(cov)
            
            calibration_features.append(feature_vector)
            # Map ground truth text states to targets (ENGAGED = 100.0, RESTING = 0.0)
            target = 100.0 if packet["state_ground_truth"] == "ENGAGED" else 0.0
            calibration_labels.append(target)
            
            # Slide window slightly
            raw_buffer = raw_buffer[-WINDOW_SIZE+STEP_SIZE:]

    print(f"✅ Collected {len(calibration_features)} calibration windows.")
    print("🤖 Training Tournament Ridge Model on Tangent Space features...")
    
    X_train = np.array(calibration_features)
    y_train = np.array(calibration_labels)
    
    model = MiniRidgeRegression(alpha=10.0)
    model.fit(X_train, y_train)
    print("🏆 Tournament Model Trained successfully!")
    
    # ----------------------------------------------------
    # PHASE 2: CLOSED-LOOP LIVE SCORING ENGINE
    # ----------------------------------------------------
    print("\n🧠 [PHASE 2: LIVE ENGINE ACTIVE] Processing incoming matrix geometry...\n")
    raw_buffer = [] # Clear out buffer for clean streaming
    
    while True:
        data, addr = sock.recvfrom(4096)
        packet = json.loads(data.decode('utf-8'))
        raw_buffer.append(packet["eeg"])
        
        if len(raw_buffer) >= WINDOW_SIZE:
            recent_data = np.array(raw_buffer[-WINDOW_SIZE:])
            filtered_data = bandpass_filter(recent_data)
            cov = compute_covariance(filtered_data)
            feature_vector = tangent_space_vector(cov).reshape(1, -1)
            
            # Predict attention metrics using our freshly calibrated model
            engagement_score = model.predict(feature_vector)[0]
            
            # Generate visual bar indicators
            bar_length = int(engagement_score / 4)
            visual_bar = "█" * bar_length + "░" * (25 - bar_length)
            
            # Determine current operational label output
            current_label = packet["state_ground_truth"]
            status_emoji = "🔮" if current_label == "ENGAGED" else "💤"
            
            print(f"{status_emoji} [{current_label}] Attention Index: {engagement_score:05.1f}/100.0 | [{visual_bar}]", end="\r")
            
            # Flush stepping window data
            raw_buffer = raw_buffer[-WINDOW_SIZE+STEP_SIZE:]

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\nPipeline terminated safely by operator.")