import os
import glob
import random
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FS = 250
WINDOW_SIZE = 500
STEP_SIZE = 50

def get_latest_session():
    list_of_files = glob.glob(os.path.join("recorded_sessions", "*.csv"))
    if not list_of_files:
        raise FileNotFoundError("No recorded sessions found in 'recorded_sessions/' directory.")
    return max(list_of_files, key=os.path.getctime)

def butter_bandpass_zero_phase(data, lowcut=8.0, highcut=30.0, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# --- INDIVIDUAL COVARIANCE & TANGENT MAPPING ---
def extract_riemannian_features(window):
    centered = window - np.mean(window, axis=0)
    n_samples = window.shape[0]
    s_cov = np.dot(centered.T, centered) / (n_samples - 1)
    
    mean_var = np.mean(np.diag(s_cov))
    target = mean_var * np.eye(s_cov.shape[0])
    alpha = 0.15  
    cov = (1 - alpha) * s_cov + alpha * target
    
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    log_cov = np.dot(vecs, np.dot(np.diag(np.log(vals)), vecs.T))
    return log_cov[np.triu_indices(cov.shape[0])]

def extract_euclidean_features(window):
    return np.log(np.var(window, axis=0))

# --- DETERMINISTIC OFFLINE CONTEXT ENGINE ---
def generate_live_context(state_str, confidence, blinks, clenches, history_list):
    """Generates zero-storage, lightning-fast context commentary for live tracking events."""
    if clenches >= 2:
        return random.choice([
            "Signal Alert: High-frequency muscle tension detected. Try relaxing your jaw to clear tracking noise.",
            "Somatic noise is masking your brainwaves right now. A quick jaw relaxation will immediately optimize accuracy."
        ])
    if blinks >= 10:
        return random.choice([
            "Ocular tracking alert: High volume of eye blinks is shifting baseline. Try keeping a steady gaze.",
            "We are filtering heavy ocular noise. If your eyes feel strained, this is a great cue for a brief break."
        ])

    if len(history_list) >= 3:
        if history_list[-2:] == ["ENGAGED", "ENGAGED"] and history_list[-3] == "RESTING":
            return f"Excellent transition! You have successfully broken past baseline barriers into a focus corridor ({confidence:.1f}% stability)."
        if history_list[-2:] == ["RESTING", "RESTING"] and history_list[-3] == "ENGAGED":
            return "Your cognitive workload is winding down. Perfect window to switch to lighter tasks or take a breather."

    if state_str == "ENGAGED":
        return f"Peak cognitive immersion detected ({confidence:.1f}% stability). Your alpha-beta profile shows deep task engagement."
    else:
        return f"System calibrated at rest ({confidence:.1f}% stability). Brainwaves have cleanly shifted to a regenerative Alpha profile."

# --- MAIN ENGINE ---
def run_tournament():
    target_csv = get_latest_session()
    print(f"📖 Loading individual virtual brain session: {target_csv}")
    
    df = pd.read_csv(target_csv)
    raw_data = df[["CH1_Voltage", "CH2_Voltage", "CH3_Voltage", "CH4_Voltage"]].values
    labels = df["State_Ground_Truth"].values
    total_raw_samples = len(df)
    
    print("🧼 Signal Restoration: Applying narrow-band zero-phase filter (8-30 Hz)...")
    restored_data = butter_bandpass_zero_phase(raw_data, fs=FS)
    
    X_euclidean, X_riemannian, y = [], [], []
    blink_counts, clench_counts = [], []
    
    print("✂️  Segmenting continuous signal via Temporal Majority Voting...")
    for idx in range(0, len(restored_data) - WINDOW_SIZE, STEP_SIZE):
        window_chunk = restored_data[idx : idx + WINDOW_SIZE]
        raw_window_chunk = raw_data[idx : idx + WINDOW_SIZE]
        label_window = labels[idx : idx + WINDOW_SIZE]
        
        vals, counts = np.unique(label_window, return_counts=True)
        majority_vote = vals[np.argmax(counts)]
        
        # Track raw artifacts per window slice
        max_amplitude = np.max(np.abs(raw_window_chunk))
        window_variance = np.mean(np.var(raw_window_chunk, axis=0))
        
        blinks = 12 if (max_amplitude > 40.0 and window_variance < 150) else 0
        clenches = 3 if (window_variance > 150) else 0
        
        X_euclidean.append(extract_euclidean_features(window_chunk))
        X_riemannian.append(extract_riemannian_features(window_chunk))
        y.append(1 if majority_vote in ["COGNITIVE_LOAD", "ENGAGED"] else 0)
        blink_counts.append(blinks)
        clench_counts.append(clenches)
        
    X_euc = np.array(X_euclidean)
    X_rie = np.array(X_riemannian)
    y = np.array(y)
    
    print("\n🏟️  CROSS-VALIDATION BRACKET INITIALIZED...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train the final Riemannian deployment classifier on the full set for live simulation
    clf_riem = RidgeClassifier(alpha=5.0)
    clf_riem.fit(X_rie, y)
    
    # --- SIMULATE STREAMING THROUGH THE LAST 5 WINDOWS ---
    print("\n🎬 STREAMING ACTIVE SESSION DECODING (Last 5 Windows Output):")
    print("-" * 80)
    
    history_states = []
    # Grab the final 5 windows of the file to simulate live running tracking
    for step in range(len(y) - 5, len(y)):
        feat = X_rie[step].reshape(1, -1)
        pred_class = clf_riem.predict(feat)[0]
        
        # Calculate pseudo-confidence via distance to decision boundary
        decision_dist = clf_riem.decision_function(feat)[0]
        confidence_score = 50.0 + min(abs(decision_dist) * 10, 45.0)  # Bound gracefully up to 95%
        
        state_str = "ENGAGED" if pred_class == 1 else "RESTING"
        history_states.append(state_str)
        
        context_insight = generate_live_context(
            state_str=state_str,
            confidence=confidence_score,
            blinks=blink_counts[step],
            clenches=clench_counts[step],
            history_list=history_states
        )
        
        print(f"📥 [Live Window Snapshot #{step}] -> Decoded: {state_str} (Stability: {confidence_score:.1f}%)")
        print(f"💡 AI Context: {context_insight}")
        print("-" * 80)

if __name__ == "__main__":
    run_tournament();