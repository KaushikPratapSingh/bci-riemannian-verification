import socket
import struct
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
FS = 250
WINDOW_SIZE = 500

class LiveRiemannianEngine:
    def __init__(self, calibration_mean, ridge_coefs, ridge_intercept, scale_mean=0.0, scale_std=1.0):
        """
        Manages real-time Riemann manifold projection and classification.
        """
        self.M = calibration_mean
        self.coefs = ridge_coefs
        self.intercept = ridge_intercept
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        
        # Precompute matrix roots to maximize execution speeds in streaming threads
        evals, evecs = np.linalg.eigh(self.M)
        evals = np.clip(evals, 1e-10, None)
        self.M_inv_sqr = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        self.M_sqr = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

    def _matrix_log(self, O):
        evals, evecs = np.linalg.eigh(O)
        evals = np.clip(evals, 1e-10, None)
        return evecs @ np.diag(np.log(evals)) @ evecs.T

    def _upper_tri_vectorize(self, S):
        n = S.shape[0]
        idx = np.triu_indices(n)
        diag_mask = (idx[0] == idx[1])
        vec = S[idx].copy()
        vec[~diag_mask] *= np.sqrt(2.0)
        return vec

    def compute_continuous_engagement(self, C_live):
        # 1. Project onto the localized Tangent Space
        unaligned_tangent = self.M_inv_sqr @ C_live @ self.M_inv_sqr
        aligned_tangent = self.M_sqr @ self._matrix_log(unaligned_tangent) @ self.M_sqr
        
        # 2. Vectorize the upper triangle 
        s_live = self._upper_tri_vectorize(aligned_tangent)
        
        # 3. Apply global standardization scaling
        x_live = (s_live - self.scale_mean) / self.scale_std
        
        # 4. Evaluate Ridge Regressor function
        raw_score = np.dot(x_live, self.coefs) + self.intercept
        return np.clip(raw_score, 0.0, 100.0)

class OnlineFilterPipeline:
    def __init__(self, fs=250):
        low, high = 4.0, 40.0
        nyq = 0.5 * fs
        self.b, self.a = butter(4, [low / nyq, high / nyq], btype='band')
        self.zi = [lfilter_zi(self.b, self.a) for _ in range(4)]

    def process_sample(self, raw_sample):
        filtered_sample = np.zeros(4)
        for ch in range(4):
            y, next_zi = lfilter(self.b, self.a, [raw_sample[ch]], zi=self.zi[ch])
            self.zi[ch] = next_zi
            filtered_sample[ch] = y[0]
        return np.clip(filtered_sample, -300.0, 300.0)

def run_scoring_engine():
    # --- INTERFACE SETUP AND MOCK WEIGHTS ---
    # In live usage, these matrices are generated directly from your calibration run.
    mock_identity_baseline = np.eye(4) * 5000.0
    
    # 10 features corresponding to the unique elements of a symmetric 4x4 matrix
    mock_ridge_coefs = np.array([-1.5, -0.5, 2.0, 2.5, -0.2, 0.4, 1.8, -0.1, 0.5, 2.2])
    mock_intercept = 45.0
    
    scoring_engine = LiveRiemannianEngine(
        calibration_mean=mock_identity_baseline,
        ridge_coefs=mock_ridge_coefs,
        ridge_intercept=mock_intercept,
        scale_mean=0.0,
        scale_std=1.0
    )
    
    pipeline = OnlineFilterPipeline(fs=FS)
    window_buffer = np.zeros((4, WINDOW_SIZE))
    samples_collected = 0
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    print(f"🧠 Live Riemannian Scoring Engine Active. Processing matrix geometry...")
    
    while True:
        data, addr = sock.recvfrom(1024)
        if len(data) != 14:
            continue
            
        packet_id, ch1, ch2, ch3, ch4, hw_time, status = struct.unpack("<BhhhhIB", data)
        raw_vector = np.array([ch1, ch2, ch3, ch4], dtype=float)
        
        # Filter and stack
        clean_vector = pipeline.process_sample(raw_vector)
        window_buffer = np.roll(window_buffer, -1, axis=1)
        window_buffer[:, -1] = clean_vector
        samples_collected += 1
        
        # Trigger classification window loop every 50 samples (200ms)
        if samples_collected >= WINDOW_SIZE and (samples_collected % 50 == 0):
            # Compute real-time empirical covariance matrix (4x4)
            # Rowvar=True means rows are our 4 distinct channels
            C_live = np.cov(window_buffer, rowvar=True)
            
            # Prevent singular matrix estimation errors via shrinkage regularization
            C_live += np.eye(4) * 1e-6
            
            # Compute engagement score
            engagement_score = scoring_engine.compute_continuous_engagement(C_live)
            
            state_str = "ENGAGED" if status == 0x01 else "RESTING"
            
            # Render visual indicator bar
            bar_length = int(engagement_score / 4)
            visual_bar = "█" * bar_length + "░" * (25 - bar_length)
            
            print(f"🔮 [{state_str}] Engagement: {engagement_score:05.1f}/100.0 | [{visual_bar}]")

if __name__ == "__main__":
    try:
        run_scoring_engine()
    except KeyboardInterrupt:
        print("\n🛑 Scoring engine stopped.")