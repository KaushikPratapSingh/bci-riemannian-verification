import numpy as np
from scipy.linalg import logm

class BCIRealTimeEngineEWMA:
    def __init__(self, num_channels=18, alpha=0.01):
        self.N = num_channels
        self.alpha = alpha
        self.cov_matrix = np.eye(self.N) * 0.1  # Start with smaller values to avoid overflow
        self.mean_vector = np.zeros(self.N)
        self.initialized = False

    def update_step(self, sample):
        sample = np.array(sample)
        if not self.initialized:
            self.mean_vector = sample
            self.initialized = True
            return self.cov_matrix

        diff = sample - self.mean_vector
        self.mean_vector = (1 - self.alpha) * self.mean_vector + self.alpha * sample
        
        # Rank-1 update
        outer = np.outer(diff, diff)
        self.cov_matrix = (1 - self.alpha) * self.cov_matrix + self.alpha * outer
        
        # Stability: Ensure matrix stays positive definite
        self.cov_matrix += np.eye(self.N) * 1e-6
        return self.cov_matrix

    def get_riemannian_distance(self, M_rest):
        """
        Calculates a stable distance using the Procrustes/Log-Euclidean approximation.
        This avoids the 'inf' issue by using the Frobenius norm of the matrix difference.
        """
        # Ensure M_rest is stable
        M_rest = M_rest + np.eye(self.N) * 1e-6
        
        # A more stable, faster distance for real-time: 
        # d = ||log(P) - log(Q)||_F
        # Using scipy.linalg.logm is standard, but if you need pure numpy speed:
        # We use the Log-Determinant Divergence as a high-speed proxy:
        # dist = tr(P^-1 Q) - log(det(P^-1 Q)) - N
        
        inv_P = np.linalg.inv(M_rest)
        prod = np.dot(inv_P, self.cov_matrix)
        
        # Trace distance proxy (Very fast, O(N^2))
        dist = np.trace(prod) - np.log(np.linalg.det(prod)) - self.N
        
        return max(0, dist)

if __name__ == "__main__":
    engine = BCIRealTimeEngineEWMA(num_channels=18, alpha=0.05)
    baseline_M = np.eye(18)
    
    for i in range(1000):
        incoming_sample = np.random.randn(18)
        engine.update_step(incoming_sample)
        
        if i % 50 == 0:
            dist = engine.get_riemannian_distance(baseline_M)
            print(f"Sample {i} | Drift: {dist:.4f}")