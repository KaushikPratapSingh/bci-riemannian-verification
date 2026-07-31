import numpy as np
from collections import deque

class BCIProcessor:
    def __init__(self, confidence_threshold=0.75, buffer_size=5):
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.current_state = 0
        
    def process_window(self, model_output_probs):
        """
        model_output_probs: array-like, e.g., [prob_class0, prob_class1]
        """
        # 1. Confidence Thresholding
        max_prob = np.max(model_output_probs)
        predicted_class = np.argmax(model_output_probs)
        
        if max_prob < self.confidence_threshold:
            return None # Ignore uncertain predictions
            
        # 2. Temporal Smoothing (Majority Voting)
        self.buffer.append(predicted_class)
        if len(self.buffer) < self.buffer.maxlen:
            return None # Wait for buffer to fill
            
        # Decide state based on majority
        votes = np.bincount(self.buffer)
        new_state = np.argmax(votes)
        
        return new_state

# --- Simulation Integration Example ---
def run_enhanced_simulation(data_stream):
    processor = BCIProcessor(confidence_threshold=0.80, buffer_size=3)
    
    print(f"{'Index':<10} | {'Raw Prob':<15} | {'Decision':<15}")
    print("-" * 45)
    
    for i, window in enumerate(data_stream):
        # Dummy prediction logic representing your NN
        probs = [0.2, 0.8] if i % 10 == 0 else [0.9, 0.1]
        
        decision = processor.process_window(probs)
        
        if decision is not None:
            state_label = "ACTIVE" if decision == 1 else "REST"
            print(f"{i:<10} | {str(probs):<15} | {state_label:<15}")

# Example stream generator
dummy_data = [None] * 100
run_enhanced_simulation(dummy_data)