import torch
import torch.nn as nn
import numpy as np

# 1. Define a Mock Mini-EEGNet for demonstration
class MiniEEGNet(nn.Module):
    def __init__(self):
        super(MiniEEGNet, self).__init__()
        # Simplified: Input (1 channel, 64 time points)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=8)
        self.fc = nn.Linear(16 * 57, 2) # Output: 2 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 2. Saliency Mapping (Explainability)
def get_saliency_map(model, input_tensor):
    """
    Computes which parts of the input contribute most to the classification.
    """
    model.eval()
    input_tensor.requires_grad_()
    
    output = model(input_tensor)
    # Get the score for the highest probability class
    score, index = torch.max(output, 1)
    
    # Calculate gradient of the output with respect to the input
    score.backward()
    
    # The saliency map is the absolute value of the input gradients
    saliency = input_tensor.grad.data.abs()
    return saliency.squeeze().numpy()

# 3. Real-Time Streaming Simulation (Pipeline)
def simulate_streaming(model, data_stream, window_size=64, stride=10):
    """
    Simulates a real-time buffer processing EEG signals.
    """
    print(f"\n--- Starting Streaming Simulation (Window: {window_size}) ---")
    predictions = []
    
    # Process stream in sliding windows
    for i in range(0, len(data_stream) - window_size, stride):
        window = data_stream[i : i + window_size]
        # Reshape for model: [Batch=1, Channel=1, Samples=64]
        input_tensor = torch.tensor(window).view(1, 1, -1).float()
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)
        
        if i % (stride * 5) == 0:
            print(f"Processed window at index {i}: Classified as class {pred}")
            
    return predictions

# --- Execution ---
if __name__ == "__main__":
    # Initialize Model
    model = MiniEEGNet()
    
    # Generate Dummy EEG Data (1000 samples)
    dummy_data = np.random.randn(1000)
    
    # A. Explainability Test
    sample_window = torch.randn(1, 1, 64)
    saliency = get_saliency_map(model, sample_window)
    print("Saliency Map computed successfully.")
    print(f"Max influence observed at sample index: {np.argmax(saliency)}")
    
    # B. Streaming Test
    preds = simulate_streaming(model, dummy_data)
    print(f"\nSimulation complete. Total windows processed: {len(preds)}")