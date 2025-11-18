"""
Complete Elman Network Implementation with 4 Hidden Units

A fully functional Simple Recurrent Network (Elman Network) for sequence prediction.

Total parameters: 29
- W_ih: 4 weights (Input → Hidden)
- W_hh: 16 weights (Context → Hidden, 4x4 matrix)  
- W_ho: 4 weights (Hidden → Output)
- b_h: 4 biases (Hidden layer)
- b_o: 1 bias (Output layer)
"""

import numpy as np

class ElmanNetwork:
    """
    Simple Recurrent Network (Elman Network) with 4 hidden units
    
    Architecture:
    - 1 input unit
    - 4 hidden units
    - 4 context units (copy of previous hidden state)
    - 1 output unit
    """
    
    def __init__(self, learning_rate=0.1):
        """
        Initialize the Elman network with 4 hidden units
        
        Args:
            learning_rate: Learning rate for gradient descent (default: 0.1)
        """
        self.learning_rate = learning_rate
        
        # W_ih: Input → Hidden (4 weights)
        # Using specific values from the essay for consistency
        self.W_ih = np.array([[0.5], [0.3], [0.8], [0.2]])
        
        # W_hh: Context → Hidden (4x4 matrix = 16 weights)
        self.W_hh = np.array([
            [0.1, 0.2, 0.0, 0.1],
            [0.3, 0.1, 0.2, 0.0],
            [0.0, 0.2, 0.3, 0.1],
            [0.2, 0.0, 0.1, 0.2]
        ])
        
        # W_ho: Hidden → Output (4 weights)
        self.W_ho = np.array([[0.4, 0.3, 0.5, 0.2]])
        
        # b_h: Hidden biases (4 values)
        self.b_h = np.array([[0.1], [0.1], [0.1], [0.1]])
        
        # b_o: Output bias (1 value)
        self.b_o = np.array([[0.0]])
        
        # Context units (initialized to zeros)
        self.context = np.zeros((4, 1))
    
    def sigmoid(self, x):
        """
        Sigmoid activation function
        Args: x: Input value or array
        Returns: Sigmoid of x: 1 / (1 + e^(-x))
        """
        # Clip to avoid overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, y):
        """
        Derivative of sigmoid function
        Args: y: Output of sigmoid function (already computed)
        Returns: Derivative: y * (1 - y)
        """
        return y * (1 - y)
    
    def forward(self, x):
        """
        Forward pass: compute hidden state and output for one time step
        
        Args:
            x: Input value (scalar, will be converted to array)
            
        Returns:
            output: Predicted output (scalar)
            hidden: Hidden state (4x1 array)
        """
        # Convert input to column vector
        x_vec = np.array([[x]])
        
        # Compute hidden layer input
        # hidden_input = W_ih * x + W_hh * context + b_h
        hidden_input = np.dot(self.W_ih, x_vec) + np.dot(self.W_hh, self.context) + self.b_h
        
        # Apply sigmoid activation
        hidden = self.sigmoid(hidden_input)
        
        # Compute output layer input
        # output_input = W_ho * hidden + b_o
        output_input = np.dot(self.W_ho, hidden) + self.b_o
        
        # Apply sigmoid activation
        output = self.sigmoid(output_input)
        
        # Return scalar output and hidden state
        return output[0, 0], hidden
    
    def backward(self, x, hidden, output, target):
        """
        Backward pass: compute gradients and update weights
        
        Args:
            x: Input that was used (scalar)
            hidden: Hidden state from forward pass (4x1 array)
            output: Output from forward pass (scalar)
            target: Target/desired output (scalar)
            
        Returns:
            error: Difference between target and output
        """
        # Convert scalars to proper shapes
        x_vec = np.array([[x]])
        output_vec = np.array([[output]])
        target_vec = np.array([[target]])
        
        # 1. Compute output layer error
        error = output_vec - target_vec
        delta_o = error * self.sigmoid_derivative(output_vec)
        
        # 2. Compute hidden layer gradients
        delta_h = np.dot(self.W_ho.T, delta_o) * self.sigmoid_derivative(hidden)
        
        # 3. Update W_ho: Hidden → Output
        self.W_ho -= self.learning_rate * np.dot(delta_o, hidden.T)
        
        # 4. Update b_o: Output bias
        self.b_o -= self.learning_rate * delta_o
        
        # 5. Update W_ih: Input → Hidden
        self.W_ih -= self.learning_rate * np.dot(delta_h, x_vec.T)
        
        # 6. Update W_hh: Context → Hidden (recurrent)
        self.W_hh -= self.learning_rate * np.dot(delta_h, self.context.T)
        
        # 7. Update b_h: Hidden biases
        self.b_h -= self.learning_rate * delta_h
        
        # Return error as scalar
        return error[0, 0]
    
    def update_context(self, hidden):
        """
        Update context units with current hidden state
        
        Args:
            hidden: Current hidden state (4x1 array)
        """
        self.context = hidden.copy()
    
    def reset_context(self):
        """
        Reset context to zeros (call at start of new sequence)
        """
        self.context = np.zeros((4, 1))
    
    def train_step(self, x, target):
        """
        Complete training step: forward + backward + update context
        
        Args:
            x: Input value (scalar)
            target: Target output (scalar)
            
        Returns:
            output: Predicted output
            error: Prediction error
        """
        # Forward pass
        output, hidden = self.forward(x)
        
        # Backward pass
        error = self.backward(x, hidden, output, target)
        
        # Update context for next time step
        self.update_context(hidden)
        
        return output, error
    
    def train_sequence(self, sequence, num_epochs=100, verbose=True):
        """
        Train on a complete sequence for multiple epochs
        
        Args:
            sequence: List of values (e.g., [0, 1, 1, 0, 1, ...])
            num_epochs: Number of times to iterate through sequence
            verbose: Whether to print progress
            
        Returns:
            losses: List of average loss per epoch
        """
        losses = []
        
        for epoch in range(num_epochs):
            # Reset context at start of sequence
            self.reset_context()
            
            total_loss = 0
            
            # Train on sequence
            for t in range(len(sequence) - 1):
                x = sequence[t]
                target = sequence[t + 1]
                
                output, error = self.train_step(x, target)
                
                # Accumulate squared error
                total_loss += error ** 2
            
            # Calculate average loss
            avg_loss = total_loss / (len(sequence) - 1)
            losses.append(avg_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict_sequence(self, sequence):
        """
        Predict next value for each position in sequence (without training)
        
        Args:
            sequence: List of values
            
        Returns:
            predictions: List of predicted values
        """
        # Reset context
        self.reset_context()
        
        predictions = []
        
        # Make predictions
        for t in range(len(sequence) - 1):
            x = sequence[t]
            output, hidden = self.forward(x)
            
            predictions.append(output)
            
            # Update context for next step
            self.update_context(hidden)
        
        return predictions
    
    def get_weights(self):
        """
        Get all network weights as a dictionary
        
        Returns:
            Dictionary with all 29 parameters
        """
        return {
            'W_ih': self.W_ih.copy(),
            'W_hh': self.W_hh.copy(),
            'W_ho': self.W_ho.copy(),
            'b_h': self.b_h.copy(),
            'b_o': self.b_o.copy()
        }
    
    def set_weights(self, weights):
        """
        Set network weights from a dictionary
        
        Args:
            weights: Dictionary with keys W_ih, W_hh, W_ho, b_h, b_o
        """
        self.W_ih = weights['W_ih'].copy()
        self.W_hh = weights['W_hh'].copy()
        self.W_ho = weights['W_ho'].copy()
        self.b_h = weights['b_h'].copy()
        self.b_o = weights['b_o'].copy()

# ============================================================================
# Example Usage and Testing
# ============================================================================
def generate_xor_sequence(length=100, seed=None):
    """
    Generate XOR sequence where each bit is XOR of previous two bits
    
    Args:
        length: Total length of sequence
        seed: Random seed for reproducibility
        
    Returns:
        sequence: List of bits (0s and 1s)
    """
    if seed is not None:
        np.random.seed(seed)
    
    sequence = [np.random.randint(0, 2), np.random.randint(0, 2)]
    
    for i in range(2, length):
        sequence.append(sequence[i-1] ^ sequence[i-2])
    
    return sequence

def main():
    """
    Main function demonstrating the Elman network
    """
    print("=" * 70)
    print("Elman Network with 4 Hidden Units - Complete Implementation")
    print("=" * 70)
    print()
    
    # Create network
    print("Creating Elman network...")
    network = ElmanNetwork(learning_rate=0.1)
    print(f"Total parameters: 29 (W_ih:4, W_hh:16, W_ho:4, b_h:4, b_o:1)")
    print()
    
    # Generate XOR sequence
    print("Generating XOR sequence (100 bits)...")
    sequence = generate_xor_sequence(100, seed=42)
    print(f"First 20 bits: {sequence[:20]}")
    print()
    
    # Train network
    print("Training network...")
    losses = network.train_sequence(sequence, num_epochs=100, verbose=True)
    print()
    
    # Test predictions
    print("Testing predictions on first 20 bits...")
    test_sequence = sequence[:20]
    predictions = network.predict_sequence(test_sequence)
    
    print("\nPredictions vs Actual:")
    print("Time | Input | Actual Next | Predicted | Rounded | Match")
    print("-" * 65)
    
    correct = 0
    for i in range(len(predictions)):
        input_val = test_sequence[i]
        actual = test_sequence[i + 1]
        predicted = predictions[i]
        rounded = 1 if predicted > 0.5 else 0
        match = "✓" if actual == rounded else "✗"
        
        if actual == rounded:
            correct += 1
        
        print(f" {i:2d}  |   {input_val}   |      {actual}      |  {predicted:.4f}   |    {rounded}    |  {match}")
    
    accuracy = (correct / len(predictions)) * 100
    print("-" * 65)
    print(f"Accuracy: {correct}/{len(predictions)} = {accuracy:.1f}%")
    print()
    
    # Show final weights
    weights = network.get_weights()
    print("Final weights (first few):")
    print(f"W_ih: {weights['W_ih'].flatten()}")
    print(f"W_ho: {weights['W_ho'].flatten()}")
    print()
    
    # Plot training loss
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss over Time')
        plt.grid(True, alpha=0.3)
        
        # Prompt user for save location
        print("\nWhere would you like to save the training plot?")
        save_path = input("Enter file path (or press Enter for default '/mnt/user-data/outputs/elman_4unit_training.png'): ").strip()
        print("\nsave_path:", save_path)
        if not save_path:
            save_path = '/mnt/user-data/outputs/elman_4unit_training.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")
    except ImportError:
        print("Matplotlib not available, skipping plot")
if __name__ == "__main__":
    main()
