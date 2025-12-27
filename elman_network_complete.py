"""
Extended Elman Network Implementation

Supports both:
1. XOR task (1-bit input/output)
2. Consonant-Vowel prediction task (6-bit input/output)

Usage:
    python elman_network_extended.py          # Run both tasks (default)
    python elman_network_extended.py XOR      # Run only XOR task
    python elman_network_extended.py CV       # Run only CV task
"""

import numpy as np
from datetime import datetime

class ElmanNetwork:
    """
    Flexible Simple Recurrent Network (Elman Network)
    
    Architecture:
    - n_input input units
    - n_hidden hidden units
    - n_hidden context units (copy of previous hidden state)
    - n_output output units
    """
    
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1):
        """
        Initialize the Elman network
        
        Args:
            n_input: Number of input units
            n_hidden: Number of hidden units
            n_output: Number of output units
            learning_rate: Learning rate for gradient descent (default: 0.1)
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        # W_ih: Input → Hidden (n_hidden x n_input)
        self.W_ih = np.random.randn(n_hidden, n_input) * 0.1
        
        # W_hh: Context → Hidden (n_hidden x n_hidden matrix)
        self.W_hh = np.random.randn(n_hidden, n_hidden) * 0.1
        
        # W_ho: Hidden → Output (n_output x n_hidden)
        self.W_ho = np.random.randn(n_output, n_hidden) * 0.1
        
        # b_h: Hidden biases (n_hidden x 1)
        self.b_h = np.zeros((n_hidden, 1))
        
        # b_o: Output bias (n_output x 1)
        self.b_o = np.zeros((n_output, 1))
        
        # Context units (initialized to zeros)
        self.context = np.zeros((n_hidden, 1))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, y):
        """Derivative of sigmoid function"""
        return y * (1 - y)
    
    def forward(self, x):
        """
        Forward pass: compute hidden state and output for one time step
        
        Args:
            x: Input vector (n_input x 1) or scalar (for 1D input)
            
        Returns:
            output: Predicted output (n_output x 1)
            hidden: Hidden state (n_hidden x 1)
        """
        # Convert input to column vector if needed
        if np.isscalar(x):
            x_vec = np.array([[x]])
        elif x.ndim == 1:
            x_vec = x.reshape(-1, 1)
        else:
            x_vec = x
        
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
        
        return output, hidden
    
    def backward(self, x, hidden, output, target):
        """
        Backward pass: compute gradients and update weights
        
        Args:
            x: Input that was used (n_input x 1) or scalar
            hidden: Hidden state from forward pass (n_hidden x 1)
            output: Output from forward pass (n_output x 1)
            target: Target/desired output (n_output x 1) or scalar
            
        Returns:
            error: Mean squared error
        """
        # Convert scalars to proper shapes
        if np.isscalar(x):
            x_vec = np.array([[x]])
        elif x.ndim == 1:
            x_vec = x.reshape(-1, 1)
        else:
            x_vec = x
            
        if np.isscalar(target):
            target_vec = np.array([[target]])
        elif target.ndim == 1:
            target_vec = target.reshape(-1, 1)
        else:
            target_vec = target
        
        # 1. Compute output layer error
        error = output - target_vec
        delta_o = error * self.sigmoid_derivative(output)
        
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
        
        # Return mean squared error
        return np.mean(error ** 2)
    
    def update_context(self, hidden):
        """Update context units with current hidden state"""
        self.context = hidden.copy()
    
    def reset_context(self):
        """Reset context to zeros (call at start of new sequence)"""
        self.context = np.zeros((self.n_hidden, 1))
    
    def train_step(self, x, target):
        """
        Complete training step: forward + backward + update context
        
        Args:
            x: Input value or vector
            target: Target output value or vector
            
        Returns:
            output: Predicted output
            error: Prediction error (MSE)
        """
        # Forward pass
        output, hidden = self.forward(x)
        
        # Backward pass
        error = self.backward(x, hidden, output, target)
        
        # Update context for next time step
        self.update_context(hidden)
        
        return output, error
    
    def train_sequence(self, sequence, num_epochs=600, verbose=True):
        """
        Train on a complete sequence for multiple epochs
        
        Args:
            sequence: List of input values/vectors
            num_epochs: Number of times to iterate through sequence
            verbose: Whether to print progress
            
        Returns:
            losses: List of average loss per epoch
        """
        losses = []
        
        for epoch in range(num_epochs):
            # Reset context at start of sequence
            self.reset_context() #Reset context to zeros
            
            total_loss = 0
            
            # Train on sequence
            for t in range(len(sequence) - 1):
                x = sequence[t]
                target = sequence[t + 1]
                
                output, error = self.train_step(x, target)
                
                # Accumulate error
                total_loss += error
            
            # Calculate average loss
            avg_loss = total_loss / (len(sequence) - 1)
            losses.append(avg_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        return losses
    
    def predict_sequence(self, sequence):
        """
        Use the trained network to predict next value for each position in sequence.
        
        Args:
            sequence: List of values/vectors
            
        Returns:
            predictions: List of predicted values/vectors
        """
        # Reset context
        self.reset_context() #Reset context to zeros
        
        predictions = []
        
        # Make predictions
        for t in range(len(sequence) - 1):
            x = sequence[t]
            output, hidden = self.forward(x)
            
            predictions.append(output)
            
            # Update context for next step
            self.update_context(hidden)
        
        return predictions


# ============================================================================
# XOR Task Functions
# ============================================================================

def generate_xor_sequence(length=100, seed=None):
    """
    Generate XOR sequence in Elman's format: groups of (bit1, bit2, bit1 XOR bit2)
    
    Args:
        length: Total length of sequence (will be rounded down to nearest multiple of 3)
        seed: Random seed for reproducibility
        
    Returns:
        sequence: List of bits (0s and 1s) in groups of 3
    """
    if seed is not None:
        np.random.seed(seed)
    
    sequence = []
    num_groups = length // 3
    
    for _ in range(num_groups):
        bit1 = np.random.randint(0, 2)
        bit2 = np.random.randint(0, 2)
        xor_result = bit1 ^ bit2
        
        sequence.extend([bit1, bit2, xor_result])
    
    return sequence


# ============================================================================
# Consonant-Vowel Task Functions
# ============================================================================

# Letter representations as 6-bit vectors
# Bits: [Consonant, Vowel, Interrupted, High, Back, Voiced]
LETTERS = {
    'b': np.array([1, 0, 1, 0, 0, 1]),
    'd': np.array([1, 0, 1, 1, 0, 1]),
    'g': np.array([1, 0, 1, 0, 1, 1]),
    'a': np.array([0, 1, 0, 0, 1, 1]),
    'i': np.array([0, 1, 0, 1, 0, 1]),
    'u': np.array([0, 1, 0, 1, 1, 1])
}

# Consonant-vowel replacement rules
CV_RULES = {
    'b': 'ba',      # b followed by 1 a
    'd': 'dii',     # d followed by 2 i's
    'g': 'guuu'     # g followed by 3 u's
}

def generate_cv_sequence(length=1000, seed=None):
    """
    Generate consonant-vowel sequence following Elman's rules
    
    Args:
        length: Approximate length of final sequence
        seed: Random seed for reproducibility
        
    Returns:
        sequence: List of 6-bit vectors representing letters
        letter_sequence: List of letter names (for visualization)
    """
    if seed is not None:
        np.random.seed(seed)
    
    consonants = ['b', 'd', 'g']
    
    # Generate random consonant sequence
    num_consonants = length // 3  # Rough estimate
    consonant_seq = [consonants[np.random.randint(0, 3)] for _ in range(num_consonants)]
    
    # Expand using CV rules
    letter_sequence = []
    for c in consonant_seq:
        letter_sequence.extend(list(CV_RULES[c]))
    
    # Convert to 6-bit vectors
    vector_sequence = [LETTERS[letter] for letter in letter_sequence]
    
    return vector_sequence, letter_sequence

def analyze_hidden_activations(network, sequence, letters, max_steps=30):
    """
    Analyze hidden unit activations over time (recreates Elman 1990 Figure 4)
    
    Args:
        network: Trained ElmanNetwork
        sequence: List of input vectors
        letters: Corresponding letter names
        max_steps: Number of time steps to analyze
        
    Returns:
        activations: Array of shape (n_hidden, max_steps) with hidden unit activations
    """
    network.reset_context()
    
    n_steps = min(max_steps, len(sequence))
    activations = np.zeros((network.n_hidden, n_steps))
    
    for t in range(n_steps):
        x = sequence[t]
        output, hidden = network.forward(x)
        
        # Store hidden activations (flatten to 1D)
        activations[:, t] = hidden.flatten()
        
        # Update context for next step
        network.update_context(hidden)
    
    return activations

def plot_figure4_prediction_errors(test_letters, errors, max_steps=30):
    """
    Recreate Figure 4 from Elman (1990): Prediction errors over time
    Shows prediction error for each letter in the CV sequence
    
    Args:
        test_letters: List of letter names in sequence
        errors: List of prediction errors (MSE) for each time step
        max_steps: Number of time steps to show
    """
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    n_steps = min(max_steps, len(errors))
    
    # Create figure
    plt.figure(figsize=(14, 6))
    
    # Plot errors as points connected by lines
    plt.plot(range(n_steps), errors[:n_steps], 'o-', markersize=6, linewidth=1.5)
    
    # Set up x-axis with letter labels
    plt.xticks(range(n_steps), test_letters[:n_steps], fontsize=10)
    plt.xlabel('Letter in Sequence', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Prediction Error by Letter Position (Elman 1990, Figure 4)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at mean error
    mean_error = np.mean(errors[:n_steps])
    plt.axhline(y=mean_error, color='r', linestyle='--', alpha=0.5, 
                label=f'Mean Error: {mean_error:.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    # Save
    output_dir = '/mnt/user-data/outputs' if os.path.exists('/mnt/user-data') else './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'elman_figure4_prediction_errors_{timestamp}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure 4 (Prediction Errors) saved to: {save_path}")
    
    plt.show()

def plot_hidden_unit_activations(network, sequence, letters, max_steps=30):
    """
    Recreate Figure 4 from Elman (1990): Hidden unit activations over time
    Shows how different hidden units respond to consonant-vowel patterns
    
    Args:
        network: Trained ElmanNetwork
        sequence: CV sequence vectors
        letters: Letter names for labeling
        max_steps: Number of time steps to show
    """
    import matplotlib.pyplot as plt
    
    # Get hidden activations
    activations = analyze_hidden_activations(network, sequence, letters, max_steps)
    
    # Create figure with subplots for each hidden unit
    n_hidden = network.n_hidden
    n_rows = 4
    n_cols = 5  # For 20 hidden units
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    fig.suptitle('Hidden Unit Activations Over Time (Elman 1990, Figure 4)', 
                 fontsize=14, fontweight='bold')
    
    for i in range(n_hidden):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot activation over time
        ax.plot(activations[i, :], linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_title(f'Unit {i+1}', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add letter labels on x-axis
        if row == n_rows - 1:  # Bottom row
            ax.set_xlabel('Time', fontsize=8)
            # Show some letter labels
            tick_positions = range(0, max_steps, 3)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([letters[i] if i < len(letters) else '' 
                               for i in tick_positions], fontsize=7)
        else:
            ax.set_xticks([])
        
        if col == 0:  # Left column
            ax.set_ylabel('Activation', fontsize=8)
        
    plt.tight_layout()
    
    # Save
    import os
    from datetime import datetime
    
    output_dir = '/mnt/user-data/outputs' if os.path.exists('/mnt/user-data') else './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'elman_figure4_hidden_activations_{timestamp}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure 4 (Hidden Activations) saved to: {save_path}")
    
    return activations

def analyze_bit_errors(predictions, targets, bit_names=None):
    """
    Analyze prediction errors on a bit-by-bit basis
    
    Args:
        predictions: List of predicted vectors
        targets: List of target vectors
        bit_names: Optional list of names for each bit
        
    Returns:
        bit_errors: Array of shape (n_bits, n_predictions) with squared errors
    """
    n_bits = predictions[0].shape[0]
    n_predictions = len(predictions)
    
    bit_errors = np.zeros((n_bits, n_predictions))
    
    for t in range(n_predictions):
        pred = predictions[t].flatten()
        target = targets[t].flatten()
        bit_errors[:, t] = (pred - target) ** 2
    
    return bit_errors


# ============================================================================
# Example Usage
# ============================================================================

def xor_task():
    """Test the XOR task"""
    print("=" * 70)
    print("XOR Task Test")
    print("=" * 70)
    
    # Create network for XOR (1 input, 4 hidden, 1 output)
    network = ElmanNetwork(n_input=1, n_hidden=4, n_output=1, learning_rate=0.1)
    
    # Generate XOR sequence
    sequence = generate_xor_sequence(3000, seed=42)
    print(f"First 15 bits: {sequence[:15]}")
    print()
    
    # Train network
    print("Training...")
    losses = network.train_sequence(sequence, num_epochs=600, verbose=True)
    print()
    
    # Test predictions
    test_sequence = sequence[:20]
    network.reset_context() #Reset context to zeros
    predictions = network.predict_sequence(test_sequence)
    
    print("Actual Output vs Expected Output:")
    print("Time | Input | Actual Out | Rounded Out | Expected | Match")
    print("-" * 60)
    
    correct = 0
    for i in range(len(predictions)):
        input_val = test_sequence[i]
        expect = test_sequence[i + 1]
        actual = predictions[i][0, 0]
        actual_rounded = 1 if actual > 0.5 else 0
        match = "✓" if expect == actual_rounded else "✗"

        if expect == actual_rounded:
            correct += 1

        print(f" {i:2d}  |   {input_val}   |   {actual:.4f}   |      {actual_rounded}      |     {expect}    |  {match}")

    accuracy = (correct / len(predictions)) * 100
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(predictions)} = {accuracy:.1f}%")
    print()
    
    # Test predictions of third bit only.
    # Uncomment the following to see accuracy at only the XOR output positions.
    # print("Actual vs Expected Only At XOR Positions:")
    # print("Time | Input | Actual Out 3 | Rounded Out | Expected | Match")
    # print("-" * 60)
    
    # correct = 0
    # for i in range(1, len(predictions), 3):
    #     input_val = test_sequence[i]
    #     expect = test_sequence[i + 1]
    #     actual = predictions[i][0, 0]
    #     actual_rounded = 1 if actual > 0.5 else 0
    #     match = "✓" if expect == actual_rounded else "✗"

    #     if expect == actual_rounded:
    #         correct += 1

    #     print(f" {i:2d}  |   {input_val}   |   {actual:.4f}   |      {actual_rounded}      |     {expect}    |  {match}")

    # accuracy = (correct / (len(predictions) // 3)) * 100
    # print("-" * 60)
    # print(f"Accuracy3: {correct}/{round(len(predictions)/3)} = {accuracy:.0f}%")
    # print()
    
    # Error rate at each position.
    # I want something that looks at errors at each of the three positions separately.
    # I want the root mean square error at each position.
    # Uncomment the following to see error rates by position in group of 3.
    
    # position_errors = [0, 0, 0]
    # position_counts = [0, 0, 0]
    # for i in range(len(predictions)):
    #     input_val = test_sequence[i]
    #     expect = test_sequence[i + 1]
    #     actual = predictions[i][0, 0]
    #     actual_rounded = 1 if actual > 0.5 else 0

    #     pos = i % 3
    #     position_counts[pos] += 1
    #     if expect != actual_rounded:
    #         position_errors[pos] += 1
    # print("Error Rate by Position in Group of 3:")
    # print("Position | Errors | Total | Error Rate")
    # print("-" * 40)
    # for pos in range(3):
    #     errors = position_errors[pos]
    #     total = position_counts[pos]
    #     error_rate = (errors / total) * 100
    #     print(f"   {pos}     |  {errors:3d}  |  {total:3d} |   {error_rate:.2f}%")
    # print()
        
        
    
    return network, losses


def cv_task():
    """Test the Consonant-Vowel task"""
    print("=" * 70)
    print("Consonant-Vowel Task Test")
    print("=" * 70)
    
    # Create network for CV task (6 input, 20 hidden, 6 output)
    network = ElmanNetwork(n_input=6, n_hidden=20, n_output=6, learning_rate=0.1)
    
    # Generate CV sequence
    sequence, letters = generate_cv_sequence(1000, seed=42)
    print(f"First 20 letters: {''.join(letters[:20])}")
    print(f"Sequence length: {len(sequence)} vectors")
    print()
    
    # Train network
    print("Training...")
    losses = network.train_sequence(sequence, num_epochs=200, verbose=True)
    print()
    
    # Test predictions on new sequence
    test_sequence, test_letters = generate_cv_sequence(60, seed=123)
    network.reset_context() #Reset context to zeros
    predictions = network.predict_sequence(test_sequence)
    
    # Calculate errors
    targets = test_sequence[1:]
    errors = [np.mean((predictions[i] - targets[i])**2) for i in range(len(predictions))]
    
    print("Prediction errors for first 20 time steps:")
    print("Time | Letter | Next  | Error")
    print("-" * 40)
    
    for i in range(min(20, len(errors))):
        print(f" {i:2d}  |   {test_letters[i]}    |   {test_letters[i+1]}   | {errors[i]:.6f}")
    
    print()
    
    # Analyze bit-by-bit errors
    bit_names = ['Consonant', 'Vowel', 'Interrupted', 'High', 'Back', 'Voiced']
    bit_errors = analyze_bit_errors(predictions, targets, bit_names)
    
    print("Average error per bit feature:")
    for i, name in enumerate(bit_names):
        avg_error = np.mean(bit_errors[i, :])
        print(f"  {name:12s}: {avg_error:.6f}")
    
    print()
    
    print("Generating Hidden Unit Activations...")
    activations = plot_hidden_unit_activations(network, test_sequence, test_letters, max_steps=30)
    
    # Generate Figure 4 - Prediction errors by letter
    print("Generating Figure 4 (Prediction Errors by Letter)...")
    plot_figure4_prediction_errors(test_letters, errors, max_steps=30)
    
    
    return network, losses, bit_errors


def main(task='both'):
    """
    Run Elman network tasks
    
    Args:
        task: Which task(s) to run - 'XOR', 'CV', or 'both' (default)
    """
    task = task.upper()
    
    if task not in ['XOR', 'CV', 'BOTH']:
        print(f"Invalid task '{task}'. Must be 'XOR', 'CV', or 'both'")
        return
    
    xor_network = None
    xor_losses = None
    cv_network = None
    cv_losses = None
    bit_errors = None
    
    # Run XOR task if requested
    if task in ['XOR', 'BOTH']:
        xor_network, xor_losses = xor_task()
        
        if task == 'BOTH':
            print("\n" + "=" * 70 + "\n")
    
    # Run CV task if requested
    if task in ['CV', 'BOTH']:
        cv_network, cv_losses, bit_errors = cv_task()
    
    # Plot results if matplotlib available
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Create outputs directory if it doesn't exist
        # Try /mnt/user-data/outputs first (for Claude), fall back to local ./outputs
        output_dir = '/mnt/user-data/outputs' if os.path.exists('/mnt/user-data') else './outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        if task == 'BOTH':
            # Plot both tasks
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # XOR training loss
            axes[0, 0].plot(xor_losses)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('XOR Task: Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
            # CV training loss
            axes[0, 1].plot(cv_losses)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('CV Task: Training Loss')
            axes[0, 1].grid(True, alpha=0.3)
            
            # CV bit errors (like Figure 5 from Elman)
            bit_names = ['Consonant', 'Vowel', 'Interrupted', 'High', 'Back', 'Voiced']
            
            # Plot bit 0 (Consonant)
            axes[1, 0].plot(bit_errors[0, :30], 'o-', markersize=4)
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Squared Error')
            axes[1, 0].set_title(f'Bit 1: {bit_names[0]}')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot bit 3 (High)
            axes[1, 1].plot(bit_errors[3, :30], 'o-', markersize=4)
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Squared Error')
            axes[1, 1].set_title(f'Bit 4: {bit_names[3]}')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, 'elman_tasks_comparison.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlots saved to: {save_path}")
            
        elif task == 'XOR':
            # Plot only XOR
            plt.figure(figsize=(10, 5))
            plt.plot(xor_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('XOR Task: Training Loss')
            plt.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'elman_xor_training_{timestamp}.png'
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
            
        elif task == 'CV':
            # Plot only CV
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # CV training loss
            axes[0].plot(cv_losses)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('CV Task: Training Loss')
            axes[0].grid(True, alpha=0.3)
            
            # CV bit errors (like Figure 5 from Elman)
            bit_names = ['Consonant', 'Vowel', 'Interrupted', 'High', 'Back', 'Voiced']
            
            # Plot bit 0 (Consonant)
            axes[1].plot(bit_errors[0, :30], 'o-', markersize=4)
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('Squared Error')
            axes[1].set_title(f'Bit 1: {bit_names[0]}')
            axes[1].grid(True, alpha=0.3)
            
            # Plot bit 3 (High)
            axes[2].plot(bit_errors[3, :30], 'o-', markersize=4)
            axes[2].set_xlabel('Time Step')
            axes[2].set_ylabel('Squared Error')
            axes[2].set_title(f'Bit 4: {bit_names[3]}')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'elman_cv_training_{timestamp}.png'
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nCV Plot saved to: {save_path}")
            
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")


if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1:
        task_arg = sys.argv[1]
    else:
        task_arg = 'cv'
    
    main(task=task_arg)