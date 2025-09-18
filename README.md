# 🧠 Neural Network from Scratch - MNIST Digit Classification

A complete implementation of a neural network built from the ground up using only NumPy and pure mathematics, trained on the MNIST handwritten digit dataset.

## ✨ Features

- **Pure Mathematics**: No TensorFlow, PyTorch, or Keras - built with mathematical foundations
- **From Scratch Implementation**: All components implemented manually including:
  - Forward propagation
  - Backward propagation (backpropagation)
  - Gradient descent optimization
  - ReLU and Softmax activation functions
  - One-hot encoding
- **MNIST Dataset**: Trained on 42,000 handwritten digit images (0-9)
- **High Accuracy**: Achieves ~86.9% accuracy on training data

## 🏗️ Architecture

### Network Structure
- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

### Mathematical Components

#### Forward Propagation
```
Z₁ = W₁X + b₁
A₁ = ReLU(Z₁)
Z₂ = W₂A₁ + b₂  
A₂ = Softmax(Z₂)
```

#### Backward Propagation
```
dZ₂ = A₂ - Y (one-hot encoded labels)
dW₂ = (1/m) × dZ₂ × A₁ᵀ
db₂ = (1/m) × Σ(dZ₂)
dZ₁ = W₂ᵀ × dZ₂ ⊙ ReLU'(Z₁)
dW₁ = (1/m) × dZ₁ × Xᵀ
db₁ = (1/m) × Σ(dZ₁)
```

#### Parameter Updates
```
W₁ = W₁ - α × dW₁
b₁ = b₁ - α × db₁
W₂ = W₂ - α × dW₂
b₂ = b₂ - α × db₂
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib
```

### Dataset
The implementation uses the MNIST dataset in CSV format:
- Training data: `train.csv` (42,000 samples)
- Each row: `label, pixel0, pixel1, ..., pixel783`
- Pixel values: 0-255 (grayscale)

### Usage

1. **Load and preprocess data**:
```python
df = pd.read_csv('Dataset/train.csv')
data = np.array(df)
np.random.shuffle(data)

# Split features and labels
X_train = data[:, 1:].T / 255.0  # Normalize to [0,1]
y_train = data[:, 0]
```

2. **Initialize and train the network**:
```python
# Initialize parameters
W1, b1, W2, b2 = init_params()

# Train with gradient descent
W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 600)
```

3. **Make predictions**:
```python
predictions = make_predictions(X_test, W1, b1, W2, b2)
```

## 📊 Results

### Training Progress
- **Initial Accuracy**: ~15.7% (random initialization)
- **Final Accuracy**: ~86.9% (after 600 iterations)
- **Learning Rate**: 0.10
- **Training Time**: 600 iterations

### Sample Predictions
The model successfully predicts handwritten digits with high confidence, demonstrating effective learning of digit patterns and features.

## 🔍 Key Functions

### Core Components

- `init_params()`: Initialize weights and biases randomly
- `ReLU(Z)`: ReLU activation function
- `softmax(Z)`: Softmax activation for output layer
- `forward_propagation()`: Forward pass through network
- `backward_propagation()`: Compute gradients via backpropagation
- `update_params()`: Update parameters using gradient descent
- `get_predictions()`: Convert probabilities to predicted classes
- `get_accuracy()`: Calculate prediction accuracy

### Utility Functions

- `one_hot(Y)`: Convert labels to one-hot encoded vectors
- `make_predictions()`: Generate predictions for new data
- `test_prediction()`: Visualize individual predictions

## 📈 Mathematical Insights

### Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Prevents vanishing gradients
- **Softmax**: `f(xᵢ) = e^xᵢ / Σ(e^xⱼ)` - Converts outputs to probabilities

### Loss Function
- **Cross-entropy loss**: Measures difference between predicted and actual probability distributions
- **Gradient**: Simplified to `A₂ - Y` for softmax + cross-entropy combination

### Optimization
- **Gradient Descent**: Iteratively minimizes loss by moving in direction of negative gradient
- **Learning Rate**: Controls step size (0.10 chosen for stable convergence)

## 🎯 Educational Value

This implementation demonstrates:

1. **Pure Mathematical Understanding**: Every operation implemented from mathematical principles
2. **Gradient Computation**: Manual derivation and implementation of backpropagation
3. **Matrix Operations**: Efficient vectorized operations using NumPy
4. **Neural Network Fundamentals**: Core concepts without framework abstractions
5. **Optimization Theory**: Practical application of gradient descent

## 🔧 Code Structure

- **Data Processing**: MNIST loading, normalization, shuffling
- **Initialization**: Random weight initialization with proper scaling
- **Forward Pass**: Layer-by-layer computation with activations
- **Backward Pass**: Gradient computation using chain rule
- **Training Loop**: Iterative parameter updates with progress tracking
- **Evaluation**: Accuracy calculation and visualization

## 🚀 Future Enhancements

Possible improvements and extensions:
- Multiple hidden layers (deep neural network)
- Different activation functions (Sigmoid, Tanh, Leaky ReLU)
- Regularization techniques (L1/L2 regularization, dropout)
- Advanced optimizers (Adam, RMSprop, momentum)
- Batch processing and mini-batch gradient descent
- Validation set evaluation and early stopping
- Hyperparameter tuning and cross-validation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements or additional features!

---

*Built with passion for understanding the mathematical foundations of neural networks! 🧮✨*