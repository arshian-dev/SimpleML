# SimpleML Neural Network Library - Complete Guide

A from-scratch explanation of building a neural network library in C++.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Neural Networks](#understanding-neural-networks)
3. [The Tensor Class](#the-tensor-class)
4. [Activation Functions](#activation-functions)
5. [Loss Functions](#loss-functions)
6. [Neural Network Layers](#neural-network-layers)
7. [Optimizers](#optimizers)
8. [Putting It All Together](#putting-it-all-together)
9. [Complete Training Example](#complete-training-example)

---

## Introduction

This library implements a minimal but complete neural network framework. It includes everything needed to:
- Store and manipulate multi-dimensional data (Tensors)
- Build neural network layers
- Train models using backpropagation
- Optimize weights using gradient descent

---

## Understanding Neural Networks

### What is a Neural Network?

A neural network is a computational model inspired by the human brain. It consists of:

1. **Neurons** - Basic computational units
2. **Layers** - Groups of neurons
3. **Weights** - Learnable parameters that connect neurons
4. **Biases** - Additional learnable parameters
5. **Activation Functions** - Non-linear transformations

### How Training Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. FORWARD PASS                                               │
│      Input → Layer 1 → Layer 2 → ... → Output                   │
│                                                                 │
│   2. COMPUTE LOSS                                               │
│      Loss = how wrong is our prediction?                        │
│                                                                 │
│   3. BACKWARD PASS (Backpropagation)                            │
│      Calculate gradients: ∂Loss/∂weights                        │
│                                                                 │
│   4. UPDATE WEIGHTS                                             │
│      weights = weights - learning_rate × gradients              │
│                                                                 │
│   5. REPEAT until loss is small                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Tensor Class

### What is a Tensor?

A tensor is a multi-dimensional array. Think of it as:
- **0D tensor**: A single number (scalar) → `5`
- **1D tensor**: A list of numbers (vector) → `[1, 2, 3]`
- **2D tensor**: A table of numbers (matrix) → `[[1,2], [3,4]]`
- **ND tensor**: Higher dimensional arrays

### Core Structure

```cpp
class Tensor {
private:
    std::vector<size_t> shape_;  // Dimensions, e.g., {2, 3} for 2x3 matrix
    std::vector<float> data_;    // Flattened data stored in row-major order
    std::vector<float> grad_;    // Gradients (same shape as data)
    bool requires_grad_;         // Do we need to compute gradients?
};
```

### Key Operations

#### 1. Creation
```cpp
// Create a 2x3 matrix filled with zeros
Tensor t({2, 3}, 0.0f);

// Create from existing data
Tensor t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
```

#### 2. Element Access
```cpp
// Tensors are stored as flat arrays (row-major order)
// For a 2x3 matrix:
//   [a, b, c]    stored as: [a, b, c, d, e, f]
//   [d, e, f]    indices:    0  1  2  3  4  5

float val = tensor[0];  // First element
tensor[5] = 10.0f;      // Set last element
```

#### 3. Matrix Multiplication (matmul)

This is the core operation in neural networks:

```
Matrix A (2×3)     Matrix B (3×2)     Result C (2×2)
[1, 2, 3]          [7,  8]            [58,  64]
[4, 5, 6]    ×     [9, 10]      =     [139, 154]
                   [11, 12]
```

The math: `C[i][j] = sum(A[i][k] * B[k][j])` for all k

```cpp
Tensor Tensor::matmul(const Tensor &other) const {
    size_t M = shape_[0];      // Rows of A
    size_t K = shape_[1];      // Cols of A = Rows of B
    size_t N = other.shape_[1]; // Cols of B
    
    Tensor result({M, N});
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += data_[i * K + k] * other.data_[k * N + j];
            }
            result.data_[i * N + j] = sum;
        }
    }
    return result;
}
```

---

## Activation Functions

### Why Do We Need Activations?

Without activations, a neural network is just a series of linear transformations:
```
y = W₃(W₂(W₁x)) = W'x  ← Still linear!
```

Activations add non-linearity, allowing networks to learn complex patterns.

### ReLU (Rectified Linear Unit)

The most popular activation function. Simple and effective.

```
ReLU(x) = max(0, x)

Graph:
        │    /
        │   /
        │  /
────────┼─/────────
        │
```

```cpp
// Forward
Tensor relu(const Tensor &input) {
    return input.apply([](float x) { 
        return x > 0.0f ? x : 0.0f; 
    });
}

// Backward: derivative is 1 if x > 0, else 0
Tensor relu_backward(const Tensor &input, const Tensor &grad_output) {
    Tensor result(input.get_shape());
    for (size_t i = 0; i < input.get_size(); ++i) {
        result[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
    }
    return result;
}
```

### Sigmoid

Squashes any value to range (0, 1). Good for probabilities.

```
σ(x) = 1 / (1 + e^(-x))

Graph:
      1 ─────────────────
        │           ____/
        │       ___/
      0.5      /
        │  ___/
        │_/
      0 ─────────────────
```

```cpp
// Forward
Tensor sigmoid(const Tensor &input) {
    return input.apply([](float x) { 
        return 1.0f / (1.0f + std::exp(-x)); 
    });
}

// Backward: derivative is σ(x) × (1 - σ(x))
Tensor sigmoid_backward(const Tensor &output, const Tensor &grad_output) {
    Tensor result(output.get_shape());
    for (size_t i = 0; i < output.get_size(); ++i) {
        float s = output[i];
        result[i] = grad_output[i] * s * (1.0f - s);
    }
    return result;
}
```

### Tanh

Similar to sigmoid but outputs range (-1, 1).

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Derivative: 1 - tanh²(x)
```

### Softmax

Converts a vector of scores into probabilities (sums to 1).

```cpp
// For numerical stability, subtract max before exponentiating
Tensor softmax(const Tensor &input) {
    // For each row in batch:
    // 1. Find max value
    // 2. Subtract max from all values
    // 3. Compute exp() for each value
    // 4. Divide by sum of exp values
}
```

---

## Loss Functions

Loss functions measure how wrong our predictions are. We want to minimize this.

### Mean Squared Error (MSE)

Best for regression problems (predicting continuous values).

```
MSE = (1/n) × Σ(prediction - target)²
```

```cpp
class MSELoss {
public:
    float forward(const Tensor &predictions, const Tensor &targets) {
        float sum = 0.0f;
        for (size_t i = 0; i < predictions.get_size(); ++i) {
            float diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return sum / predictions.get_size();
    }
    
    // Gradient: ∂MSE/∂prediction = 2(prediction - target) / n
    Tensor backward() {
        Tensor grad(cached_predictions_.get_shape());
        float n = cached_predictions_.get_size();
        for (size_t i = 0; i < n; ++i) {
            grad[i] = 2.0f * (cached_predictions_[i] - cached_targets_[i]) / n;
        }
        return grad;
    }
};
```

### Binary Cross-Entropy (BCE)

Best for binary classification (yes/no problems).

```
BCE = -mean(target × log(pred) + (1-target) × log(1-pred))
```

### Cross-Entropy

Best for multi-class classification.

```
CE = -Σ(target × log(softmax(pred)))
```

---

## Neural Network Layers

### The Dense (Fully Connected) Layer

Every input connects to every output.

```
Inputs (3)         Outputs (2)
    ○─────────────────○
    │ ╲           ╱ │
    │   ╲       ╱   │
    ○─────×─────────○
    │   ╱       ╲   │
    │ ╱           ╲ │
    ○─────────────────
         weights
```

#### Forward Pass

```
output = input × weights + bias
```

```cpp
Tensor forward(const Tensor &input) {
    // input shape: (batch_size, input_features)
    // weights shape: (input_features, output_features)
    // output shape: (batch_size, output_features)
    
    cached_input_ = input;
    
    // Matrix multiplication
    Tensor output = input.matmul(weights_);
    
    // Add bias (broadcast across batch)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t j = 0; j < output_size_; ++j) {
            output[b * output_size_ + j] += bias_[j];
        }
    }
    
    // Apply activation
    if (activation_ == Activation::ReLU) {
        output = Activations::relu(output);
    }
    
    return output;
}
```

#### Backward Pass (Backpropagation)

The chain rule tells us how to compute gradients:

```
∂Loss/∂weights = input^T × ∂Loss/∂output
∂Loss/∂bias = sum(∂Loss/∂output)
∂Loss/∂input = ∂Loss/∂output × weights^T
```

```cpp
Tensor backward(const Tensor &grad_output) {
    Tensor grad = grad_output;
    
    // If we used an activation, backprop through it first
    if (activation_ == Activation::ReLU) {
        grad = Activations::relu_backward(cached_pre_activation_, grad);
    }
    
    // Gradient for bias: sum over batch dimension
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t j = 0; j < output_size_; ++j) {
            grad_bias_[j] += grad[b * output_size_ + j];
        }
    }
    
    // Gradient for weights: input^T × grad
    grad_weights_ = cached_input_.transpose().matmul(grad);
    
    // Gradient for input (to pass to previous layer)
    return grad.matmul(weights_.transpose());
}
```

### Weight Initialization

Poor initialization leads to vanishing/exploding gradients. We use **Xavier/Glorot initialization**:

```cpp
// Uniform distribution in range [-limit, limit]
float limit = sqrt(6.0f / (input_size + output_size));
std::uniform_real_distribution<float> dist(-limit, limit);
```

### The Sequential Container

Stacks multiple layers for easy use:

```cpp
Sequential model;
model.add(std::make_shared<Dense>(2, 8, Activation::ReLU));
model.add(std::make_shared<Dense>(8, 1, Activation::Sigmoid));

// Forward pass through all layers
Tensor output = model.forward(input);

// Backward pass through all layers (in reverse)
model.backward(grad_from_loss);
```

---

## Optimizers

Optimizers update weights to minimize the loss.

### Stochastic Gradient Descent (SGD)

The simplest optimizer:

```
weights = weights - learning_rate × gradient
```

```cpp
void step(std::vector<Tensor*> &params, std::vector<Tensor*> &grads) {
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = 0; j < params[i]->get_size(); ++j) {
            (*params[i])[j] -= lr_ * (*grads[i])[j];
        }
    }
}
```

#### SGD with Momentum

Momentum helps accelerate convergence and escape local minima:

```
velocity = momentum × velocity - learning_rate × gradient
weights = weights + velocity
```

### Adam Optimizer

The most popular optimizer. Combines momentum with adaptive learning rates:

```cpp
// For each parameter:
m = β₁ × m + (1 - β₁) × gradient           // First moment (mean)
v = β₂ × v + (1 - β₂) × gradient²          // Second moment (variance)
m_hat = m / (1 - β₁^t)                     // Bias correction
v_hat = v / (1 - β₂^t)                     // Bias correction
param = param - lr × m_hat / (√v_hat + ε)  // Update
```

Default hyperparameters:
- `learning_rate = 0.001`
- `β₁ = 0.9`
- `β₂ = 0.999`
- `ε = 1e-8`

---

## Putting It All Together

### Training Loop

```cpp
// 1. Create model
Sequential model;
model.add(std::make_shared<Dense>(input_size, hidden_size, Activation::ReLU));
model.add(std::make_shared<Dense>(hidden_size, output_size, Activation::Sigmoid));

// 2. Create loss function and optimizer
Loss::BCELoss loss_fn;
Adam optimizer(0.001f);

// 3. Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Forward pass
    Tensor predictions = model.forward(X);
    
    // Compute loss
    float loss = loss_fn.forward(predictions, Y);
    
    // Backward pass
    Tensor grad = loss_fn.backward();
    model.zero_grad();        // Reset gradients to zero
    model.backward(grad);     // Compute new gradients
    
    // Update weights
    auto params = model.parameters();
    auto grads = model.gradients();
    optimizer.step(params, grads);
}
```

---

## Complete Training Example

### The XOR Problem

XOR is a classic problem that requires non-linearity:

| Input A | Input B | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single perceptron can't solve this, but a 2-layer network can!

### Full Code

```cpp
#include "Dense.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Sequential.h"

int main() {
    // Training data
    Tensor X({4, 2}, {0, 0,  0, 1,  1, 0,  1, 1});
    Tensor Y({4, 1}, {0, 1, 1, 0});
    
    // Model: 2 -> 8 -> 1
    Sequential model;
    model.add(std::make_shared<Dense>(2, 8, Activation::ReLU));
    model.add(std::make_shared<Dense>(8, 1, Activation::Sigmoid));
    
    // Training setup
    Loss::BCELoss loss_fn;
    Adam optimizer(0.1f);
    
    // Train for 1000 epochs
    for (int epoch = 0; epoch < 1000; ++epoch) {
        Tensor pred = model.forward(X);
        float loss = loss_fn.forward(pred, Y);
        
        model.zero_grad();
        model.backward(loss_fn.backward());
        
        auto p = model.parameters();
        auto g = model.gradients();
        optimizer.step(p, g);
    }
    
    // Test
    Tensor result = model.forward(X);
    // result should be close to [0, 1, 1, 0]
}
```

### Why It Works

1. **First layer (2→8)**: Learns to create useful intermediate features
2. **ReLU activation**: Adds non-linearity so the network can learn XOR
3. **Second layer (8→1)**: Combines features to produce final classification
4. **Sigmoid**: Squashes output to (0, 1) for binary classification
5. **BCE Loss**: Appropriate for binary classification
6. **Adam optimizer**: Efficiently updates weights

---

## Project Structure

```
arshian-ML/
├── include/
│   ├── Tensor.h         # Multi-dimensional array
│   ├── Activations.h    # ReLU, Sigmoid, Tanh, Softmax
│   ├── Loss.h           # MSE, BCE, CrossEntropy
│   ├── Layer.h          # Abstract base class
│   ├── Dense.h          # Fully connected layer
│   ├── Sequential.h     # Model container
│   └── Optimizer.h      # SGD, Adam
├── src/
│   └── core/
│       └── Tensor.cpp   # Tensor implementation
├── examples/
│   └── main.cpp         # XOR training demo
└── CMakeLists.txt       # Build configuration
```

---

## Building and Running

```bash
# Configure
cmake -B build -S .

# Build
cmake --build build

# Run example
./build/ml_example.exe
```

---

## Next Steps

Ideas for extending this library:

1. **Convolutional layers** - For image processing
2. **Dropout** - For regularization
3. **Batch normalization** - For faster training
4. **GPU acceleration** - Using CUDA or OpenCL
5. **Automatic differentiation** - Full autograd system
6. **More optimizers** - RMSprop, Adagrad, etc.
7. **Data loading utilities** - For real datasets

---

*This guide covers the fundamentals of neural network implementation. The actual code in this repository is a working implementation of these concepts.*
