# SimpleML

![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

**A lightweight neural network library built from scratch in C++**

*No external dependencies. Just pure C++ implementing the fundamentals of deep learning.*

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples)

---

## Features

| Component | Description |
|-----------|-------------|
| **Tensors** | N-dimensional arrays with gradient support |
| **Activations** | ReLU, Sigmoid, Tanh, Softmax |
| **Loss Functions** | MSE, Binary Cross-Entropy, Cross-Entropy |
| **Layers** | Dense (fully connected) with Xavier initialization |
| **Optimizers** | SGD with momentum, Adam |
| **Documentation** | Comprehensive guide included |

## Quick Start

### Prerequisites

- CMake 3.10+
- C++17 compatible compiler (GCC, Clang, MSVC)

### Build

```bash
# Clone the repository
git clone https://github.com/arshian-dev/SimpleML.git
cd SimpleML

# Build
cmake -B build -S .
cmake --build build

# Run example
./build/ml_example
```

## Examples

### XOR Problem

```cpp
#include "Dense.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Sequential.h"

int main() {
    // Training data
    Tensor X({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor Y({4, 1}, {0, 1, 1, 0});

    // Create model: 2 → 8 → 1
    Sequential model;
    model.add(std::make_shared<Dense>(2, 8, Activation::ReLU));
    model.add(std::make_shared<Dense>(8, 1, Activation::Sigmoid));

    // Training setup
    Loss::BCELoss loss_fn;
    Adam optimizer(0.1f);

    // Train
    for (int epoch = 0; epoch < 1000; ++epoch) {
        Tensor pred = model.forward(X);
        float loss = loss_fn.forward(pred, Y);
        
        model.zero_grad();
        model.backward(loss_fn.backward());
        
        auto p = model.parameters();
        auto g = model.gradients();
        optimizer.step(p, g);
    }

    // Predict
    Tensor result = model.forward(X);
    // Output: [0.001, 0.999, 0.999, 0.001]
}
```

### Creating Tensors

```cpp
// Create a 2x3 matrix of zeros
Tensor t1({2, 3}, 0.0f);

// Create from data
Tensor t2({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

// Random initialization
Tensor t3({100, 50});
t3.random_normal(0.0f, 1.0f);

// Matrix operations
Tensor result = t1.matmul(t2.transpose());
```

### Building Networks

```cpp
Sequential model;

// Add layers
model.add(std::make_shared<Dense>(784, 256, Activation::ReLU));
model.add(std::make_shared<Dense>(256, 128, Activation::ReLU));
model.add(std::make_shared<Dense>(128, 10, Activation::None));

// Forward pass
Tensor output = model.forward(input);

// Backward pass
model.backward(loss_gradient);

// Get parameters for optimization
auto params = model.parameters();
auto grads = model.gradients();
```

## Project Structure

```
SimpleML/
├── include/
│   ├── Tensor.h         # Multi-dimensional arrays
│   ├── Activations.h    # Activation functions
│   ├── Loss.h           # Loss functions
│   ├── Layer.h          # Base layer class
│   ├── Dense.h          # Fully connected layer
│   ├── Sequential.h     # Model container
│   └── Optimizer.h      # SGD & Adam
├── src/
│   └── core/
│       └── Tensor.cpp   # Tensor implementation
├── examples/
│   └── main.cpp         # XOR training demo
├── GUIDE.md             # Detailed tutorial
├── REPORT.md            # Technical documentation
└── CMakeLists.txt       # Build configuration
```

## Documentation

For a comprehensive guide covering:

- Neural network fundamentals
- Tensor operations and matrix math
- Activation functions with derivatives
- Backpropagation explained
- Optimizer algorithms

See **[GUIDE.md](GUIDE.md)** or the **[PDF Report](SimpleML_Neural_Network_Report.pdf)**

## API Reference

### Tensor

```cpp
// Construction
Tensor(shape, initial_value)
Tensor(shape, data_vector)

// Operations
add(other), subtract(other), multiply(other)
matmul(other), transpose()
scalar_multiply(s), scalar_add(s)
sum(), mean()
apply(function)

// Gradients
zero_grad(), get_grad(), requires_grad()
```

### Layers

```cpp
// Dense layer
Dense(input_size, output_size, activation)
// Activations: None, ReLU, Sigmoid, Tanh

// Methods
forward(input) → Tensor
backward(grad_output) → Tensor
parameters() → vector<Tensor*>
gradients() → vector<Tensor*>
```

### Optimizers

```cpp
// SGD
SGD(learning_rate, momentum = 0.0f)

// Adam
Adam(lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f)

// Methods
step(params, grads)
zero_grad(grads)
```

### Loss Functions

```cpp
Loss::MSELoss      // Regression
Loss::BCELoss      // Binary classification
Loss::CrossEntropyLoss  // Multi-class

// Methods
forward(predictions, targets) → float
backward() → Tensor
```

## Roadmap

- [ ] Convolutional layers (Conv2D)
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Dropout regularization
- [ ] Batch normalization
- [ ] Model save/load
- [ ] GPU acceleration (CUDA)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built for learning. Inspired by PyTorch and the desire to understand deep learning from first principles.
