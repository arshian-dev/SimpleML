#pragma once

#include "Tensor.h"
#include <vector>

// Base class for all neural network layers
class Layer {
public:
  virtual ~Layer() = default;

  // Forward pass: compute output from input
  virtual Tensor forward(const Tensor &input) = 0;

  // Backward pass: compute gradients
  virtual Tensor backward(const Tensor &grad_output) = 0;

  // Get all trainable parameters
  virtual std::vector<Tensor *> parameters() = 0;

  // Get gradients for all parameters
  virtual std::vector<Tensor *> gradients() = 0;

  // Get layer name for debugging
  virtual const char *name() const = 0;
};
