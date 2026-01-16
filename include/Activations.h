#pragma once

#include "Tensor.h"
#include <cmath>

namespace Activations {

// ReLU activation
inline Tensor relu(const Tensor &input) {
  return input.apply([](float x) { return x > 0.0f ? x : 0.0f; });
}

inline Tensor relu_backward(const Tensor &input, const Tensor &grad_output) {
  Tensor result(input.get_shape());
  for (size_t i = 0; i < input.get_size(); ++i) {
    result[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
  }
  return result;
}

// Sigmoid activation
inline Tensor sigmoid(const Tensor &input) {
  return input.apply([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

inline Tensor sigmoid_backward(const Tensor &output,
                               const Tensor &grad_output) {
  // output is sigmoid(input), derivative is output * (1 - output)
  Tensor result(output.get_shape());
  for (size_t i = 0; i < output.get_size(); ++i) {
    float s = output[i];
    result[i] = grad_output[i] * s * (1.0f - s);
  }
  return result;
}

// Tanh activation
inline Tensor tanh(const Tensor &input) {
  return input.apply([](float x) { return std::tanh(x); });
}

inline Tensor tanh_backward(const Tensor &output, const Tensor &grad_output) {
  // derivative is 1 - output^2
  Tensor result(output.get_shape());
  for (size_t i = 0; i < output.get_size(); ++i) {
    float t = output[i];
    result[i] = grad_output[i] * (1.0f - t * t);
  }
  return result;
}

// Softmax (along last axis for 2D tensors)
inline Tensor softmax(const Tensor &input) {
  const auto &shape = input.get_shape();
  if (shape.size() != 2) {
    throw std::invalid_argument("Softmax only supports 2D tensors");
  }

  size_t batch_size = shape[0];
  size_t num_classes = shape[1];
  Tensor result(shape);

  for (size_t b = 0; b < batch_size; ++b) {
    // Find max for numerical stability
    float max_val = input[b * num_classes];
    for (size_t c = 1; c < num_classes; ++c) {
      max_val = std::max(max_val, input[b * num_classes + c]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (size_t c = 0; c < num_classes; ++c) {
      result[b * num_classes + c] =
          std::exp(input[b * num_classes + c] - max_val);
      sum_exp += result[b * num_classes + c];
    }

    // Normalize
    for (size_t c = 0; c < num_classes; ++c) {
      result[b * num_classes + c] /= sum_exp;
    }
  }

  return result;
}

} // namespace Activations
