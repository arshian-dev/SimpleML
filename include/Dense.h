#pragma once

#include "Activations.h"
#include "Layer.h"
#include "Tensor.h"
#include <cmath>
#include <random>

enum class Activation { None, ReLU, Sigmoid, Tanh };

// Fully connected (Dense) layer
class Dense : public Layer {
public:
  Dense(size_t input_size, size_t output_size,
        Activation activation = Activation::None)
      : input_size_(input_size), output_size_(output_size),
        activation_(activation), weights_({input_size, output_size}),
        bias_({1, output_size}), grad_weights_({input_size, output_size}),
        grad_bias_({1, output_size}), cached_input_({}),
        cached_pre_activation_({}), cached_output_({}) {
    // Xavier/Glorot initialization
    float limit =
        std::sqrt(6.0f / static_cast<float>(input_size + output_size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < weights_.get_size(); ++i) {
      weights_[i] = dist(gen);
    }
    // Bias initialized to zero (already default)
  }

  Tensor forward(const Tensor &input) override {
    // Input shape: (batch_size, input_size)
    if (input.get_shape().size() != 2 || input.get_shape()[1] != input_size_) {
      throw std::invalid_argument("Dense layer: invalid input shape");
    }

    cached_input_ = input;

    // output = input @ weights + bias
    Tensor output = input.matmul(weights_);

    // Add bias (broadcast across batch)
    size_t batch_size = output.get_shape()[0];
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t j = 0; j < output_size_; ++j) {
        output[b * output_size_ + j] += bias_[j];
      }
    }

    cached_pre_activation_ = output;

    // Apply activation
    switch (activation_) {
    case Activation::ReLU:
      output = Activations::relu(output);
      break;
    case Activation::Sigmoid:
      output = Activations::sigmoid(output);
      break;
    case Activation::Tanh:
      output = Activations::tanh(output);
      break;
    case Activation::None:
    default:
      break;
    }

    cached_output_ = output;
    return output;
  }

  Tensor backward(const Tensor &grad_output) override {
    Tensor grad = grad_output;

    // Backprop through activation
    switch (activation_) {
    case Activation::ReLU:
      grad = Activations::relu_backward(cached_pre_activation_, grad);
      break;
    case Activation::Sigmoid:
      grad = Activations::sigmoid_backward(cached_output_, grad);
      break;
    case Activation::Tanh:
      grad = Activations::tanh_backward(cached_output_, grad);
      break;
    case Activation::None:
    default:
      break;
    }

    size_t batch_size = cached_input_.get_shape()[0];

    // Gradient w.r.t. bias: sum over batch
    // grad_bias = sum(grad, axis=0)
    grad_bias_.fill(0.0f);
    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t j = 0; j < output_size_; ++j) {
        grad_bias_[j] += grad[b * output_size_ + j];
      }
    }

    // Gradient w.r.t. weights: input.T @ grad
    // grad_weights = cached_input.T @ grad
    Tensor input_T = cached_input_.transpose();
    grad_weights_ = input_T.matmul(grad);

    // Gradient w.r.t. input: grad @ weights.T
    Tensor weights_T = weights_.transpose();
    Tensor grad_input = grad.matmul(weights_T);

    return grad_input;
  }

  std::vector<Tensor *> parameters() override { return {&weights_, &bias_}; }

  std::vector<Tensor *> gradients() override {
    return {&grad_weights_, &grad_bias_};
  }

  const char *name() const override { return "Dense"; }

  size_t input_size() const { return input_size_; }
  size_t output_size() const { return output_size_; }

private:
  size_t input_size_;
  size_t output_size_;
  Activation activation_;

  Tensor weights_;
  Tensor bias_;
  Tensor grad_weights_;
  Tensor grad_bias_;

  // Cached values for backward pass
  Tensor cached_input_;
  Tensor cached_pre_activation_;
  Tensor cached_output_;
};
