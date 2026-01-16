#pragma once

#include "Layer.h"
#include <memory>
#include <vector>

// Sequential container for stacking layers
class Sequential {
public:
  Sequential() = default;

  // Add a layer to the network
  void add(std::shared_ptr<Layer> layer) { layers_.push_back(layer); }

  // Forward pass through all layers
  Tensor forward(const Tensor &input) {
    Tensor output = input;
    for (auto &layer : layers_) {
      output = layer->forward(output);
    }
    return output;
  }

  // Backward pass through all layers (in reverse)
  void backward(const Tensor &grad_output) {
    Tensor grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      grad = (*it)->backward(grad);
    }
  }

  // Get all parameters from all layers
  std::vector<Tensor *> parameters() {
    std::vector<Tensor *> params;
    for (auto &layer : layers_) {
      auto layer_params = layer->parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
  }

  // Get all gradients from all layers
  std::vector<Tensor *> gradients() {
    std::vector<Tensor *> grads;
    for (auto &layer : layers_) {
      auto layer_grads = layer->gradients();
      grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return grads;
  }

  // Zero all gradients
  void zero_grad() {
    for (auto &layer : layers_) {
      for (auto *grad : layer->gradients()) {
        grad->fill(0.0f);
      }
    }
  }

  size_t num_layers() const { return layers_.size(); }

  std::shared_ptr<Layer> operator[](size_t idx) { return layers_[idx]; }

private:
  std::vector<std::shared_ptr<Layer>> layers_;
};
