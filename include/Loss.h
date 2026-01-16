#pragma once

#include "Tensor.h"
#include <cmath>
#include <stdexcept>

namespace Loss {

// Mean Squared Error Loss
class MSELoss {
public:
  // Forward: loss = mean((predictions - targets)^2)
  float forward(const Tensor &predictions, const Tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
      throw std::invalid_argument("Shapes must match for MSE loss");
    }

    cached_predictions_ = predictions;
    cached_targets_ = targets;

    float sum = 0.0f;
    for (size_t i = 0; i < predictions.get_size(); ++i) {
      float diff = predictions[i] - targets[i];
      sum += diff * diff;
    }
    return sum / static_cast<float>(predictions.get_size());
  }

  // Backward: d_loss/d_predictions = 2 * (predictions - targets) / n
  Tensor backward() {
    Tensor grad(cached_predictions_.get_shape());
    float n = static_cast<float>(cached_predictions_.get_size());

    for (size_t i = 0; i < cached_predictions_.get_size(); ++i) {
      grad[i] = 2.0f * (cached_predictions_[i] - cached_targets_[i]) / n;
    }
    return grad;
  }

private:
  Tensor cached_predictions_{{}};
  Tensor cached_targets_{{}};
};

// Binary Cross Entropy Loss
class BCELoss {
public:
  // Forward: loss = -mean(targets * log(predictions) + (1 - targets) * log(1 -
  // predictions))
  float forward(const Tensor &predictions, const Tensor &targets) {
    if (predictions.get_shape() != targets.get_shape()) {
      throw std::invalid_argument("Shapes must match for BCE loss");
    }

    cached_predictions_ = predictions;
    cached_targets_ = targets;

    float sum = 0.0f;
    const float eps = 1e-7f; // For numerical stability

    for (size_t i = 0; i < predictions.get_size(); ++i) {
      float p = std::max(eps, std::min(1.0f - eps, predictions[i]));
      float t = targets[i];
      sum += -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
    }
    return sum / static_cast<float>(predictions.get_size());
  }

  // Backward: d_loss/d_predictions = -(targets/predictions -
  // (1-targets)/(1-predictions)) / n
  Tensor backward() {
    Tensor grad(cached_predictions_.get_shape());
    float n = static_cast<float>(cached_predictions_.get_size());
    const float eps = 1e-7f;

    for (size_t i = 0; i < cached_predictions_.get_size(); ++i) {
      float p = std::max(eps, std::min(1.0f - eps, cached_predictions_[i]));
      float t = cached_targets_[i];
      grad[i] = (-(t / p) + (1.0f - t) / (1.0f - p)) / n;
    }
    return grad;
  }

private:
  Tensor cached_predictions_{{}};
  Tensor cached_targets_{{}};
};

// Cross Entropy Loss (with softmax built-in for numerical stability)
class CrossEntropyLoss {
public:
  // Forward: expects raw logits, applies softmax internally
  // targets should be one-hot encoded
  float forward(const Tensor &logits, const Tensor &targets) {
    if (logits.get_shape() != targets.get_shape()) {
      throw std::invalid_argument("Shapes must match for CrossEntropy loss");
    }
    if (logits.get_shape().size() != 2) {
      throw std::invalid_argument(
          "CrossEntropy expects 2D tensors (batch x classes)");
    }

    cached_logits_ = logits;
    cached_targets_ = targets;

    // Apply softmax
    const auto &shape = logits.get_shape();
    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    cached_softmax_ = Tensor(shape);
    float loss = 0.0f;
    const float eps = 1e-7f;

    for (size_t b = 0; b < batch_size; ++b) {
      // Find max for numerical stability
      float max_val = logits[b * num_classes];
      for (size_t c = 1; c < num_classes; ++c) {
        max_val = std::max(max_val, logits[b * num_classes + c]);
      }

      // Compute exp and sum
      float sum_exp = 0.0f;
      for (size_t c = 0; c < num_classes; ++c) {
        cached_softmax_[b * num_classes + c] =
            std::exp(logits[b * num_classes + c] - max_val);
        sum_exp += cached_softmax_[b * num_classes + c];
      }

      // Normalize and compute loss
      for (size_t c = 0; c < num_classes; ++c) {
        cached_softmax_[b * num_classes + c] /= sum_exp;
        float p = std::max(eps, cached_softmax_[b * num_classes + c]);
        loss -= targets[b * num_classes + c] * std::log(p);
      }
    }

    return loss / static_cast<float>(batch_size);
  }

  // Backward: gradient is (softmax - targets) / batch_size
  Tensor backward() {
    const auto &shape = cached_logits_.get_shape();
    size_t batch_size = shape[0];

    Tensor grad(shape);
    for (size_t i = 0; i < cached_logits_.get_size(); ++i) {
      grad[i] = (cached_softmax_[i] - cached_targets_[i]) /
                static_cast<float>(batch_size);
    }
    return grad;
  }

private:
  Tensor cached_logits_{{}};
  Tensor cached_targets_{{}};
  Tensor cached_softmax_{{}};
};

} // namespace Loss
