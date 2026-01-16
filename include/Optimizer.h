#pragma once

#include "Tensor.h"
#include <cmath>
#include <vector>

// Base optimizer class
class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step(std::vector<Tensor *> &params,
                    std::vector<Tensor *> &grads) = 0;
  virtual void zero_grad(std::vector<Tensor *> &grads) {
    for (auto *grad : grads) {
      grad->fill(0.0f);
    }
  }
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
public:
  explicit SGD(float learning_rate, float momentum = 0.0f)
      : lr_(learning_rate), momentum_(momentum), initialized_(false) {}

  void step(std::vector<Tensor *> &params,
            std::vector<Tensor *> &grads) override {
    if (momentum_ > 0.0f && !initialized_) {
      // Initialize velocity tensors
      velocities_.clear();
      for (auto *param : params) {
        velocities_.push_back(Tensor(param->get_shape(), 0.0f));
      }
      initialized_ = true;
    }

    for (size_t i = 0; i < params.size(); ++i) {
      Tensor *param = params[i];
      Tensor *grad = grads[i];

      if (momentum_ > 0.0f) {
        // v = momentum * v - lr * grad
        // param = param + v
        for (size_t j = 0; j < param->get_size(); ++j) {
          velocities_[i][j] = momentum_ * velocities_[i][j] - lr_ * (*grad)[j];
          (*param)[j] += velocities_[i][j];
        }
      } else {
        // Simple SGD: param = param - lr * grad
        for (size_t j = 0; j < param->get_size(); ++j) {
          (*param)[j] -= lr_ * (*grad)[j];
        }
      }
    }
  }

private:
  float lr_;
  float momentum_;
  bool initialized_;
  std::vector<Tensor> velocities_;
};

// Adam optimizer
class Adam : public Optimizer {
public:
  explicit Adam(float learning_rate = 0.001f, float beta1 = 0.9f,
                float beta2 = 0.999f, float epsilon = 1e-8f)
      : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
        t_(0), initialized_(false) {}

  void step(std::vector<Tensor *> &params,
            std::vector<Tensor *> &grads) override {
    if (!initialized_) {
      // Initialize moment tensors
      m_.clear();
      v_.clear();
      for (auto *param : params) {
        m_.push_back(Tensor(param->get_shape(), 0.0f));
        v_.push_back(Tensor(param->get_shape(), 0.0f));
      }
      initialized_ = true;
    }

    t_++;

    float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params.size(); ++i) {
      Tensor *param = params[i];
      Tensor *grad = grads[i];

      for (size_t j = 0; j < param->get_size(); ++j) {
        float g = (*grad)[j];

        // Update biased first moment estimate
        m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * g;

        // Update biased second raw moment estimate
        v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * g * g;

        // Bias-corrected estimates
        float m_hat = m_[i][j] / bias_correction1;
        float v_hat = v_[i][j] / bias_correction2;

        // Update parameters
        (*param)[j] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
      }
    }
  }

private:
  float lr_;
  float beta1_;
  float beta2_;
  float epsilon_;
  int t_;
  bool initialized_;
  std::vector<Tensor> m_; // First moment
  std::vector<Tensor> v_; // Second moment
};
