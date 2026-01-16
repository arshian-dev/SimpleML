#include "Tensor.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

Tensor::Tensor(const std::vector<size_t> &shape, float initial_value)
    : shape_(shape) {
  size_t total_size = 1;
  for (size_t dim : shape_) {
    total_size *= dim;
  }
  data_.resize(total_size, initial_value);
  grad_.resize(total_size, 0.0f);
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data)
    : shape_(shape), data_(data) {
  size_t total_size = 1;
  for (size_t dim : shape_) {
    total_size *= dim;
  }
  if (data_.size() != total_size) {
    throw std::invalid_argument("Data size doesn't match shape");
  }
  grad_.resize(total_size, 0.0f);
}

const std::vector<size_t> &Tensor::get_shape() const { return shape_; }

const std::vector<float> &Tensor::get_data() const { return data_; }

std::vector<float> &Tensor::get_data_mut() { return data_; }

size_t Tensor::get_size() const { return data_.size(); }

// Gradient support
std::vector<float> &Tensor::get_grad() { return grad_; }

const std::vector<float> &Tensor::get_grad() const { return grad_; }

void Tensor::zero_grad() { std::fill(grad_.begin(), grad_.end(), 0.0f); }

void Tensor::set_requires_grad(bool requires) {
  requires_grad_ =
    requires;
}

bool Tensor::requires_grad() const { return requires_grad_; }

float &Tensor::operator[](size_t index) { return data_[index]; }

const float &Tensor::operator[](size_t index) const { return data_[index]; }

void Tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

void Tensor::random_normal(float mean, float stddev) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> d(mean, stddev);

  for (auto &val : data_) {
    val = d(gen);
  }
}

Tensor Tensor::clone() const {
  Tensor result(shape_, data_);
  result.grad_ = grad_;
  result.requires_grad_ = requires_grad_;
  return result;
}

void Tensor::print() const {
  std::cout << "Tensor shape: [";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i] << (i < shape_.size() - 1 ? ", " : "");
  }
  std::cout << "]" << std::endl;

  if (data_.size() < 100) {
    std::cout << "[";
    for (size_t i = 0; i < data_.size(); ++i) {
      std::cout << data_[i] << (i < data_.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
  } else {
    std::cout << "[Data too large to print fully]" << std::endl;
  }
}

Tensor Tensor::add(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shapes must match for addition");
  }
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }
  return result;
}

Tensor Tensor::subtract(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shapes must match for subtraction");
  }
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] - other.data_[i];
  }
  return result;
}

Tensor Tensor::multiply(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Shapes must match for element-wise multiplication");
  }
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] * other.data_[i];
  }
  return result;
}

Tensor Tensor::matmul(const Tensor &other) const {
  // Only support 2D matrices for now
  if (shape_.size() != 2 || other.shape_.size() != 2) {
    throw std::invalid_argument("Matmul only supports 2D tensors for now");
  }
  if (shape_[1] != other.shape_[0]) {
    throw std::invalid_argument(
        "Shapes incompatible for matmul: (" + std::to_string(shape_[0]) + "," +
        std::to_string(shape_[1]) + ") vs (" + std::to_string(other.shape_[0]) +
        "," + std::to_string(other.shape_[1]) + ")");
  }

  size_t M = shape_[0];
  size_t K = shape_[1];
  size_t N = other.shape_[1];

  Tensor result({M, N});

  // Naive O(N^3) multiplication
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

Tensor Tensor::transpose() const {
  if (shape_.size() != 2) {
    throw std::invalid_argument("Transpose only supports 2D tensors for now");
  }
  size_t rows = shape_[0];
  size_t cols = shape_[1];

  Tensor result({cols, rows});
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result.data_[j * rows + i] = data_[i * cols + j];
    }
  }
  return result;
}

// Scalar operations
Tensor Tensor::scalar_multiply(float scalar) const {
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] * scalar;
  }
  return result;
}

Tensor Tensor::scalar_add(float scalar) const {
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] + scalar;
  }
  return result;
}

// Reduction operations
float Tensor::sum() const {
  float total = 0.0f;
  for (const auto &val : data_) {
    total += val;
  }
  return total;
}

float Tensor::mean() const { return sum() / static_cast<float>(data_.size()); }

// Element-wise apply
Tensor Tensor::apply(std::function<float(float)> func) const {
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = func(data_[i]);
  }
  return result;
}
