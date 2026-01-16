#pragma once

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

class Tensor {
public:
  // Constructors
  Tensor(const std::vector<size_t> &shape, float initial_value = 0.0f);
  Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);

  // Getters
  const std::vector<size_t> &get_shape() const;
  const std::vector<float> &get_data() const;
  std::vector<float> &get_data_mut();
  size_t get_size() const;

  // Gradient support
  std::vector<float> &get_grad();
  const std::vector<float> &get_grad() const;
  void zero_grad();
  void set_requires_grad(bool requires);
  bool requires_grad() const;

  // Accessors
  float &operator[](size_t index);
  const float &operator[](size_t index) const;

  // Utilities
  void print() const;
  void fill(float value);
  void random_normal(float mean = 0.0f, float stddev = 1.0f);
  Tensor clone() const;

  // Math Operations
  Tensor add(const Tensor &other) const;
  Tensor subtract(const Tensor &other) const;
  Tensor multiply(const Tensor &other) const; // Element-wise
  Tensor matmul(const Tensor &other) const;
  Tensor transpose() const;

  // Scalar operations
  Tensor scalar_multiply(float scalar) const;
  Tensor scalar_add(float scalar) const;

  // Reduction operations
  float sum() const;
  float mean() const;

  // Element-wise apply
  Tensor apply(std::function<float(float)> func) const;

private:
  std::vector<size_t> shape_;
  std::vector<float> data_;
  std::vector<float> grad_;
  bool requires_grad_ = false;
};
