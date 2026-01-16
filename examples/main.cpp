#include "Dense.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Sequential.h"
#include "Tensor.h"
#include <iomanip>
#include <iostream>
#include <memory>

int main() {
  std::cout << "=== SimpleML Neural Network Demo ===" << std::endl;
  std::cout << "Training a neural network to learn XOR function\n" << std::endl;

  // XOR training data
  // Inputs: 4 samples, 2 features each
  Tensor X({4, 2}, {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f});

  // Targets: XOR outputs (0, 1, 1, 0)
  Tensor Y({4, 1}, {0.0f, 1.0f, 1.0f, 0.0f});

  std::cout << "Training Data:" << std::endl;
  std::cout << "  [0, 0] -> 0" << std::endl;
  std::cout << "  [0, 1] -> 1" << std::endl;
  std::cout << "  [1, 0] -> 1" << std::endl;
  std::cout << "  [1, 1] -> 0\n" << std::endl;

  // Create a simple MLP: 2 -> 8 -> 1
  Sequential model;
  model.add(std::make_shared<Dense>(
      2, 8, Activation::ReLU)); // Hidden layer with ReLU
  model.add(std::make_shared<Dense>(8, 1, Activation::Sigmoid)); // Output layer

  std::cout << "Model Architecture:" << std::endl;
  std::cout << "  Input:  2 features" << std::endl;
  std::cout << "  Hidden: 8 neurons (ReLU)" << std::endl;
  std::cout << "  Output: 1 neuron (Sigmoid)\n" << std::endl;

  // Loss function and optimizer
  Loss::BCELoss loss_fn;
  Adam optimizer(0.1f); // Learning rate 0.1

  // Training loop
  const int epochs = 1000;
  const int print_every = 100;

  std::cout << "Training for " << epochs << " epochs..." << std::endl;
  std::cout << std::string(40, '-') << std::endl;

  for (int epoch = 1; epoch <= epochs; ++epoch) {
    // Forward pass
    Tensor predictions = model.forward(X);

    // Compute loss
    float loss = loss_fn.forward(predictions, Y);

    // Backward pass
    Tensor grad = loss_fn.backward();
    model.zero_grad();
    model.backward(grad);

    // Update weights
    auto params = model.parameters();
    auto grads = model.gradients();
    optimizer.step(params, grads);

    // Print progress
    if (epoch % print_every == 0 || epoch == 1) {
      std::cout << "Epoch " << std::setw(4) << epoch
                << " | Loss: " << std::fixed << std::setprecision(6) << loss
                << std::endl;
    }
  }

  std::cout << std::string(40, '-') << std::endl;

  // Test the trained model
  std::cout << "\nFinal Predictions:" << std::endl;
  Tensor final_predictions = model.forward(X);

  const char *inputs[] = {"[0, 0]", "[0, 1]", "[1, 0]", "[1, 1]"};
  float targets[] = {0.0f, 1.0f, 1.0f, 0.0f};

  bool all_correct = true;
  for (int i = 0; i < 4; ++i) {
    float pred = final_predictions[i];
    float expected = targets[i];
    bool correct = (pred > 0.5f) == (expected > 0.5f);
    all_correct = all_correct && correct;

    std::cout << "  " << inputs[i] << " -> " << std::fixed
              << std::setprecision(4) << pred << " (expected: " << expected
              << ") " << (correct ? "✓" : "✗") << std::endl;
  }

  std::cout << "\n=== " << (all_correct ? "SUCCESS" : "NEEDS MORE TRAINING")
            << " ===" << std::endl;

  return all_correct ? 0 : 1;
}
