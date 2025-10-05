#include "../tensor/tensor.h"
#include "../tensor/tensor_operations.h"
#include "../utils/logging.h"
#include <cassert>
#include <random>
#include <variant>
using namespace std;

namespace ml {

/**
 * Abstract class defining template for neural network layers
 */
class Optimizer {
public:
  // Current optimizer step
  int current_step = 0;
  // List of quantities to optimize. Each quantity must be a Tensor object.
  vector<shared_ptr<Tensor>> optimize_parameters = {};

  // Logger
  shared_ptr<Logger> logger = nullptr;

  Optimizer() = default;

  Optimizer(shared_ptr<Logger> logger) : logger(logger) {}

  Optimizer(vector<shared_ptr<Tensor>> optimize_parameters,
            shared_ptr<Logger> logger)
      : optimize_parameters(optimize_parameters), logger(logger) {}

  virtual void step() = 0;

  // virtual ~Optimizer() {}

  void zero_gradients() {
    for (shared_ptr<Tensor> parameter : optimize_parameters) {
      fill(parameter->gradients.begin(), parameter->gradients.end(), 0.0);
    }
  }
};

/**
 * Stochastic gradient descent optimizer
 */
class SGDOptimizer : public Optimizer {
public:
  // Learning rate
  double learning_rate = 0.01;

  // Momentum
  double momentum = 0.9;

  SGDOptimizer(vector<shared_ptr<Tensor>> optimize_parameters,
               shared_ptr<Logger> logger, double learning_rate = 0.01,
               double momentum = 0.9)
      : Optimizer(optimize_parameters, logger), learning_rate(learning_rate),
        momentum(momentum) {}

  /**
   * Function to update parameters
   * @param input: Input tensor to layer
   */
  void step() override {
    for (shared_ptr<Tensor> parameter : optimize_parameters) {
      for (int i = 0; i < parameter->values.size(); i++) {
        if (momentum > std::numeric_limits<double>::epsilon()) {
          if (current_step > 0)
            parameter->velocities[i] =
                momentum * parameter->velocities[i] + parameter->gradients[i];
          else
            parameter->velocities[i] = parameter->gradients[i];
          // Ignore nesterov method for now
          parameter->gradients[i] = parameter->velocities[i];
        }
        parameter->values[i] -= learning_rate * parameter->gradients[i];
      }
    }
  }
};
} // namespace ml
