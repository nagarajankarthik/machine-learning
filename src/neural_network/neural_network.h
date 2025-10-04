#include "../base_model/base_model.h"
#include "layer.h"
#include "optimizer.h"

#include <memory>
#include <omp.h>
#include <random>
#include <vector>

using namespace std;

namespace ml {

class NeuralNetwork : public BaseModel {
public:
  /**
   * Shape of input tensor for a single mini-batch. First element should be
   * equal to batch size.
   */
  vector<int> input_shape{};

  /**
   * Shape of ground truth labels for a single mini-batch. First element should
   * be equal to batch size.
   */
  vector<int> labels_shape{};

  /**
   * Array of input tensors
   */
  vector<shared_ptr<Tensor>> inputs{};

  /**
   * Array of ground truth label tensors
   */
  vector<shared_ptr<Tensor>> labels{};

  /**
   * Batch size
   */
  int batch_size = 1;

  /**
   * Optimizer
   */
  shared_ptr<Optimizer> optimizer = nullptr;

  /**
   * An array of Layer objects
   */
  vector<shared_ptr<Layer>> layers{};

  /**
   * Constructor
   */
  NeuralNetwork(nlohmann::json parameters, shared_ptr<Logger> logger);

  /**
   * Destructor
   */
  ~NeuralNetwork() {};

  /**
   * Prepare input tensors for training
   */
  void prepare_input(const vector<vector<double>> &features,
                     const vector<vector<double>> &labels);

  /**
   * Perform model training.
   */
  void fit(const vector<vector<double>> &&features,
           const vector<vector<double>> &&labels);
  /**
   * Perform a single training epoch
   */
  void train_step();

  /**
   * Perform model inference
   */
  vector<vector<double>> predict(const vector<vector<double>> &test_features);

  /**
   * Evaluate model using test data
   */
  void evaluate(const vector<vector<double>> &test_features,
                const vector<vector<double>> &test_labels);
};

} // namespace ml
