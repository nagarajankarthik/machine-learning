#include "../base_model/base_model.h"
#include "layer.h"
#include "optimizer.h"

#include <memory>
#include <mpi.h>
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
  vector<shared_ptr<Tensor>> train_inputs{};

  /**
   * Array of ground truth label tensors
   */
  vector<shared_ptr<Tensor>> train_labels{};

  /**
   * Array of input tensors
   */
  vector<shared_ptr<Tensor>> inference_inputs{};

  /**
   * Array of ground truth label tensors
   */
  vector<shared_ptr<Tensor>> inference_labels{};

  /**
   * Maximum number of batches per epoch. Used for early stopping
   */
  int max_batches_per_epoch = 1;

  /**
   * Number of epochs
   */
  int number_epochs = 1;

  /**
   * Current epoch
   */
  int current_epoch = 1;

  /**
   * Batch size
   */
  int micro_batch_size = 1;

  /**
   * Global MPI rank
   */
  int global_rank = 0;

  /**
   * Number of MPI processes
   */
  int world_size = 1;

  /**
   * Gradient buffer used for MPI communication
   */
  vector<double> gradient_buffer{};

  /**
   * Optimizer
   */
  shared_ptr<Optimizer> optimizer = nullptr;

  /**
   * An array of Layer objects
   */
  vector<shared_ptr<Layer>> layers{};

  /**
   * Actual categories for validation data
   */
  vector<vector<double>> inference_categories{};

  /**
   * Loss functions
   */
  unordered_map<string, function<shared_ptr<Tensor>(shared_ptr<Tensor>,
                                                    shared_ptr<Tensor>)>>
      _loss_functions = {{"cross_entropy",
                          [](shared_ptr<Tensor> x, shared_ptr<Tensor> y) {
                            return categorical_cross_entropy_forward(x, y);
                          }},
                         {"mean_squared_error",
                          [](shared_ptr<Tensor> x, shared_ptr<Tensor> y) {
                            return mean_squared_error_forward(x, y);
                          }}};

  /**
   * Loss function
   **/
  function<shared_ptr<Tensor>(shared_ptr<Tensor>, shared_ptr<Tensor>)>
      loss_function;

  /**
   * Constructor
   */
  NeuralNetwork(nlohmann::json parameters, shared_ptr<Logger> logger);

  /**
   * Destructor
   */
  ~NeuralNetwork() {};

  /**
   * Get actual categories from one hot encoded labels
   */
  void
  prepare_inference_categories(const vector<vector<double>> &validation_labels);

  /**
   * Prepare input tensors for training and inference
   */
  void prepare_inputs_labels(const vector<vector<double>> &features,
                             const vector<vector<double>> &labels,
                             vector<shared_ptr<Tensor>> &prepared_inputs,
                             vector<shared_ptr<Tensor>> &prepared_labels);
  /**
   * Prepare input tensors for training
   */
  [[deprecated]]
  void prepare_train_input(const vector<vector<double>> &features,
                           const vector<vector<double>> &input_labels);

  /**
   * Prepare input tensors for training
   */
  [[deprecated]]
  void prepare_inference_input(const vector<vector<double>> &features,
                               const vector<vector<double>> &labels);
  /**
   * Set training and test data.
   */
  void set_data(TrainTestData &&train_test);

  /**
   * Perform model training.
   */
  void fit();

  /**
   * Calculate validation loss.
   */
  shared_ptr<Tensor> validate();

  /**
   * Perform a single training epoch
   */
  void train_epoch();

  /**
   * Convert predictions and labels from tensors to array
   */
  vector<vector<double>> get_categories(shared_ptr<Tensor> tensor);

  /**
   * Perform model inference
   */
  vector<vector<double>> predict();

  /**
   * Evaluate model using test data
   */
  void evaluate();

  /**
   * Collect gradients from all layers before communication
   */
  void collect_gradients();

  /**
   * Update gradients in each layer after MPI Allreduce operation
   */
  void update_gradients();

  /**
   * Communicate gradients between processes
   */
  void communicate_gradients();
};

} // namespace ml
