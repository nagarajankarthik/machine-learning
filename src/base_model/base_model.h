#include "../utils/json.hpp"
#include "../utils/logging.h"
#include "../utils/utils.h"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#pragma once

using namespace std;

namespace ml {

struct DoubleHash {
  size_t operator()(double key) const {
    return std::hash<int64_t>()(
        static_cast<int64_t>(key * 1000000)); // Scale and hash
  }
};

struct DoubleEqual {
  bool operator()(double a, double b) const { return std::fabs(a - b) < 1e-6; }
};

class BaseModel {

public:
  /**
   * random seed
   */
  int random_seed = 0;

  /**
   * Random number engine
   */
  std::mt19937 random_generator;

  /**
   * Logger
   */
  shared_ptr<Logger> logger;

  /**
   * Constructor for base model.
   */
  BaseModel(nlohmann::json model_parameters, shared_ptr<Logger> logger);

  /**
   * Destructor for base model
   */
  virtual ~BaseModel() {};

  /**
   * Prototype for data initialization
   */
  virtual void set_data(TrainTestData &&train_test) = 0;

  /**
   * Prototype for model training
   */
  virtual void fit() = 0;

  /**
   * Prototype for model inference
   */
  // virtual vector<vector<double>>
  // predict(const vector<vector<double>> &test_features) = 0;

  /**
   * Convert elements of 2D array from type double to int.
   */
  vector<vector<int>> double_to_int(const vector<vector<double>> &data);

  /**
   * Get unique entries in each column of an input 2D vector containing
   * integers.
   */
  vector<unordered_set<int>> get_unique_classes(vector<vector<int>> outputs);

  /**
   * Get confusion matrix for a single output variable
   */
  vector<vector<int>>
  get_confusion_matrix_single(const vector<vector<int>> &predictions,
                              const vector<vector<int>> &test_outputs,
                              unordered_set<int> unique_classes,
                              int index_output);

  /**
   * Convert 2D array to string for logging purposes
   */
  template <class T> string array_2d_to_string(vector<vector<T>> matrix);

  /**
   * Get confusion matrix for each output variable.
   */
  void
  get_confusion_matrices(const vector<vector<double>> &test_predictions_double,
                         const vector<vector<double>> &test_labels_double,
                         const vector<vector<double>> &train_labels_double);

  /**
   * Get root mean square error for each output variable.
   */
  void get_root_mean_square_error(const vector<vector<int>> &test_predictions,
                                  const vector<vector<int>> &test_labels);

  /**
   * Prototype for model evaluation
   */
  virtual void evaluate() = 0;
};
} // namespace ml
