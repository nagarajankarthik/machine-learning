/*
 * header file for random forest class
 */
#include "decision_tree_classifier.h"
#include <omp.h>
#include <random>

using namespace std;

namespace ml {

class RandomForestClassifier : public BaseModel {
public:
  /**
   * Parameters to pass to Decision Tree constructor
   */
  nlohmann::json parameters;

  // Number of trees
  int number_trees = 20;

  /**
   * An array of decision trees
   */
  vector<shared_ptr<DecisionTreeClassifier>> trees{};

  // Train and test data

  /**
   * Training features
   */
  vector<vector<double>> train_features{};

  /**
   * Training labels
   */
  vector<vector<double>> train_labels{};

  /**
   * Test features
   */
  vector<vector<double>> test_features{};

  /**
   * Test labels
   */
  vector<vector<double>> test_labels{};

  /**
   * Constructor
   */
  RandomForestClassifier(nlohmann::json parameters, shared_ptr<Logger> logger);

  /**
   * Destructor
   */
  ~RandomForestClassifier() {};

  /**
   * Initialize data
   */
  void set_data(TrainTestData &&train_test);

  /**
   * Get bootstrap sample
   */
  void get_bootstrap_sample(vector<vector<double>> &features_sample,
                            vector<vector<double>> &outputs_sample);

  /**
   * Perform model training.
   */
  void fit();

  /**
   * Perform model inference
   */
  vector<vector<double>> predict();

  /**
   * Evaluate model using test data
   */
  void evaluate();
};

} // namespace ml
