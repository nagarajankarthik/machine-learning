/*
 * Header file for decision tree class
 */

#pragma once
#include "../base_model/base_model.h"
#include <list>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

namespace ml {
/**
 * Class defining attributes of each node in a decision tree.
 */
struct TreeNode {
  shared_ptr<TreeNode> left_child;
  shared_ptr<TreeNode> right_child;
  int feature_split = -1;
  double value_split = std::numeric_limits<double>::min();
  vector<int> node_indices{};
  int depth = -1;
  double impurity = -1;
  TreeNode(vector<int> node_indices, int depth)
      : node_indices(node_indices), depth(depth) {};
};

/**
 * Implement decision tree using recursive partitioning algorithm.
 */
class DecisionTreeClassifier : public BaseModel {
public:
  /**
   * Criterion to quantify individual node impurities.
   * Allowed values are 'gini' and 'entropy'.
   */
  string impurity_method = "gini";

  /**
   * Algorithm to grow decision tree.
   * Allowed values are "breadth" or "depth".
   * They correspond to breadth-first and depth-first searches
   * respectively.
   */
  string search_algorithm = "breadth";

  /**
   * Total number of nodes.
   */
  int total_nodes = 0;

  /**
   * Number of leaf nodes
   */
  int number_leaf_nodes = 0;

  /**
   * Number of splits
   */
  int number_splits = 0;

  /**
   * Maximum depth of all nodes
   */
  int max_depth = 0;

  /**
   * Fraction determining the number of features
   * that will be considered during each split.
   */
  double max_feature_fraction = 1.0;

  /**
   * Maximum number of features that
   * will be considered during each split.
   */
  int max_features = 0;

  /**
   * Feature Importances
   */
  vector<double> feature_importances{};

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
   * Pointer to tree root
   */
  shared_ptr<TreeNode> root;

  /**
   * Constructor
   */
  DecisionTreeClassifier(nlohmann::json model_parameters,
                         shared_ptr<Logger> logger);

  /**
   * Destructor
   */
  ~DecisionTreeClassifier() {};

  /**
   * Initialize data
   */
  void set_data(TrainTestData &&train_test);

  /**
   * Compile number of occurrences of each unique class for each output
   * variable.
   */
  vector<unordered_map<double, int, DoubleHash, DoubleEqual>>
  get_classes_frequencies(const vector<int> &indices);

  /**
   * Calculate node impurity by evaluating the impurity separately for each
   * output variable using the gini or entropy method and averaging the result.
   * This is supposedly how scikit-learn implements a decision tree.
   * See
   * https://stackoverflow.com/questions/50715574/how-is-the-impurity-decrease-of-a-split-computed-in-case-we-have-multiple-output
   */
  double get_impurity(const vector<int> &indices);

  /**
   * Determine best feature and value for splitting a node.
   */
  pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>>
  split_node(shared_ptr<TreeNode> node);

  /**
   * Grow decision tree using breadth first search.
   */
  void breadth_first_search();

  /**
   * Grow decision tree using depth first search.
   */
  void depth_first_search(shared_ptr<TreeNode> node);

  /**
   * Grow a decision tree using the recursive partitioning algorithm.
   * Calls either depth_first_search or breadth_first_search.
   */
  void fit();

  /**
   * Perform inference using decision tree grown by call to fit method.
   */
  vector<vector<double>> predict();

  /**
   * Log characteristics of decision tree after training.
   */
  void report_fit_results();

  /**
   * Evaluate model performance on test data
   */
  void evaluate();
};

} // namespace ml
