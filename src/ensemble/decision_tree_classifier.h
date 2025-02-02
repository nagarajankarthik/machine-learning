/*
 * Header file for decision tree class
 */

#include <string>
#include <vector>
#include <list>
#include <unordered_set>
#include <memory>
#include "../utils/json.hpp"
#include "../utils/logging.cpp"

using namespace std;

namespace ml
{
	/**
	 * Class defining attributes of each node in a decision tree.
	 */
	struct TreeNode {
		shared_ptr<TreeNode> left_child  ;
		shared_ptr<TreeNode> right_child ;
		int feature_split = -1;
		double value_split = std::numeric_limits<double>::min();
		vector<int> node_indices {};
	        int depth = -1;
		double impurity = -1;
		TreeNode(vector<int> node_indices, int depth): node_indices(node_indices), depth(depth) {};
	};	       

	/**
	 * Implement decision tree using recursive partitioning algorithm.
	 */
	class DecisionTreeClassifier
	{
	public:
		/**
		 * Pointer to an instance of Logger.
		 */
		Logger * logger ;

		/**
		 * Criterion to quantify individual node impurities.
		 * Allowed values are 'gini' and 'entropy'.
		 */
		string impurity_method = "gini";


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
		 * Feature Importances
		 */
		vector<double> feature_importances {};

		/**
		 * Pointer to tree root
		 */

		shared_ptr<TreeNode> root ;


		/**
		 * Container of TreeNodes.
		 */
		vector<shared_ptr<TreeNode>> tree {};

		/**
		 * Constructor
		 */
		DecisionTreeClassifier(nlohmann::json model_parameters, Logger * logger);

		/**
		 * Destructor
		 */
		~DecisionTreeClassifier() ;

		/**
		 * Compile number of occurrences of each unique class for each output variable.
		 */
		vector<unordered_map<int, int>> get_classes_frequencies(const vector<int> & indices, const vector<vector<int>> & outputs) ;

		/**
		 * Calculate node impurity by evaluating the impurity separately for each output variable using the gini or entropy method and averaging the result.
		 * This is supposedly how scikit-learn implements a decision tree.
		 * See https://stackoverflow.com/questions/50715574/how-is-the-impurity-decrease-of-a-split-computed-in-case-we-have-multiple-output

		 */
		double get_impurity(const vector<int> & indices, const vector<vector<int>> & outputs) ;

		/**
		 * Determine best feature and value for splitting a node.
		 */
		pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>> split_node(shared_ptr<TreeNode> node, const vector<vector<double>> & features, const vector<vector<int>> & outputs) ;

		/**
		 * Train a decision tree using the recursive partitioning algorithm.
		 */
		void fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs);

		/**
		 * Log characteristics of decision tree after training.
		 */
		void report_fit_results();



	};
			
}
