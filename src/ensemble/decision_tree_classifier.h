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
		 * Train a decision tree using the recursive partitioning algorithm.
		 */
		void fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs);



	};
			
}
