/*
 * header file for random forest class
 */
#include "decision_tree_classifier.h"
#include <random>

using namespace std;

namespace ml
{

	class RandomForestClassifier
	{
	public:

		/**
		 * Pointer to an instance of Logger
		 */
		shared_ptr<Logger> logger;

		/**
		 * Parameters to pass to Decision Tree constructor
		 */
		nlohmann::json parameters ;

		/**
		 * Random number engine
		 */
		std::mt19937 random_generator;

		// Number of trees
		int number_trees = 20;

		/**
		 * An array of decision trees
		 */
		vector<shared_ptr<DecisionTreeClassifier>> trees {};

		/**
		 * Features for training data
		 */
		vector<vector<double>> train_features {};

		/**
		 * Labels for training data
		 */
		vector<vector<int>> train_labels {} ;

		/**
		 * Constructor
		 */
		RandomForestClassifier(nlohmann::json parameters, shared_ptr<Logger> logger)  ;

		/**
		 * Destructor
		 */
		~RandomForestClassifier() {};

		/**
		 * Get bootstrap sample
		 */
		void get_bootstrap_sample(vector<vector<double>> & features_sample, vector<vector<int>> & outputs_sample);

		/**
		 * Perform model training.
		 */
		void fit(const vector<vector<double>> && features, const vector<vector<int>> && labels) ;

		/**
		 * Perform model inference
		 */
		vector<vector<int>> predict(const vector<vector<double>> & test_features);
	};
	
}
