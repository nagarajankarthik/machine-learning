/*
 * header file for random forest class
 */
#include "decision_tree_classifier.h"
#include <random>
#include <omp.h>

using namespace std;

namespace ml
{

	class RandomForestClassifier: public BaseModel
	{
	public:

		/**
		 * Parameters to pass to Decision Tree constructor
		 */
		nlohmann::json parameters ;

		// Number of trees
		int number_trees = 20;

		/**
		 * An array of decision trees
		 */
		vector<shared_ptr<DecisionTreeClassifier>> trees {};

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
		void get_bootstrap_sample(vector<vector<double>> & features_sample, vector<vector<double>> & outputs_sample);

		/**
		 * Perform model training.
		 */
		void fit(const vector<vector<double>> && features, const vector<vector<double>> && labels) ;

		/**
		 * Perform model inference
		 */
		vector<vector<double>> predict(const vector<vector<double>> & test_features);

		/**
		 * Evaluate model using test data
		 */
		void evaluate(const vector<vector<double>> & test_features, const vector<vector<double>> & test_labels) ;
	};
	
}
