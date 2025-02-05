/*
 * header file for random forest class
 */
#include "decision_tree_classifier.h"

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
		 * Perform model training.
		 */
		void fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs) ;
	};
	
}
