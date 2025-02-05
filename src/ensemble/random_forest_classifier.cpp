#include "random_forest_classifier.h"

namespace ml
{
	
	RandomForestClassifier::RandomForestClassifier(nlohmann::json parameters, shared_ptr<Logger> logger): parameters(parameters), logger(logger) {
	
		if (parameters.contains("number_trees")) {
				int number_trees_input = parameters["number_trees"];
				if (number_trees_input > 0) number_trees = number_trees_input;
	}
} 
    void RandomForestClassifier::fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {

	    trees.clear();

	    vector<int> data_indices {};


	    for (int i = 0; i < number_trees; i++) {
		    int tr = 2;
	    }
    
    
    }

}
