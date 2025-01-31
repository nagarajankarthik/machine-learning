#include "decision_tree_classifier.h"

#include <vector>

using namespace std;

namespace ml{
	DecisionTreeClassifier::DecisionTreeClassifier(nlohmann::json parameters, Logger* _logger): logger(_logger) {

		// Override values of parameters specified in input.
		if (parameters.contains("impurity_method")) impurity_method = parameters["impurity_method"] ;
	}

	DecisionTreeClassifier::~DecisionTreeClassifier() {
		_tree.clear();
	}

	void DecisionTreeClassifier::fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {

		_tree.clear();

		int number_instances = features.size();


		return;

	}
}
