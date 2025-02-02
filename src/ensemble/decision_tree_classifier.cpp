#include "decision_tree_classifier.h"

#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace ml{
	DecisionTreeClassifier::DecisionTreeClassifier(nlohmann::json parameters, Logger* logger): logger(logger) {

		if (parameters.contains("impurity_method")) {
		       string impurity_method_input = parameters["impurity_method"] ;
		       if (impurity_method_input != "gini" && impurity_method_input != "entropy") logger ->log(WARNING, "The specified method of calculating node impurity, " + impurity_method_input + " is currently unsupported. The default gini method will be used instead.");
		       else impurity_method = impurity_method_input;
		}

		if (parameters.contains("search_algorithm")) { 
			string search_algorithm_input = parameters["search_algorithm"] ;
			if (search_algorithm_input != "breadth" && search_algorithm_input != "depth") logger->log(WARNING, "The specified method of growing the tree, " + search_algorithm_input + " is currently unsupported. The default breadth first search method will be used instead.");
		}
	}

	DecisionTreeClassifier::~DecisionTreeClassifier() {
		tree.clear();
	}

	vector<unordered_map<int, int>> DecisionTreeClassifier::get_classes_frequencies(const vector<int> & indices, const vector<vector<int>> & outputs) {

		int number_output_variables = outputs[0].size();

		vector<unordered_map<int, int>> classes_frequencies {}; 
		for (int i = 0; i < number_output_variables; i++) {
			unordered_map<int, int> current_frequencies {};
			for (int index:indices) {
				int current_value = outputs[index][i];
				if (current_frequencies.count(current_value)) current_frequencies[current_value]++;
				else current_frequencies.insert(std::make_pair(current_value, 1));
			}
			classes_frequencies.push_back(current_frequencies);
		}
		return classes_frequencies;
	}

	double DecisionTreeClassifier::get_impurity(const vector<int> & indices, const vector<vector<int>> & outputs) {

		int number_outputs = outputs[0].size();
		int number_instances = indices.size();
		vector<unordered_map<int, int>> classes_frequencies = get_classes_frequencies(indices, outputs);

		double mean_impurity = 0.0;


		if (impurity_method == "gini") {
			for (int i = 0; i < number_outputs; i++) {
				unordered_map<int, int> current_frequencies = classes_frequencies[i];
				double output_impurity = 0.0;
				for (unordered_map<int, int>::iterator it = current_frequencies.begin(); it != current_frequencies.end(); it++) {
					double class_probability = 1.0*(it->second)/number_instances;
					output_impurity += class_probability*(1.0 - class_probability);
				}
				mean_impurity += output_impurity;
			}
		} else if (impurity_method == "entropy") {
			for (int i = 0; i < number_outputs; i++) {
				unordered_map<int, int> current_frequencies = classes_frequencies[i];
				double output_impurity = 0.0;
				for (unordered_map<int, int>::iterator it = current_frequencies.begin(); it != current_frequencies.end(); it++) {
					double class_probability = 1.0*(it->second)/number_instances;
					output_impurity += -1.0*class_probability*log(1.0 - class_probability);
				}
				mean_impurity += output_impurity;
			}
		}
		mean_impurity /= number_outputs;

		return mean_impurity;
	}


	pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>> DecisionTreeClassifier::split_node(shared_ptr<TreeNode> node, const vector<vector<double>> & features, const vector<vector<int>> & outputs) {
		vector<int> indices = node->node_indices;

		int best_feature_split = -1;
		double best_value_split = std::numeric_limits<double>::min();
		double parent_impurity = node->impurity;
		vector<int> best_left_indices {};
		vector<int> best_right_indices {};
		double best_left_impurity = -1.0;
		double best_right_impurity = -1.0;

		double max_impurity_reduction = 0.;
		
		pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>> children {};

		// loop over features
		for (int i = 0; i < features[0].size(); i++) {
			unordered_set<double> unique_values {};
			for (int index:indices) unique_values.insert(features[index][i]);
			vector<int> left_indices {};
			vector<int> right_indices {};
			for (double value:unique_values) {
				for (int index:indices) {
					double current_value = features[index][i];
					if (current_value <= value) left_indices.push_back(index);
					else right_indices.push_back(index);
				}

					// Compare impurities of current node and left and right splits
				if (left_indices.empty() || right_indices.empty()) continue;
				double left_impurity = get_impurity(left_indices, outputs);
				double right_impurity = get_impurity(right_indices, outputs);
				double average_child_impurity = (1.0*left_indices.size()/indices.size())*left_impurity + (1.0*right_indices.size()/indices.size())*right_impurity ;
				double impurity_reduction = parent_impurity - average_child_impurity;
				if (impurity_reduction > max_impurity_reduction) {
					best_feature_split = i;
					best_value_split = value;
					max_impurity_reduction = impurity_reduction;
					best_left_indices.assign(left_indices.begin(), left_indices.end());
					best_right_indices.assign(right_indices.begin(), right_indices.end());
					best_left_impurity = left_impurity;
					best_right_impurity = right_impurity;
				}
			}
		}

		// Create children if a suitable split was found
		if (best_feature_split > -1) {
			shared_ptr<TreeNode> left_child (new TreeNode(best_left_indices, node->depth + 1));
			shared_ptr<TreeNode> right_child (new TreeNode(best_right_indices, node->depth + 1));
			left_child -> impurity = best_left_impurity;
			right_child-> impurity = best_right_impurity;
			node->feature_split = best_feature_split;
			node-> value_split = best_value_split;
			node -> left_child = left_child;
			node->right_child = right_child;
			children.first = left_child;
			children.second = right_child;
			max_depth = max(max_depth, 1 + node->depth);
			number_splits++;
			feature_importances[best_feature_split] += (1.0*indices.size()/features.size())*max_impurity_reduction;
		} else number_leaf_nodes++;

		return children;

	}

	void DecisionTreeClassifier::breadth_first_search(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {

		tree.clear();
		int number_instances = features.size();


		int number_features = features[0].size();
		feature_importances.resize(number_features);
		fill(feature_importances.begin(), feature_importances.end(), 0.);
		vector<int> all_indices(number_instances, 0);
		for (int i = 0; i < all_indices.size(); i++) all_indices[i] = i;
		root = make_shared<TreeNode>(TreeNode(all_indices, 0));
		list<shared_ptr<TreeNode>> node_queue {};
		node_queue.push_back(root);

		while (!node_queue.empty()) {
			shared_ptr<TreeNode> current_node = node_queue.front() ;
			node_queue.pop_front();
			pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>> children = split_node(current_node, features, outputs);
			shared_ptr<TreeNode> left_child = children.first;
			shared_ptr<TreeNode> right_child = children.second;
			if (left_child != nullptr) {
				node_queue.push_back(left_child);
				node_queue.push_back(right_child);
			}
			tree.push_back(current_node);
		}

		double feature_importances_sum = accumulate(feature_importances.begin(), feature_importances.end(), 0.);
		for (double value:feature_importances) value /= feature_importances_sum;
		report_fit_results();


	}



	void DecisionTreeClassifier::fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {

		if (impurity_method == "breadth") this->breadth_first_search(features, outputs);
		else this -> depth_first_search(features, outputs);
	}

	void DecisionTreeClassifier::report_fit_results() {
		logger->log(INFO, "Decision tree characteristics");
		logger->log(INFO, "Total number of nodes = " + to_string(tree.size()));
		logger->log(INFO, "Number of splits = " + to_string(number_splits));
		logger->log(INFO, "Number of leaf nodes = " + to_string(number_leaf_nodes));
		logger->log(INFO, "Maximum node depth = " + to_string(max_depth));
		logger->log(INFO, "Feature Importances");
		for (int i = 0; i < feature_importances.size(); i++) 
			logger->log(INFO, "Feature " + to_string(i) + " importance = "+ to_string(feature_importances[i]));

	}
}
