#include "decision_tree_classifier.h"

#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace ml{
	DecisionTreeClassifier::DecisionTreeClassifier(nlohmann::json parameters, shared_ptr<Logger> logger): logger(logger) {

		if (parameters.contains("impurity_method")) {
		       string impurity_method_input = parameters["impurity_method"] ;
		       if (impurity_method_input != "gini" && impurity_method_input != "entropy") logger ->log(WARNING, "The specified method of calculating node impurity, " + impurity_method_input + " is currently unsupported. The default gini method will be used instead.");
		       else impurity_method = impurity_method_input;
		}

		if (parameters.contains("search_algorithm")) { 
			string search_algorithm_input = parameters["search_algorithm"] ;
			if (search_algorithm_input != "breadth" && search_algorithm_input != "depth")  
				logger->log(WARNING, "The specified method of growing the tree, " + search_algorithm_input + " is currently unsupported. The default breadth first search method will be used instead.");
			else search_algorithm = "depth";
		}
		

		if (parameters.contains("max_feature_fraction")) {
				double max_feature_fraction_input = parameters["max_feature_fraction"];
				if (max_feature_fraction_input < 0.2) max_feature_fraction = 0.2;
				else if (max_feature_fraction > 1.0) max_feature_fraction = 1.0;
				else max_feature_fraction = max_feature_fraction_input;
		}
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

		vector<int> data_indices = node->node_indices;

		int best_feature_split = -1;
		double best_value_split = std::numeric_limits<double>::min();
		double parent_impurity = node->impurity;
		vector<int> best_left_indices {};
		vector<int> best_right_indices {};
		double best_left_impurity = -1.0;
		double best_right_impurity = -1.0;

		double max_impurity_reduction = 0.;
		
		pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>> children {nullptr, nullptr};

		int number_features = features[0].size();
		vector<int> feature_indices(number_features, 0);
		for (int i = 0; i < number_features; i++) feature_indices[i] = i;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(feature_indices.begin(), feature_indices.end(), std::default_random_engine(seed));

		for (int i = 0; i < max_features; i++) {
			unordered_set<double> unique_values {};
			int feature_index = feature_indices[i];
			for (int data_index:data_indices) unique_values.insert(features[data_index][feature_index]);
			vector<int> left_indices {};
			vector<int> right_indices {};
			for (double value:unique_values) {
				for (int data_index:data_indices) {
					double current_value = features[data_index][feature_index];
					if (current_value <= value) left_indices.push_back(data_index);
					else right_indices.push_back(data_index);
				}

					// Compare impurities of current node and left and right splits
				if (left_indices.empty() || right_indices.empty()) continue;
				double left_impurity = get_impurity(left_indices, outputs);
				double right_impurity = get_impurity(right_indices, outputs);
				double average_child_impurity = (1.0*left_indices.size()/data_indices.size())*left_impurity + (1.0*right_indices.size()/data_indices.size())*right_impurity ;
				double impurity_reduction = parent_impurity - average_child_impurity;
				if (impurity_reduction > max_impurity_reduction) {
					best_feature_split = feature_index;
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
			total_nodes += 2;
			max_depth = max(max_depth, 1 + node->depth);
			number_splits++;
			feature_importances[best_feature_split] += (1.0*data_indices.size()/features.size())*max_impurity_reduction;
		} else number_leaf_nodes++;

		return children;

	}

	void DecisionTreeClassifier::breadth_first_search(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {

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
		}

		double feature_importances_sum = accumulate(feature_importances.begin(), feature_importances.end(), 0.);
		for (double value:feature_importances) value /= feature_importances_sum;

	}

	void DecisionTreeClassifier::dfs_recurse(shared_ptr<TreeNode> node, const vector<vector<double>> & features, const vector<vector<int>> & outputs) {
		pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>> children = split_node(node, features, outputs);
		shared_ptr<TreeNode> left_child = children.first;
		shared_ptr<TreeNode> right_child = children.second;
		if (left_child != nullptr) {
			dfs_recurse(left_child, features, outputs);
			dfs_recurse(right_child, features, outputs);
		}

	}


	void DecisionTreeClassifier::depth_first_search(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {
		dfs_recurse(root, features, outputs);

		double feature_importances_sum = accumulate(feature_importances.begin(), feature_importances.end(), 0.);
		for (double value:feature_importances) value /= feature_importances_sum;

	}



	void DecisionTreeClassifier::fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {


		root = nullptr;
		int number_features = features[0].size();
		max_features = (int) round(max_feature_fraction*number_features);

		int number_instances = features.size();
		vector<int> all_indices(number_instances, 0);
		for (int i = 0; i < all_indices.size(); i++) all_indices[i] = i;
		root = make_shared<TreeNode>(TreeNode(all_indices, 0));
		root->impurity = get_impurity(all_indices, outputs);
		total_nodes = 1;


		logger->log(DEBUG, "Number of training instances = " + to_string(features.size()));

		logger->log(DEBUG, "Number of features = " + to_string(features[0].size()));
		logger->log(DEBUG, "Number of outputs = " + to_string(outputs[0].size()));
		feature_importances.resize(number_features);
		fill(feature_importances.begin(), feature_importances.end(), 0.);
		if (impurity_method == "breadth") this->breadth_first_search(features, outputs);
		else this -> depth_first_search(features, outputs);
		report_fit_results();
	}


	vector<vector<int>> DecisionTreeClassifier::predict(const vector<vector<int>> & train_outputs, const vector<vector<double>> & test_features) { 
		if (this->root == nullptr) {
			logger->log(ERROR, "Decision tree must be grown by calling fit method before performing inference.");
			return {};
		}

		int number_test_instances = test_features.size();

		vector<int> tmp {};

		vector<vector<int>> predictions(number_test_instances, tmp);

		int number_outputs = train_outputs[0].size();

		for (int j = 0; j < number_test_instances; j++) {
			vector<double> instance_features = test_features[j];
			vector<int> instance_predictions {};
			shared_ptr<TreeNode> current_node(root);
			while (current_node -> left_child) {
				int split_feature = current_node->feature_split;
				double split_value = current_node->value_split;
				double instance_value = instance_features[split_feature];
				if (instance_value <= split_value) current_node = current_node->left_child;
				else current_node = current_node->right_child;
			}
			int current_leaf_size = current_node->node_indices.size();
			vector<unordered_map<int, int>> classes_frequencies = get_classes_frequencies(current_node-> node_indices, train_outputs);
			vector<int> current_predictions (number_outputs,0);
			for (int i = 0; i < number_outputs; i++) {
				unordered_map<int, int> current_frequencies = classes_frequencies[i];
				int predicted_class = 0;
				double max_probability = 0.;
				for (unordered_map<int, int>::iterator it = current_frequencies.begin(); it != current_frequencies.end(); it++) {
					double class_probability = 1.0*(it->second)/current_leaf_size;
					if (class_probability > max_probability) {
						max_probability = class_probability;
						predicted_class = it->first;
					}
				}
				current_predictions[i] = predicted_class;
			}
			predictions[j] = current_predictions;
		}
		return predictions;
	}




	void DecisionTreeClassifier::report_fit_results() {
		logger->log(INFO, "Decision tree characteristics");
		logger->log(INFO, "Total number of nodes = " + to_string(total_nodes));
		logger->log(INFO, "Number of splits = " + to_string(number_splits));
		logger->log(INFO, "Number of leaf nodes = " + to_string(number_leaf_nodes));
		logger->log(INFO, "Maximum node depth = " + to_string(max_depth));
		logger->log(INFO, "Feature Importances");
		for (int i = 0; i < feature_importances.size(); i++) 
			logger->log(INFO, "Feature " + to_string(i) + " importance = "+ to_string(feature_importances[i]));

	}
}
