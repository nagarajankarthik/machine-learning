#include "random_forest_classifier.h"
#include <random>

namespace ml
{
	
	RandomForestClassifier::RandomForestClassifier(nlohmann::json parameters, shared_ptr<Logger> logger): parameters(parameters), logger(logger) {
	
		if (parameters.contains("number_trees")) {
				int number_trees_input = parameters["number_trees"];
				if (number_trees_input > 0) number_trees = number_trees_input;
		}

		int random_seed = 0;

		if (parameters.contains("random_seed")) {
			int random_seed_input = parameters["random_seed"];
			if (random_seed_input > -1) random_seed = random_seed_input;
		}

		random_generator = std::mt19937(random_seed);

	}


	void RandomForestClassifier::get_bootstrap_sample(const vector<vector<double>> & features,const vector<vector<int>> & outputs, vector<vector<double>> & features_sample, vector<vector<int>> & outputs_sample) {
		int number_instances = features.size();
		vector<int> data_indices(number_instances, 0);

		std::uniform_int_distribution<int> distribution_uniform(0, number_instances - 1);
	       	
		for (int i = 0; i < number_instances; i++) {
			auto random_integer = distribution_uniform(random_generator);
			vector<double> features_random = features[random_integer];
			vector<int> outputs_random = outputs[random_integer];
			features_sample[i].assign(features_random.begin(), features_random.end());
			outputs_sample[i].assign(outputs_random.begin(), outputs_random.end());
		}
	}


	
    void RandomForestClassifier::fit(const vector<vector<double>> & features, const vector<vector<int>> & outputs) {

	    trees.clear();
	    int number_instances = features.size();
	    vector<vector<double>> features_sample {}; 
	    features_sample.assign(features.begin(), features.end());
	    vector<vector<int>> outputs_sample {};
	    outputs_sample.assign(outputs.begin(), outputs.end());


	    for (int i = 0; i < number_trees; i++) {
		    shared_ptr<DecisionTreeClassifier> classifier_tree (new DecisionTreeClassifier(parameters, logger));
		    get_bootstrap_sample(features, outputs, features_sample, outputs_sample);
		    classifier_tree ->fit(features_sample, outputs_sample);
		    trees.push_back(classifier_tree);
	    }
    
    }

	vector<vector<int>> RandomForestClassifier::predict(const vector<vector<int>> & train_outputs, const vector<vector<double>> & test_features) {

		unordered_map<int, int> tmp {};
		int number_test_instances = test_features.size();
		int number_outputs = train_outputs[0].size();
		int list_map_size = number_test_instances * number_outputs;
		vector<unordered_map<int,int>> list_freq_map(list_map_size, tmp) ;
			for (auto tree:trees) {
				vector<vector<int>> current_predictions = tree->predict(train_outputs, test_features);
				for (int i = 0; i < number_test_instances; i++) {
					for (int j = 0; j < number_outputs;j++) {
						int predicted_class = current_predictions[i][j];
						int index = i*number_outputs +j;
						unordered_map<int,int> current_map = list_freq_map.at(index);
						if (current_map.count(predicted_class)) current_map[predicted_class]++;
						else current_map.insert(make_pair(predicted_class, 1));
						list_freq_map[index] = current_map;
			}
		}
	}
			vector<int> tmp2(number_outputs, 0);
			vector<vector<int>> test_predictions(number_test_instances, tmp2);

			for (int index = 0; index < list_map_size; index++) {
				unordered_map<int, int> freq_map =  list_freq_map[index];
				int max_freq = 0;
				int max_freq_class = -1;
				for (auto it = freq_map.begin(); it != freq_map.end(); it++) {
					int current_class = it->first;
					int current_freq = it->second;
					if (current_freq > max_freq) {
						max_freq_class = current_class;
						max_freq = current_freq;
					}
				}
				if (max_freq_class == -1) logger ->log(ERROR, "Maximum frequency class could not be found correctly.");

				int j = index%number_outputs;
				int i = (index - j)/number_outputs;
				test_predictions[i][j] = max_freq_class;

			}

			return test_predictions;
	}


} //namespace ml
