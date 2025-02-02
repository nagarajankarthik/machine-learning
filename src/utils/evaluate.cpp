#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

namespace ml {

vector<vector<int>> get_confusion_matrix(const vector<vector<int>> & predictions, const vector<vector<int>> & test_outputs, unordered_set<int> unique_classes, int index_output) {
	int number_instances = test_outputs.size();
	int number_classes = unique_classes.size();
	vector<int> tmp(number_classes, 0);
	vector<vector<int>> confusion_matrix (number_classes, tmp);
	vector<int> unique_classes_array(unique_classes.begin(), unique_classes.end());
	sort(unique_classes_array.begin(), unique_classes_array.end());
	unordered_map<int, int> class_index {};
	for (int i = 0; i < number_classes; i++) class_index.insert(make_pair(unique_classes_array[i],i));


	for (int i = 0; i < number_instances; i++) {
		int predicted_class = predictions[i][index_output];
		int ground_truth_class = test_outputs[i][index_output];
		int predicted_index = class_index[predicted_class];
		int ground_truth_index = class_index[ground_truth_class];
		confusion_matrix[ground_truth_index][predicted_index]++;

	}

	return confusion_matrix;
}


}
