/*
 * header file for random forest class
 */
#include "../utils/json.hpp"
#include <vector>

using namespace std;

namespace ml {
	
	class RandomForestClassifier {
		public:

			// Data for features
			vector<vector<double>> features {};

			// Data for outputs
			vector<vector<double>> outputs {};


			// Number of trees 
			int numberTrees;

			void setParameters(nlohmann::json parameters);

			void fit();

			void predict_proba();




			void RunSingle();
	};
}

