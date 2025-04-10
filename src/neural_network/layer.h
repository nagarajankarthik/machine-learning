
#include "../tensor/tensor.h"
#include "../tensor/tensor_operations.h"
#include <random>
#include <variant>
using namespace std;

namespace ml {

/**
 * Abstract class defining template for neural network layers
 */
class Layer {
      public:
	// Random seed
	int random_seed = 0;
	// Random generator
	std::mt19937 random_generator;

	Layer() { random_generator = std::mt19937(random_seed); };
	virtual shared_ptr<Tensor> forward(shared_ptr<Tensor> input) = 0;

      protected:
	shared_ptr<Tensor> weights_bias = nullptr;
};

class FullConnectedLayer : public Layer {
      public:
	// Number of input values
	int number_inputs = 0;

	// Number of output values. Equal to number of nodes in layer.
	int number_outputs = 0;
	FullConnectedLayer(int number_inputs, int number_outputs,
			   string init_method, shared_ptr<Logger> logger)
	    : number_inputs(number_inputs), number_outputs(number_outputs) {

		// Initialize weights
		int number_weights = number_inputs * number_outputs;
		int number_biases = number_outputs;
		int number_values = number_weights + number_biases;
		int number_columns = number_inputs + 1;
		vector<int> weights_bias_shape{1, number_outputs,
					       1 + number_inputs};
		vector<double> weights_bias_values(number_values, 0.0);

		double init_limit =
		    init_method == "glorot"
			? sqrt(6.0 / (number_inputs + number_outputs))
			: 1.0;
		variant<normal_distribution<double>,
			uniform_real_distribution<double>>
		    distribution;

		variant<normal_distribution<double>,
			uniform_real_distribution<double>>
		    distribution_variant;

		if (init_method == "glorot_normal") {
			double stddev =
			    sqrt(2.0 / (number_inputs + number_outputs));
			distribution_variant =
			    normal_distribution<double>(0.0, stddev);
		} else if (init_method == "glorot_uniform") {
			double limit =
			    sqrt(6.0 / (number_inputs + number_outputs));
			distribution_variant =
			    uniform_real_distribution<double>(-limit, limit);
		} else if (init_method == "he_normal") {
			double stddev = sqrt(2.0 / number_inputs);
			distribution_variant =
			    normal_distribution<double>(0.0, stddev);
		} else if (init_method == "he_uniform") {
			double limit = sqrt(6.0 / number_inputs);
			distribution_variant =
			    uniform_real_distribution<double>(-limit, limit);
		} else {
			throw invalid_argument("Unknown init method: " +
					       init_method);
		}

		// Use std::visit to apply the chosen distribution
		// https://medium.com/@weidagang/modern-c-std-variant-and-std-visit-3c16084db7dc
		for (int i = 0; i < number_values; i++) {
			weights_bias_values[i] =
			    (i + 1) % number_columns
				? std::visit(
				      [&](auto &dist) {
					      return dist(random_generator);
				      },
				      distribution_variant)
				: 0.0;
		}

		weights_bias = make_shared<Tensor>(weights_bias_values,
						   weights_bias_shape, logger);
	};

	/**
	 * Function to calculate outputs
	 * @param input
	 * @return
	 */
	shared_ptr<Tensor> forward(shared_ptr<Tensor> input) override {
		int input_columns = input->shape[input->shape.size() - 1];
		vector<double> last_row_values(input_columns, 1.0);
		shared_ptr<Tensor> last_row = make_shared<Tensor>(
		    last_row_values, vector<int>{1, 1, input_columns},
		    input->logger);
		shared_ptr<Tensor> input_extended =
		    concatenate_forward(input, last_row, 1);
		return batch_matmul_forward(weights_bias, input_extended);
	}
};

} // namespace ml
