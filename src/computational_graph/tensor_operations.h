#include "tensor.h"
#include "../utils/logging.h"
using namespace std;

namespace ml {

	inline Tensor add_forward(const Tensor& t1, const Tensor& t2) {

		shared_ptr<Logger> logger = t1.logger;

		if (t1.shape != t2.shape) {
			logger->log(ERROR, "Tensors have different shapes");
			exit(1);
		}
		vector<double> sum_values(t1.values.begin(), t1.values.end());
		for (int i = 0; i < t2.values.size(); i++) {
			sum_values[i] += t2.values[i];
		}

		return Tensor(sum_values, t1.shape, logger, t1.input_first, t1.input_second);
	}

	inline Tensor elementwise_product_forward(const Tensor& t1, const Tensor& t2) {

		shared_ptr<Logger> logger = t1.logger;

		if (t1.shape != t2.shape) {
			logger->log(ERROR, "Tensors have different shapes");
			exit(1);
		}
		vector<double> product_values(t1.values.begin(), t1.values.end());
		for (int i = 0; i < t2.values.size(); i++) {
			product_values[i] *= t2.values[i];
		}

		return Tensor(product_values, t1.shape, logger, t1.input_first, t1.input_second);
		
	}

	inline vector<int> get_shape_after_matmul(const Tensor& t1, const Tensor& t2) {
		vector<int> new_shape {};
		int m = t1.shape.size();
		int n = t2.shape.size();
		shared_ptr<Logger> logger = t1.logger;

		// Process matrix dimensions
		if (t1.shape[m-1] != t2.shape[n-2]) {
			logger->log(ERROR, "Size of tensors unsuitable for batch matrix multiplication.");
			logger->log(ERROR, "Size of last dimension of t1 is " + to_string(t1.shape[m-1]) + " but size of second to last dimension of t2 is " + to_string(t2.shape[n-2]));
			exit(1);
		} else {
		       new_shape.push_back(t2.shape[n-1]);
		       new_shape.push_back(t1.shape[m-2]);
		}	       

		// Process batch (non-matrix) dimensions
		for (int i = 2; i < max(m, n); i++) {
			int first_index = m - 1 - i;
			int second_index = n - 1 - i;
			if (first_index < 0) new_shape.push_back(t2.shape[second_index]);
			else if (second_index < 0) new_shape.push_back(t1.shape[first_index]);
			else if (t1.shape[first_index] == t2.shape[second_index]) new_shape.push_back(t1.shape[first_index]);
			else {
			       int min_size = min(t1.shape[first_index], t2.shape[second_index]);
			       int max_size = max(t1.shape[first_index], t2.shape[second_index]);
			       if (min_size == 1) new_shape.push_back(max_size);
			       else { 
				       logger->log(ERROR, "Size mismatch at non-singleton index " + to_string(i) + " from the end of two tensors' shape arrays.\n The size of tensor a is " + to_string(t1.shape[first_index]) + " and the size of tensor b is " + to_string(t2.shape[second_index]));
				       exit(1);
			       }

			}
		}
		new_shape.assign(new_shape.rbegin(), new_shape.rend());
		return new_shape;
	}


	inline void recurse_matmul(const vector<int> & new_shape, const Tensor& t1, const Tensor& t2, vector<int> & new_position, vector<double> & new_values, int axis=0) {

		if (axis == new_shape.size() - 2) {
			vector<vector<double>> m1 = t1.get_matrix_at(new_position);
			vector<vector<double>> m2 = t2.get_matrix_at(new_position);
			vector<vector<double>> matrix_product = matmul(m1, m2);
			for (int i = 0; i < matrix_product.size(); i++) {
				for (int j = 0; j < matrix_product[i].size(); j++) {
					new_values.push_back(matrix_product[i][j]);
				}
			}

			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < new_shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_matmul(new_shape, t1, t2, new_position, new_values, axis+1);
		}
		
	}

	inline Tensor batch_matmul_forward(const Tensor& t1, const Tensor& t2) {

		shared_ptr<Logger> logger = t1.logger;
		vector<int> new_shape = get_shape_after_matmul(t1, t2);
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) {
			new_size *= new_shape[i];
		}
		vector<double> new_values(new_size, 0.);

		if (t1.shape[1] != t2.shape[0]) {
			logger->log(ERROR, "Tensors have different shapes");
			exit(1);
		}
		vector<double> product_values(t1.values.size());
		for (int i = 0; i < t1.shape[0]; i++) {
			for (int j = 0; j < t2.shape[1]; j++) {
				double sum = 0;
				for (int k = 0; k < t1.shape[1]; k++) {
					sum += t1.values[i*t1.shape[1] + k] * t2.values[k*t2.shape[1] + j];
				}
				product_values[i*t2.shape[1] + j] = sum;
			}
		}
		return Tensor(product_values, {t1.shape[0], t2.shape[1]}, logger, t1.input_first, t1.input_second);
	}
}// namespace ml
