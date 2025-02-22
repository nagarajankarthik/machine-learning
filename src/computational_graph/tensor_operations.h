#include "tensor.h"
#include "../utils/logging.h"
using namespace std;

namespace ml {

	inline void add_backward(const vector<double>& seed, shared_ptr<Tensor> t1, shared_ptr<Tensor> t2) {

		shared_ptr<Logger> logger = t1->logger;

		if (t1->shape != t2->shape) {
			logger->log(ERROR, "Cannot back-propagate gradients in add_backward: Input Tensors have different shapes");
			exit(1);
		}

		for (int i = 0; i < t1->values.size(); i++) {
			t1->gradients[i] += seed[i];
			t2->gradients[i] += seed[i];
		}
	}


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

		function<void(const vector<double>&, shared_ptr<Tensor>, shared_ptr<Tensor>)> add_back = add_backward;

		return Tensor(sum_values, t1.shape, logger, make_shared<Tensor>(t1), make_shared<Tensor>(t2), add_back); 
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

	inline vector<vector<double>> matmul(vector<vector<double>> m1, vector<vector<double>> m2, shared_ptr<Logger> logger) {

		if (m1[0].size() != m2.size()) {
			logger->log(ERROR, "Invalid matrix dimensions");
			exit(1);
		}

		vector<vector<double>> matrix_product(m1.size(), vector<double>(m2[0].size()));
		for (int i = 0; i < m1.size(); i++) {
			for (int j = 0; j < m2[0].size(); j++) {
				double sum = 0;
				for (int k = 0; k < m2.size(); k++) {
					sum += m1[i][k]*m2[k][j];
				}
				matrix_product[i][j] = sum;
			}
		}
		return matrix_product;
	}


	inline void recurse_matmul(Tensor & t3, const Tensor& t1, const Tensor& t2, vector<int> & new_position, int axis=0) {

		if (axis == t3.shape.size() - 2) {
			vector<vector<double>> m1 = t1.get_matrix_at(new_position);
			vector<vector<double>> m2 = t2.get_matrix_at(new_position);
			vector<vector<double>> matrix_product = matmul(m1, m2, t1.logger);
			t3.set_matrix_at(new_position, matrix_product);

			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3.shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_matmul(t3, t1, t2, new_position, axis+1);
		}
		new_position.pop_back();
		
	}

	inline void batch_matmul_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		shared_ptr<Tensor> t2 = t3->input_second;

		shared_ptr<Logger> logger = t1->logger;
		vector<int> new_shape = get_shape_after_matmul(t1, t2);
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) {
			new_size *= new_shape[i];
		}
	}


	inline Tensor batch_matmul_forward(const Tensor& t1, const Tensor& t2) {

		shared_ptr<Logger> logger = t1.logger;
		vector<int> new_shape = get_shape_after_matmul(t1, t2);
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) {
			new_size *= new_shape[i];
		}

		function<void(const vector<double>&, shared_ptr<Tensor>, shared_ptr<Tensor>)> batch_matmul_back = batch_matmul_backward;

		Tensor result = Tensor(vector<double>(new_size, 0.), new_shape, logger, make_shared<Tensor>(t1), make_shared<Tensor>(t2), batch_matmul_backward);

		vector<int> new_position{};
		recurse_matmul(result, t1, t2, new_position);
		return result;

	}


}// namespace ml
