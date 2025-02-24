#include "tensor.h"
#include "../utils/logging.h"
using namespace std;

namespace ml {


	inline vector<int> broadcast_shape(vector<int> t1_shape, vector<int> t2_shape, shared_ptr<Logger> logger) {
		vector<int> new_shape {};
		int m = t1_shape.size();
		int n = t2_shape.size();

		// Process batch (non-matrix) dimensions
		for (int i = 2; i < max(m, n); i++) {
			int first_index = m - 1 - i;
			int second_index = n - 1 - i;
			if (first_index < 0) new_shape.push_back(t2_shape[second_index]);
			else if (second_index < 0) new_shape.push_back(t1_shape[first_index]);
			else if (t1_shape[first_index] == t2_shape[second_index]) new_shape.push_back(t1_shape[first_index]);
			else {
			       int min_size = min(t1_shape[first_index], t2_shape[second_index]);
			       int max_size = max(t1_shape[first_index], t2_shape[second_index]);
			       if (min_size == 1) new_shape.push_back(max_size);
			       else { 
				       logger->log(ERROR, "Size mismatch at non-singleton index " + to_string(i) + " from the end of two tensors' shape arrays.\n The size of tensor a is " + to_string(t1_shape[first_index]) + " and the size of tensor b is " + to_string(t2_shape[second_index]));
				       exit(1);
			       }

			}
		}
		new_shape.assign(new_shape.rbegin(), new_shape.rend());
		return new_shape;
	}

	inline vector<vector<double>> add_matrix(vector<vector<double>> m1, vector<vector<double>> m2, shared_ptr<Logger> logger) {

		if (m1.size() != m2.size()) {
			logger->log(ERROR, "Error in add_matrix: First input has " + to_string(m1.size()) + " rows but second input has " + to_string(m2.size()) + " rows.");
			exit(1);
		}

		if (m1[0].size() != m2[0].size()) {
			logger->log(ERROR, "Error in add_matrix: First input has " + to_string(m1[0].size()) + " columns but second input has " + to_string(m2[0].size()) + " columns.");
			exit(1);
		}
		vector<vector<double>> matrix_sum(m1.size(), vector<double>(m1[0].size()));
		for (int i = 0; i < m1.size(); i++) {
			for (int j = 0; j < m1[0].size(); j++) {
				matrix_sum[i][j] = m1[i][j] + m2[i][j];
			}
		}
		return matrix_sum;
	}

	inline void recurse_add_backward(shared_ptr<Tensor> t3, shared_ptr<Tensor> t1, shared_ptr<Tensor> t2, vector<int> new_position, int axis=0) {

		if (axis == t3->shape.size() - 2) {
			vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
			t1->set_matrix(new_position, g3, "gradients");
			t2->set_matrix(new_position, g3, "gradients");
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_add_backward(t3, t1, t2, new_position, axis+1);
		}
		new_position.pop_back();

	}


	inline void add_batch_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		shared_ptr<Tensor> t2 = t3->input_second;

		shared_ptr<Logger> logger = t1->logger;

		vector<int> new_position{};
		recurse_add_backward(t3, t1, t2, new_position);

	}

	inline void recurse_add_forward(shared_ptr<Tensor> t3, const shared_ptr<Tensor> t1, const shared_ptr<Tensor> t2, vector<int> new_position, int axis=0) {

		if (axis == t3->shape.size() - 2) {
			vector<vector<double>> m1 = t1->get_matrix(new_position, "values");
			vector<vector<double>> m2 = t2->get_matrix(new_position, "values");
			vector<vector<double>> matrix_sum = add_matrix(m1, m2, t1->logger);
			t3->set_matrix(new_position, matrix_sum);
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_add_forward(t3, t1, t2, new_position, axis+1);
		}
		new_position.pop_back();

	}


	inline shared_ptr<Tensor> add_batch_forward(const shared_ptr<Tensor> t1, const shared_ptr<Tensor> t2) {

		shared_ptr<Logger> logger = t1->logger;
		vector<int> new_shape = broadcast_shape(t1->shape, t2->shape, logger);
		int m = t1->shape.size();
		int n = t2->shape.size();

		if (t1->shape[m-2] != t2->shape[n-2]) {
			logger->log(ERROR, " Error in add_batch_forward: Input Tensors have different shapes at second last dimension.");
			exit(1);
		}

		if (t1->shape[m-1] != t2->shape[n-1]) {
			logger->log(ERROR, " Error in add_batch_forward: Input Tensors have different shapes at last dimension.");
			exit(1);
		}

		new_shape.push_back(t1->shape[m-2]);
		new_shape.push_back(t1->shape[m-1]);

		vector<double> sum_values(t1->values.begin(), t1->values.end());

		function<void(shared_ptr<Tensor>)> add_back = add_batch_backward;
		shared_ptr<Tensor> t3 = make_shared<Tensor>(sum_values, new_shape, logger, t1, t2, add_batch_backward);
		vector<int> new_position{};
		recurse_add_forward(t3, t1, t2, new_position);
		return t3;

	}


	inline vector<int> get_shape_after_matmul(shared_ptr<Tensor> t1, shared_ptr<Tensor> t2) {
		int m = t1->shape.size();
		int n = t2->shape.size();
		shared_ptr<Logger> logger = t1->logger;


		vector<int> new_shape = broadcast_shape(t1->shape, t2->shape, logger);

		// Process matrix dimensions
		if (t1->shape[m-1] != t2->shape[n-2]) {
			logger->log(ERROR, "Size of tensors unsuitable for batch matrix multiplication.");
			logger->log(ERROR, "Size of last dimension of t1 is " + to_string(t1->shape[m-1]) + " but size of second to last dimension of t2 is " + to_string(t2->shape[n-2]));
			exit(1);
		} else {
		       new_shape.push_back(t1->shape[m-2]);
		       new_shape.push_back(t2->shape[n-1]);
		}	       

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



	inline vector<vector<double>> transpose_matrix(vector<vector<double>> matrix) {

		vector<vector<double>> transpose(matrix[0].size(), vector<double>(matrix.size()));
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[0].size(); j++) {
				transpose[j][i] = matrix[i][j];
			}
		}
		return transpose;
	}

	inline void recurse_matmul_backward(const shared_ptr<Tensor> t3, shared_ptr<Tensor> t1, shared_ptr<Tensor> t2, vector<int> & new_position, int axis=0) {

		if (axis == t3->shape.size() - 2) {
			vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
			vector<vector<double>> m1 = t1->get_matrix(new_position);
			vector<vector<double>> m2 = t2->get_matrix(new_position);
			vector<vector<double>> m1_transpose = transpose_matrix(m1);
			vector<vector<double>> m2_transpose = transpose_matrix(m2);
			vector<vector<double>> g1 = matmul(g3, m2_transpose, t1->logger);
			vector<vector<double>> g2 = matmul(m1_transpose, g3, t1->logger);
			t1->set_matrix(new_position, g1, "gradients");
			t2->set_matrix(new_position, g2, "gradients");
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_matmul_backward(t3, t1, t2, new_position, axis+1);
		}
		new_position.pop_back();
		
	}


	inline void batch_matmul_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		shared_ptr<Tensor> t2 = t3->input_second;

		shared_ptr<Logger> logger = t1->logger;

		vector<int> new_position{};
		recurse_matmul_backward(t3, t1, t2, new_position);

	}

	inline void recurse_matmul_forward(shared_ptr<Tensor> t3, const shared_ptr<Tensor> t1, const shared_ptr<Tensor> t2, vector<int> & new_position, int axis=0) {

		if (axis == t3->shape.size() - 2) {
			vector<vector<double>> m1 = t1->get_matrix(new_position);
			vector<vector<double>> m2 = t2->get_matrix(new_position);
			vector<vector<double>> matrix_product = matmul(m1, m2, t1->logger);
			t3->set_matrix(new_position, matrix_product);
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_matmul_forward(t3, t1, t2, new_position, axis+1);
		}
		new_position.pop_back();
		
	}

	inline shared_ptr<Tensor> batch_matmul_forward(shared_ptr<Tensor> t1, shared_ptr<Tensor> t2) {

		shared_ptr<Logger> logger = t1->logger;
		vector<int> new_shape = get_shape_after_matmul(t1, t2);
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) {
			new_size *= new_shape[i];
		}

		function<void(shared_ptr<Tensor>)> batch_matmul_back = batch_matmul_backward;

		shared_ptr<Tensor> t3 = make_shared<Tensor>(vector<double>(new_size, 0.), new_shape, logger, t1, t2, batch_matmul_backward);

		vector<int> new_position{};
		recurse_matmul_forward(t3, t1, t2, new_position);
		return t3;

	}


}// namespace ml
