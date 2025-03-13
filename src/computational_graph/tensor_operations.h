#include "tensor.h"
#include <math.h>
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

	inline shared_ptr<Tensor> concatenate_forward(shared_ptr<Tensor> t1, shared_ptr<Tensor> t2, int concat_dim = 0) {
		vector<int> first_shape = t1->shape;
		vector<int> second_shape = t2->shape;
		shared_ptr<Logger> logger = t1->logger;

		if (first_shape.size() != second_shape.size()) {
			logger->log(ERROR, "Error in concatenate_forward: Input Tensors have different shapes.");
			exit(1);
		}

		for (int i = 0; i < first_shape.size(); i++) {
			if (i != concat_dim && first_shape[i] != second_shape[i]) {
				logger->log(ERROR, "Error in concatenate_forward: Input Tensors have different shapes along dimension " + to_string(i) + ".");
				logger->log(ERROR, "Concatenation dimension is " + to_string(concat_dim) + ".");
				exit(1);
			}
		}

		vector<int> new_shape = first_shape;
		new_shape[concat_dim] += second_shape[concat_dim];
		int new_size = t1->values.size() + t2->values.size();
		shared_ptr<Tensor> t3 = make_shared<Tensor>(vector<double>(new_size, 0.), new_shape, logger, t1, t2, concatenate_backward);
		
	}

	// Activation functions
	inline void relu_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		for (int i = 0; i < t1->values.size(); i++) {
			t1->gradients[i] = t3->values[i] > 0. ? t3->gradients[i] : 0.;
		}
	}

	inline shared_ptr<Tensor> relu_forward(shared_ptr<Tensor> t1) {
		shared_ptr<Logger> logger = t1->logger;
		shared_ptr<Tensor> t3 = make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape, logger, t1, nullptr, relu_backward);
		for (int i = 0; i < t1->values.size(); i++) {
			t3->values[i] = max(0., t1->values[i]);
		}
		return t3;
	}

	inline void sigmoid_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		for (int i = 0; i < t1->values.size(); i++) {
			int function_result = t3->values[i] ;
			t1->gradients[i] = function_result * (1. - function_result) * t3->gradients[i]; 
		}
	}

	inline shared_ptr<Tensor> sigmoid_forward(shared_ptr<Tensor> t1) {
		shared_ptr<Logger> logger = t1->logger;
		shared_ptr<Tensor> t3 = make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape, logger, t1, nullptr, sigmoid_backward);
		for (int i = 0; i < t1->values.size(); i++) {
			t3->values[i] = 1. / (1. + exp(-t1->values[i])); 
		}
		return t3;
	}

	inline void tanh_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		for (int i = 0; i < t1->values.size(); i++) {
			int function_result = t3->values[i] ;
			t1->gradients[i] = (1. - function_result*function_result) * t3->gradients[i]; 
		}
	}


	inline shared_ptr<Tensor> tanh_forward(shared_ptr<Tensor> t1) {
		shared_ptr<Logger> logger = t1->logger;
		shared_ptr<Tensor> t3 = make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape, logger, t1, nullptr, tanh_backward);
		for (int i = 0; i < t1->values.size(); i++) {
			t3->values[i] = tanh(t1->values[i]); 
		}
		return t3;
	}

	inline vector<vector<double>> elementwise_multiplication(vector<vector<double>> m1, vector<vector<double>> m2, shared_ptr<Logger> logger) {
		if (m1.size() != m2.size() || m1[0].size() != m2[0].size()) {
			logger->log(ERROR, "Matrix dimensions do not match in elementwise_multiplication.");
			exit(1);
		}
		vector<vector<double>> result(m1.size(), vector<double>(m1[0].size(), 0.));
		for (int i = 0; i < m1.size(); i++) {
			for (int j = 0; j < m1[0].size(); j++) {
				result[i][j] = m1[i][j] * m2[i][j];
			}
		}
		return result;
	}

	/**
	 * Returns the sum of the values in each column of the input matrix.
	 */
	inline vector<double> matrix_col_sum(vector<vector<double>> matrix) {
		vector<double> result(matrix[0].size(), 0.);
		for (int j = 0; j < matrix[0].size(); j++) {
			double sum = 0.;
			for (int i = 0; i < matrix.size(); i++) {
				sum += matrix[i][j];
			}
			result[j] = sum;
		}
		return result;
	}

	inline void recurse_softmax_backward(const shared_ptr<Tensor> t3, shared_ptr<Tensor> t1, vector<int> & new_position, int axis=0) {

		if (axis == t3->shape.size() - 2) {
			vector<vector<double>> g3 = t3->get_matrix(new_position, "gradients");
			vector<vector<double>> m1 = t1->get_matrix(new_position);
			vector<vector<double>> gradient_values_product = elementwise_multiplication(g3, m1, t1->logger);
			vector<double> gradient_values_product_sum = matrix_col_sum(gradient_values_product);

			vector<vector<double>> g1(g3.size(), vector<double>(g3[0].size(), 0.));
			for (int j = 0; j < g1[0].size(); j++) {
				for (int i = 0; i < g1.size(); i++) {
					g1[i][j] = m1[i][j] * (g3[i][j] - gradient_values_product_sum[j]);
				}
			}
			t1->set_matrix(new_position, g1, "gradients");
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_softmax_backward(t3, t1, new_position, axis+1);
		}
		new_position.pop_back();
		
	}


	inline void softmax_backward(shared_ptr<Tensor> t3) {

		shared_ptr<Tensor> t1 = t3->input_first;
		if (!t1) return;
		vector<int> new_position {};
		recurse_softmax_backward(t3, t1, new_position);
	}


	inline vector<vector<double>> evaluate_softmax(const vector<vector<double>> & m1, shared_ptr<Logger> logger) {
		vector<vector<double>> softmax_result (m1.size(), vector<double>(m1[0].size(), 0.));
		for (int j = 0; j < m1[0].size(); j++) {
			double sum = 0.;
			for (int i = 0; i < m1.size(); i++) {
				softmax_result[i][j] = exp(m1[i][j]);
				sum += softmax_result[i][j];
			}
			for (int i = 0; i < m1.size(); i++) {
				softmax_result[i][j] /= sum;
			}
		}
		return softmax_result;
	}

	inline void recurse_softmax_forward(shared_ptr<Tensor> t3, const shared_ptr<Tensor> t1, vector<int> & new_position, int axis=0) {

		if (axis == t3->shape.size() - 2) {
			vector<vector<double>> m1 = t1->get_matrix(new_position);
			vector<vector<double>> softmax_result = evaluate_softmax(m1, t1->logger);
			t3->set_matrix(new_position, softmax_result);
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < t3->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_softmax_forward(t3, t1, new_position, axis+1);
		}
		new_position.pop_back();

	}

	inline shared_ptr<Tensor> softmax_forward(shared_ptr<Tensor> t1) {
		shared_ptr<Logger> logger = t1->logger;
		shared_ptr<Tensor> t3 = make_shared<Tensor>(vector<double>(t1->values.size(), 0.), t1->shape, logger, t1, nullptr, softmax_backward);
		vector<int> new_position {};
		recurse_softmax_forward(t3, t1, new_position);
		return t3;
	}


	// Loss functions
	

	inline void recurse_cross_entropy_backward(shared_ptr<Tensor> predicted, const shared_ptr<Tensor> ground_truth, vector<int> & new_position, int axis=0) {

		if (axis == predicted->shape.size() - 2) {
			vector<vector<double>> predicted_values = predicted->get_matrix(new_position);
			vector<vector<double>> ground_truth_values = ground_truth->get_matrix(new_position);

			vector<vector<double>> loss_gradient(predicted_values.size(), vector<double>(predicted_values[0].size(), 0.));

			for (int j = 0; j < predicted_values[0].size(); j++) {
				for (int i = 0; i < predicted_values.size(); i++) {
					loss_gradient[i][j] = -1.0*ground_truth_values[i][j] / predicted_values[i][j] ;
				}
			}
			predicted->set_matrix(new_position, loss_gradient, "gradients");
			return;
		}




		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < predicted->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_cross_entropy_backward(predicted, ground_truth, new_position, axis+1);
		}
		new_position.pop_back();
		
	}


	inline void cross_entropy_backward(shared_ptr<Tensor> loss) {

		shared_ptr<Tensor> predicted = loss->input_first;
		if (!predicted) return;
		shared_ptr<Tensor> ground_truth = loss->input_second;
		vector<int> new_position {};
		recurse_cross_entropy_backward(predicted, ground_truth, new_position);
	}

	
	
	
	inline vector<double> evaluate_cross_entropy(vector<vector<double>> predicted, vector<vector<double>> ground_truth, shared_ptr<Logger> logger) {
		if (predicted.size() != ground_truth.size() || predicted[0].size() != ground_truth[0].size()) {
			logger->log(ERROR, "Matrix dimensions do not match in evaluate_cross_entropy.");
			exit(1);
		}
		vector<double> loss(predicted[0].size(), 0.);
		for (int j = 0; j < predicted[0].size(); j++) {
			double cross_entropy = 0.;
			for (int i = 0; i < predicted.size(); i++) {
				cross_entropy += ground_truth[i][j] * log(predicted[i][j]);
			}
			loss[j] = -cross_entropy;
		}
		return loss;
	}
	 
	
	inline void recurse_cross_entropy_forward(shared_ptr<Tensor> loss, const shared_ptr<Tensor> predicted, const shared_ptr<Tensor> ground_truth, vector<int> & new_position, int axis=0) {

		if (axis == loss->shape.size() - 2) {
			vector<vector<double>> m1 = predicted->get_matrix(new_position);
			vector<vector<double>> m2 = ground_truth->get_matrix(new_position);
			vector<double> cross_entropy_loss = evaluate_cross_entropy(m1, m2, predicted->logger);
			vector<vector<double>> result {cross_entropy_loss};
			loss->set_matrix(new_position, result, "values");
			return;
		}

		new_position.push_back(0);
		int position_index = new_position.size() - 1;

		for (int i = 0; i < loss->shape[axis]; i++) {
			new_position[position_index] = i;
			recurse_cross_entropy_forward(loss, predicted, ground_truth, new_position, axis+1);
		}
		new_position.pop_back();
	}


	inline shared_ptr<Tensor> categorical_cross_entropy_forward(shared_ptr<Tensor> predicted, shared_ptr<Tensor> ground_truth) {
		shared_ptr<Logger> logger = predicted ->logger;
		if (predicted->shape != ground_truth->shape) {
			logger->log(ERROR, "The shapes of the predicted and ground truth arrays are mismatched in categorical cross entropy.");
			exit(1);
		}

		vector<int> loss_shape = predicted->shape;
		int last_index = loss_shape.size() - 1;
		loss_shape[last_index - 1] = 1;

		int number_elements = 1;
		for (int i = 0; i < loss_shape.size(); i++) {
			number_elements *= loss_shape[i];
		}

		vector<double> loss_values(number_elements, 0.);
		shared_ptr<Tensor> loss = make_shared<Tensor>(loss_values, loss_shape, logger, predicted, ground_truth, cross_entropy_backward);
		vector<int> new_position {};
		recurse_cross_entropy_forward(loss, predicted, ground_truth, new_position);
		return loss;
	}

	inline void mean_squared_error_backward(shared_ptr<Tensor> loss) {
		shared_ptr<Tensor> predicted = loss -> input_first;
		shared_ptr<Tensor> ground_truth = loss -> input_second;

		for (int i = 0; i < predicted->values.size(); i++) {
			predicted->gradients[i] += 2.0 * (predicted->values[i] - ground_truth->values[i]);
		} 

	}


	inline shared_ptr<Tensor> mean_squared_error_forward(shared_ptr<Tensor> predicted, shared_ptr<Tensor> ground_truth) {
		shared_ptr<Logger> logger = predicted ->logger;
		if (predicted->shape != ground_truth->shape) {
			logger->log(ERROR, "The shapes of the predicted and ground truth arrays are mismatched in mean_squared_error_forward.");
			exit(1);
		}

		vector<int> loss_shape = predicted->shape;

		int number_elements = 1;
		for (int i = 0; i < loss_shape.size(); i++) {
			number_elements *= loss_shape[i];
		}

		vector<double> loss_values(number_elements, 0.);

		for (int i = 0; i < number_elements; i++) 
			loss_values[i] = (predicted->values[i] - ground_truth->values[i]) * (predicted->values[i] - ground_truth->values[i]);


		shared_ptr<Tensor> loss = make_shared<Tensor>(loss_values, loss_shape, logger, predicted, ground_truth, mean_squared_error_backward);
		return loss;
	}



}// namespace ml
