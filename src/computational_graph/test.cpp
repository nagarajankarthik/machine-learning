#include "tensor.h"
#include "tensor_operations.h"
#include <memory>
#include <vector>
using namespace std;
using namespace ml;

/**
 * This file is created for the purpose of testing the code in computational_graph.
 * Compile the code as follows:
 * g++ ../utils/logging.h ../utils/logging.cpp tensor.h tensor.cpp tensor_operations.h test.cpp -std=c++20 -g -o ct.o
 * Run as follows: ./ct.o
 */

int main() {

	shared_ptr<Logger> logger = make_shared<Logger>("test_tensor.log");


	vector<int> shape = {1, 2, 2};
	vector<double> values {1., 1., 1., 1.};
	shared_ptr<Tensor> u = make_shared<Tensor>(values, shape, logger);
	logger->log(INFO, "Created tensor u");
	shape[0] = 2;
	values.insert(values.end(), {1., 1., 1., 1.});
	shared_ptr<Tensor> v = make_shared<Tensor>(values, shape, logger);
	logger->log(INFO, "Created tensor v.");
	shared_ptr<Tensor> w = batch_matmul_forward(u, v);
	logger->log(INFO, "Created tensor w by multiplying u and v.");
	vector<int> position {0};
	cout << "Shape of w: " << w->shape[0] << " by " << w->shape[1] << endl;
	vector<vector<double>> matrix = w->get_matrix(position, "values");

	logger->log(INFO, "Obtained values of tensor w.");
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}

	fill(w->gradients.begin(), w->gradients.end(), 1.);
	w->backward();
	vector<vector<double>> grad = u->get_matrix(position, "gradients");
	for (int i = 0; i < grad.size(); i++) {
		for (int j = 0; j < grad[0].size(); j++) {
			cout << grad[i][j] << " ";
		}
		cout << endl;
	}

	return 0;
}
