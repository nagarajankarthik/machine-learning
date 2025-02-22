#include "tensor.h"

namespace ml {
	Tensor::Tensor(vector<double> values, vector<int> shape, shared_ptr<Logger> logger):values(values), shape(shape), logger(logger) {
		update_strides();
	
	} ;

	Tensor::Tensor(vector<double> values, vector<int> shape, shared_ptr<Logger> logger, shared_ptr<Tensor> input_first, shared_ptr<Tensor> input_second):Tensor(values, shape, logger)  {
	
		this->input_first = input_first;
		this->input_second = input_second;

	};

	Tensor::Tensor(vector<double> values, vector<int> shape, shared_ptr<Logger> logger, shared_ptr<Tensor> input_first, shared_ptr<Tensor> input_second, function<void(const vector<double>&, shared_ptr<Tensor>, shared_ptr<Tensor>)> backward_function):Tensor(values, shape, logger, input_first, input_second) {
		this->backward_function = backward_function;
	};



	double Tensor::at(vector<int> position) {
		int index = 0;
		for (int i = 0; i < shape.size(); i++) {
			index += strides[i]*position[i];
		}
		return values[index];
	}

	vector<vector<double>> Tensor::get_matrix_at(vector<int> position) const {
		if (position.size() != shape.size() - 2) {
			logger->log(ERROR, "The specified position has " + to_string(position.size()) + " indices, but the tensor has " + to_string(shape.size()) + " dimensions");
			exit(1);
		}

		int index = 0;
		for (int i = 0; i < shape.size(); i++) {
			index += strides[i]*position[i];
		}
		int number_rows = shape[shape.size() - 2];
		int number_cols = shape[shape.size() - 1];
		vector<double> tmp(*shape.rbegin(), 0.);
		vector<vector<double>> matrix(*(++shape.rbegin()), tmp);

		for (int i = 0; i < number_rows; i++) {
			for (int j = 0; j < number_cols; j++) {
				matrix[i][j] = values[index + i*number_cols + j];
			}
		}
		return matrix;
	}

	void Tensor::set_matrix_at(vector<int> position,const vector<vector<double>>& matrix) {
		if (position.size() != shape.size() - 2) {
			logger->log(ERROR, "The specified position has " + to_string(position.size()) + " indices, but the tensor has " + to_string(shape.size()) + " dimensions");
			exit(1);
		}

		int index = 0;
		for (int i = 0; i < shape.size(); i++) {
			index += strides[i]*position[i];
		}
		int number_rows = shape[shape.size() - 2];
		int number_cols = shape[shape.size() - 1];

		if (matrix.size() != number_rows || matrix[0].size() != number_cols) {
			logger->log(ERROR, "The matrix specified has size " + to_string(matrix.size()) + " by " + to_string(matrix[0].size()) + " but the tensor has " + to_string(number_rows) + " by " + to_string(number_cols));
			exit(1);
		}

		for (int i = 0; i < number_rows; i++) {
			for (int j = 0; j < number_cols; j++) {
				values[index + i*number_cols + j] = matrix[i][j];
			}
		}
	}


	void Tensor::update_strides() {
		strides.clear();
		int product = 1;
		strides.push_back(product);
		for (auto it = shape.rbegin(); it != --shape.rend(); it++) {
			product *= *it;
			strides.push_back(product);
		}
		strides.assign(strides.rbegin(), strides.rend());
	}

	void Tensor::reshape(vector<int> new_shape) {
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) {
			new_size *= new_shape[i];
		}
		if (new_size == values.size()) {
			shape = new_shape;
			update_strides();
		} else logger->log(ERROR, "Reshape error: new size does not match current size");
	}

	void Tensor::backward(const vector<double>& seed) {
		backward_function(seed, input_first, input_second);
	}


}// namespace ml
