#include "tensor.h"

namespace ml {
	Tensor::Tensor(vector<double> values, vector<int> shape):values(values), shape(shape) {
		update_strides();
	
	} ;

	Tensor::Tensor(vector<double> values, vector<int> shape, shared_ptr<Tensor> input_first, shared_ptr<Tensor> input_second):Tensor(values, shape)  {
	
		this->input_first = input_first;
		this->input_second = input_second;

	};

	double Tensor::at(vector<int> position) {
		int index = 0;
		for (int i = 0; i < shape.size(); i++) {
			index += strides[i]*position[i];
		}
		return values[index];
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
		}
	}

	void Tensor::backward(const vector<double>& seed) {
		backward_function(seed, input_first, input_second);
	}


}// namespace ml
