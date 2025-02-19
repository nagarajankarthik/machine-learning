#include "tensor.h"

namespace ml {
	Tensor::Tensor(vector<double> values):values(values) {} ;


	Tensor::Tensor(vector<double> values, shared_ptr<Tensor> input_first, shared_ptr<Tensor> input_second):values(values), input_first(input_first), input_second(input_second) {};
}// namespace ml
