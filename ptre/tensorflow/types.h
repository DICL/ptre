#ifndef PTRE_TENSORFLOW_TYPES_H_
#define PTRE_TENSORFLOW_TYPES_H_

//#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace ptre {
  typedef Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> Flat;
}

#endif
