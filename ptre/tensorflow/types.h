#ifndef PTRE_TENSORFLOW_TYPES_H_
#define PTRE_TENSORFLOW_TYPES_H_

//#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace ptre {
  typedef Eigen::TensorMap<
      Eigen::Tensor<float, 1, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
            Flat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const float, 1, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
            ConstFlat;
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<float, Eigen::Sizes<>, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
            Scalar;
  typedef Eigen::TensorMap<
      Eigen::TensorFixedSize<const float, Eigen::Sizes<>, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
            ConstScalar;
}

#endif
