#ifndef PTRE_TENSORFLOW_KERNELS_PTRE_OPS_H_
#define PTRE_TENSORFLOW_KERNELS_PTRE_OPS_H_

#include "ptre/common/operations.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace ptre {
namespace tensorflow {

template <typename T>
using TTypes = ::tensorflow::TTypes<T>;

namespace functor {

template <typename Device, typename T>
struct Modelaverage {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar m,
                  typename TTypes<T>::ConstFlat other);
};

template <typename Device, typename T>
struct LinearWeightedAverageApprox {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar c1,
                  typename TTypes<T>::ConstFlat other,
                  typename TTypes<T>::ConstScalar c2);
};

template <typename Device, typename T>
struct CopyTensorToSendBuf {
  void operator()(const Device& d,
                  typename TTypes<T>::Flat src,
                  typename TTypes<T>::Flat dst);
};

template <typename Device, typename T>
struct CopyRemoteToVar {
  void operator()(const Device& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstFlat remote);
};

}  // namespace functor
}  // namespace tensorflow
}  // namespace ptre

#endif  // PTRE_TENSORFLOW_KERNELS_PTRE_OPS_H_
