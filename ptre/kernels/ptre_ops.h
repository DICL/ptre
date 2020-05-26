#ifndef PTRE_KERNELS_PTRE_OPS_H_
#define PTRE_KERNELS_PTRE_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

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

}  // namespace functor
}  // namespace tensorflow
#endif  // PTRE_KERNELS_PTRE_OPS_H_
