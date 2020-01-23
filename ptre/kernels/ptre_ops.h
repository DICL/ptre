#ifndef PTRE_KERNELS_PTRE_OPS_H_
#define PTRE_KERNELS_PTRE_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
//using CPUDevice = Eigen::ThreadPoolDevice;
//using GPUDevice = Eigen::GpuDevice;

namespace functor {
template <typename Device>
struct Modelaverage {
  void operator()(const Device& d, typename TTypes<float>::Flat var,
                  //const Tensor& other);
                  typename TTypes<float>::ConstFlat other);
                  //typename TTypes<float>::Flat other);
};

template <typename Device>
struct CopyTensorToSendBuf {
  void operator()(const Device& d,
                  typename TTypes<float>::Flat src,
                  typename TTypes<float>::Flat dst);
};

}  // namespace functor
}  // namespace tensorflow
#endif  // PTRE_KERNELS_PTRE_OPS_H_
