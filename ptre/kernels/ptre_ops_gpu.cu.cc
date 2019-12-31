#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "ptre/kernels/ptre_ops.h"
//#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <>
struct Modelaverage<GPUDevice> {
  void operator()(const GPUDevice& d,
                  typename TTypes<float>::Flat var,
                  typename TTypes<float>::ConstFlat other) {
    var.device(d) = var.constant(float(0.5)) * (var + other);
  }
};
}  // namespace functor

template struct functor::Modelaverage<GPUDevice>;

int dummy_ptre_ops_gpu() {
  const GPUDevice d(nullptr, 0);
  Tensor var;
  const Tensor other;
  //struct Modelaverage<GPUDevice> f;
  //f()(d, var.flat<float>(), other.flat<float>());
  functor::Modelaverage<GPUDevice>()(d, var.flat<float>(), other.flat<float>());
  return 0;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
