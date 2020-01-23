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
    auto other_bytes = sizeof(float) * other.size();
    auto other_buf = d.allocate(other_bytes);
    d.memcpyHostToDevice(other_buf, other.data(), other_bytes);
    Eigen::TensorMap<Eigen::Tensor<float, 1>> other_gpu((float*) other_buf,
        other.size());
    var.device(d) = var.constant(float(0.5)) * (var + other_gpu);
    d.deallocate(other_buf);
  }
};

template <>
struct CopyTensorToSendBuf<GPUDevice> {
  void operator()(const GPUDevice& d,
                  typename TTypes<float>::Flat src,
                  typename TTypes<float>::Flat dst) {
    auto bytes = sizeof(float) * src.size();
    d.memcpyDeviceToHost(dst.data(), src.data(), bytes);
  }
};
}  // namespace functor

template struct functor::Modelaverage<GPUDevice>;
template struct functor::CopyTensorToSendBuf<GPUDevice>;

int dummy_ptre_ops_gpu() {
  const GPUDevice d(nullptr, 0);
  Tensor var;
  const Tensor other;
  functor::Modelaverage<GPUDevice>()(d, var.flat<float>(), other.flat<float>());
  functor::CopyTensorToSendBuf<GPUDevice>()(d, var.flat<float>(), var.flat<float>());
  return 0;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
