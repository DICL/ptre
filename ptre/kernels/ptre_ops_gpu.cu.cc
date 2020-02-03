#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "ptre/kernels/ptre_ops.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T>
struct Modelaverage<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstFlat other) {
    auto other_bytes = sizeof(T) * other.size();
    auto other_buf = d.allocate(other_bytes);
    d.memcpyHostToDevice(other_buf, other.data(), other_bytes);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> other_gpu((T*) other_buf,
        other.size());
    var.device(d) = var.constant(T(0.5)) * (var + other_gpu);
    d.deallocate(other_buf);
  }
};

template <typename T>
struct CopyTensorToSendBuf<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Flat src,
                  typename TTypes<T>::Flat dst) {
    auto bytes = sizeof(T) * src.size();
    d.memcpyDeviceToHost(dst.data(), src.data(), bytes);
  }
};
}  // namespace functor

template struct functor::Modelaverage<GPUDevice, Eigen::half>;
template struct functor::Modelaverage<GPUDevice, float>;
template struct functor::Modelaverage<GPUDevice, double>;

template struct functor::CopyTensorToSendBuf<GPUDevice, Eigen::half>;
template struct functor::CopyTensorToSendBuf<GPUDevice, float>;
template struct functor::CopyTensorToSendBuf<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
