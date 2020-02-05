#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "ptre/kernels/ptre_ops.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T>
struct Modelaverage<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar m,
                  typename TTypes<T>::ConstFlat other) {
    auto other_bytes = sizeof(T) * other.size();
    auto other_buf = d.allocate(other_bytes);
    d.memcpyHostToDevice(other_buf, other.data(), other_bytes);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> other_gpu((T*) other_buf,
        other.size());

    auto m_bytes = sizeof(T);
    auto m_buf = d.allocate(m_bytes);
    d.memcpyHostToDevice(m_buf, m.data(), m_bytes);
    typename TTypes<T>::ConstScalar m_gpu((T*) m_buf, 1);
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = var.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) = (var + other_gpu) / m_gpu.reshape(single).broadcast(bcast);
    d.deallocate(m_buf);
    d.deallocate(other_buf);
  }
};

// tensorflow/core/framework/tensor_types.h
//  typedef Eigen::TensorMap<Eigen::TensorFixedSize<const T, Eigen::Sizes<>,
//                                                  Eigen::RowMajor, IndexType>,
//                           Eigen::Aligned>
//      ConstScalar;


//template <typename T>
//struct ApplyGradientDescent<GPUDevice, T> {
//  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
//                  typename TTypes<T>::ConstScalar lr,
//                  typename TTypes<T>::ConstFlat grad) {
//    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
//    bcast[0] = grad.dimension(0);
//    Eigen::Sizes<1> single;
//    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad;
//  }
//};


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
