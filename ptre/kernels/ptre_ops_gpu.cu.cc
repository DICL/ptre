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
    /*
    T first_elem[17];
    void* src_ptr = var.data();
    d.memcpyDeviceToHost(&first_elem, static_cast<const T*>(src_ptr),
        sizeof(T) * 17);
    LOG(INFO) << "\n"
        << "var[0]=" << first_elem[0] << ", other[0]=" << other(0) << "\n"
        << "var[15]=" << first_elem[15] << ", other[15]=" << other(15) << "\n"
        << "var[16]=" << first_elem[16] << ", other[16]=" << other(16) << "\n"
        << "m=" << m();
    */

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

    /*
    d.memcpyDeviceToHost(&first_elem, static_cast<const T*>(src_ptr),
        sizeof(T) * 17);
    LOG(INFO) << "\n"
        << "result[0]=" << first_elem[0] << "\n"
        << "result[15]=" << first_elem[15] << "\n"
        << "result[16]=" << first_elem[16] << "\n"
        << "m=" << m();
    */
  }
};

template <typename T>
struct LinearWeightedAverageApprox<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar c1,
                  typename TTypes<T>::ConstFlat other,
                  typename TTypes<T>::ConstScalar c2) {

    auto other_bytes = sizeof(T) * other.size();
    auto other_buf = d.allocate(other_bytes);
    d.memcpyHostToDevice(other_buf, other.data(), other_bytes);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> other_gpu((T*) other_buf,
        other.size());

    auto c1_bytes = sizeof(T);
    auto c1_buf = d.allocate(c1_bytes);
    d.memcpyHostToDevice(c1_buf, c1.data(), c1_bytes);
    typename TTypes<T>::ConstScalar c1_gpu((T*) c1_buf, 1);

    auto c2_bytes = sizeof(T);
    auto c2_buf = d.allocate(c2_bytes);
    d.memcpyHostToDevice(c2_buf, c2.data(), c2_bytes);
    typename TTypes<T>::ConstScalar c2_gpu((T*) c2_buf, 1);

    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = var.dimension(0);
    Eigen::Sizes<1> single;

    var.device(d) = var * c1_gpu.reshape(single).broadcast(bcast)
                  + other_gpu * c2_gpu.reshape(single).broadcast(bcast);
    d.deallocate(c1_buf);
    d.deallocate(c2_buf);
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
    /*
    T first_elem[17];  // 64 + 4 bytes
    void* src_ptr = src.data();
    d.memcpyDeviceToHost(&first_elem, static_cast<const T*>(src_ptr),
        sizeof(T) * 17);
    */

    auto bytes = sizeof(T) * src.size();
    d.memcpyDeviceToHost(dst.data(), src.data(), bytes);


    /*
    LOG(INFO) << "\n"
        << "var[0]=" << first_elem[0] << ", send[0]=" << dst(0) << "\n"
        << "var[15]=" << first_elem[15] << ", send[15]=" << dst(15) << "\n"
        << "var[16]=" << first_elem[16] << ", send[16]=" << dst(16);
    */
  }
};
}  // namespace functor

template struct functor::Modelaverage<GPUDevice, Eigen::half>;
template struct functor::Modelaverage<GPUDevice, float>;
template struct functor::Modelaverage<GPUDevice, double>;

template struct functor::LinearWeightedAverageApprox<GPUDevice, Eigen::half>;
template struct functor::LinearWeightedAverageApprox<GPUDevice, float>;
template struct functor::LinearWeightedAverageApprox<GPUDevice, double>;

template struct functor::CopyTensorToSendBuf<GPUDevice, Eigen::half>;
template struct functor::CopyTensorToSendBuf<GPUDevice, float>;
template struct functor::CopyTensorToSendBuf<GPUDevice, double>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
