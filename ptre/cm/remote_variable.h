#ifndef PTRE_CM_REMOTE_VARIABLE_H_
#define PTRE_CM_REMOTE_VARIABLE_H_

#define EIGEN_USE_THREADS

#include <mutex>

#include "ptre/cm/push_permit.h"
#include "ptre/core/allocator.h"
#include "ptre/tensorflow/types.h"
#include "tensorflow/core/framework/tensor.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace ptre {

using ::tensorflow::Tensor;

class RemoteVariable {
 public:
  RemoteVariable(const Tensor& var);
  RemoteVariable(const Tensor& var, Allocator* a);
  void StartRecv();
  int EnqueueSenderCandidate(int src_rank);
  void StopRecv();
  void NewIncoming(int src_rank);
  void SetAggState(int state);
  void Aggregate();
  void AggregateEigenDevice(const Eigen::ThreadPoolDevice& d);
  int AggCount();
  int GetGlcTensor(Tensor*& out);
  void* rcv_data();
  size_t rcv_length();
  int agg_state();
  int agg_count();
  Tensor* tensor();
  int permit();
  void* permit_data();

 private:
  std::mutex mu_;
  // Storage
  Tensor* tensor_;
  // Receive Buffer
  Tensor* rcv_tensor_;
  void* rcv_buf_;
  size_t rcv_length_;

  // State Member Variables
  int rcv_state_;
  int agg_state_;
  int agg_cnt_;

  Permit* permit_;
};

}  // namespace ptre

#endif  // PTRE_CM_REMOTE_VARIABLE_H_
