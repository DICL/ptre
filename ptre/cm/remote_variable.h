#ifndef PTRE_CM_REMOTE_VARIABLE_H_
#define PTRE_CM_REMOTE_VARIABLE_H_

#include <mutex>

#include "ptre/cm/push_permit.h"
#include "tensorflow/core/framework/tensor.h"
#include "ptre/tensorflow/types.h"

namespace ptre {

using ::tensorflow::Tensor;

class RemoteVariable {
 public:
  RemoteVariable(const Tensor& var);
  void StartRecv();
  void EnqueueSenderCandidate(int src_rank);
  void StopRecv();
  void SetAggState(int state);
  void Aggregate();
  int AggCount();
  int GetGlcTensor(Tensor*& out);
  void* rcv_data();
  size_t rcv_length();
  void* permit_data();

 private:
  std::mutex mu_;
  // Storage
  Tensor* tensor_;
  // Receive Buffer
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
