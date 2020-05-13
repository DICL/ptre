#ifndef PTRE_CM_REMOTE_VARIABLE_H_
#define PTRE_CM_REMOTE_VARIABLE_H_

#include <mutex>

#include "ptre/cm/push_permit.h"
#include "tensorflow/core/framework/tensor.h"

namespace ptre {

class RemoteVariable {
 public:
  void StartRecv();
  void EnqueueSenderCandidate(int src_rank);
  void StopRecv();
  void SetAggState(int state);
  void Aggregate();
  int GetGlcTensor(Tensor*& out);

 private:
  std::mutex mu_;
  Tensor* tensor_;
  float* rcv_buf_;

  // State Member Variables
  int rcv_state_;
  int agg_state_;
  int agg_cnt_;

  Permit* permit_;
};

}  // namespace ptre

#endif  // PTRE_CM_REMOTE_VARIABLE_H_
