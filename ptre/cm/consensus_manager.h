#ifndef PTRE_CM_CONSENSUS_MANAGR_H_
#define PTRE_CM_CONSENSUS_MANAGR_H_

#include <vector>
#include <mutex>

#include "ptre/communication/rdma/rdma_manager.h"
#include "tensorflow/core/framework/tensor.h"

namespace ptre {

namespace {
using tensorflow::Tensor;
}  // namespace

class ConsensusManager {
 public:
  void InitGlobalConsensus(std::vector<const Tensor*>& vars);
  bool IsInitialized() { return is_initialized_; }
  void SetRdmaManager(RdmaManager* rdma_manager);
  void EnqueuePush(std::vector<const Tensor*>& vars);
  const Tensor& global_consensus(int index);
  int num_vars() { return num_vars_; }

  const std::vector<Tensor*>& GetGlobalConsensusList();
  const std::vector<Tensor*>& GetSendTensorsList();

 private:
  std::vector<Tensor*> global_consensus_;
  std::vector<Tensor*> send_tensors_;
  int num_vars_;
  bool is_initialized_ = false;

  std::mutex mu_;
  std::vector<Tensor*> for_push_;
  bool ready_to_push_ = false;
  RdmaManager* rdma_manager_;
};

}  // namespace ptre


#endif  // PTRE_CM_CONSENSUS_MANAGR_H_
