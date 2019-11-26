#include "ptre/cm/consensus_manager.h"

namespace ptre {

void ConsensusManager::InitGlobalConsensus(std::vector<const Tensor*>& vars) {
  int num_vars = vars.size();
  for (int i = 0; i < num_vars; i++) {
    const Tensor* var = vars[i];
    Tensor* tensor = new Tensor(var->dtype(), var->shape());
    std::copy(var->tensor_data().begin(), var->tensor_data().end(),
              const_cast<char*>(tensor->tensor_data().begin()));
    global_consensus_.push_back(tensor);
    send_tensors_.push_back(new Tensor(var->dtype(), var->shape()));
  }
  num_vars_ = num_vars;
  is_initialized_ = true;
}

void ConsensusManager::SetRdmaManager(RdmaManager* rdma_manager) {
  rdma_manager_ = rdma_manager;
}

void ConsensusManager::EnqueuePush(std::vector<const Tensor*>& vars) {
  //std::lock_guard<std::mutex> guard(mu_);
  int num_vars = vars.size();
  for (int i = 0; i < num_vars; i++) {
    const Tensor* var = vars[i];
    Tensor* tensor = new Tensor(var->dtype(), var->shape());
    std::copy(var->tensor_data().begin(), var->tensor_data().end(),
              const_cast<char*>(tensor->tensor_data().begin()));
    for_push_.push_back(tensor);
  }
  ready_to_push_ = true;
}

const Tensor& ConsensusManager::global_consensus(int index) {
  const Tensor& tensor = *(global_consensus_[index]);
  return tensor;
}

const std::vector<Tensor*>& ConsensusManager::GetGlobalConsensusList() {
  return global_consensus_;
}

const std::vector<Tensor*>& ConsensusManager::GetSendTensorsList() {
  return send_tensors_;
}

}  // namespace ptre
