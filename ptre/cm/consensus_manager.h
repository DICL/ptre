#ifndef PTRE_CM_CONSENSUS_MANAGR_H_
#define PTRE_CM_CONSENSUS_MANAGR_H_

#include <vector>
#include <map>
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
  void InitBufTensor(const std::string& name, const Tensor& tensor);
  bool IsInitialized() { return is_initialized_; }
  void SetRdmaManager(RdmaManager* rdma_manager);
  void EnqueuePushList(std::vector<const Tensor*>& vars);

  void CopyTensorSend(const std::string& name, const Tensor& tensor);
  void PushTensors(int dst_rank);
  void SetPushReady() { ready_to_push_ = true; }
  bool IsPushReady() { return ready_to_push_; }
  void UnsetPushReady() { ready_to_push_ = false; }
  int GetRandomTarget();

  const std::vector<Tensor*>& GetGlobalConsensusList();
  const std::vector<Tensor*>& GetSendTensorsList();

  const Tensor& global_consensus(int index);
  int num_vars() { return num_vars_; }
  void set_size(int size) { ptre_size_ = size; }
  void set_rank(int rank) { ptre_rank_ = rank; }
  int size() { return ptre_size_; }
  int rank() { return ptre_rank_; }
  Tensor* send_tensor(int index) { return send_tensors_list_[index]; }

 private:
  int ptre_size_;
  int ptre_rank_;

  std::vector<Tensor*> global_consensus_;
  std::map<std::string, Tensor*> recv_tensors_;
  std::map<std::string, Tensor*> send_tensors_;
  std::vector<Tensor*> send_tensors_list_;
  int num_vars_;
  bool is_initialized_ = false;

  std::mutex mu_;
  std::vector<Tensor*> for_push_;
  bool ready_to_push_ = false;
  RdmaManager* rdma_manager_;
};

}  // namespace ptre


#endif  // PTRE_CM_CONSENSUS_MANAGR_H_