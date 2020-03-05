#ifndef PTRE_CM_CONSENSUS_MANAGR_H_
#define PTRE_CM_CONSENSUS_MANAGR_H_

#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <string>
#include <iostream>

#include "ptre/cm/peer_selector.h"
#include "ptre/communication/rdma/rdma_manager.h"
#include "ptre/communication/grpc/grpc_client_cache.h"
//#include "ptre/communication/tcp/tcp_manager.h"
#include "tensorflow/core/framework/tensor.h"

namespace ptre {
using std::string;
using std::cout;
using std::endl;

namespace {
using tensorflow::Tensor;
}  // namespace

class ConsensusManager {
 public:
  ~ConsensusManager();
  void InitGlobalConsensus(std::vector<const Tensor*>& vars);
  void InitBufTensor(const std::string& name, const Tensor& tensor);
  void InitBufParam();
  bool IsInitialized() { return is_initialized_; }
  void SetRdmaManager(RdmaManager* rdma_manager);
  void EnqueuePushList(std::vector<const Tensor*>& vars);

  void CopyTensorSend(const std::string& name, const Tensor& tensor);
  void PushModel(int dst_rank);
  void PushTensors(int dst_rank);
  void PushTensors2(int dst_rank);
  void TcpPushTensors(int dst_rank);
  void SetPushReady() { ready_to_push_ = true; }
  bool IsPushReady() { return ready_to_push_; }
  void UnsetPushReady() { ready_to_push_ = false; }
  int GetRandomTarget();
  int GetIncNeighbor();
  int get_peer();
  int get_peers(int num_peer, int* peers);

  void InitPeerSelector(int strategy);

  const std::vector<Tensor*>& GetGlobalConsensusList();
  const std::vector<Tensor*>& GetSendTensorsList();

  const Tensor& global_consensus(int index);
  const Tensor& global_consensus(const std::string& name);
  int num_vars() { return num_vars_; }
  void set_size(int size) { ptre_size_ = size; }
  void set_rank(int rank) { ptre_rank_ = rank; }
  int size() { return ptre_size_; }
  int rank() { return ptre_rank_; }
  bool* is_new_incoming_ptr() { return is_new_incoming_; }
  void MarkNoNew() { *is_new_incoming_ = false; }
  Tensor* send_tensor(int index) { return send_tensors_list_[index]; }
  Tensor* send_tensor(const string& name) { return send_tensors_[name]; }

  bool CanReceive(int src_rank);
  int FinalizeRecv(int src_rank);

  int OpenReceive();
  int CloseReceive();
  bool IsReceiveDone();
  int GetNumIncomingsOrWait();

 private:
  int ptre_size_;
  int ptre_rank_;

  std::vector<Tensor*> global_consensus_;
  std::map<std::string, Tensor*> recv_tensors_;
  std::map<std::string, Tensor*> send_tensors_;
  std::vector<Tensor*> send_tensors_list_;
  int num_vars_;
  bool is_initialized_ = false;
  bool* is_new_incoming_ = nullptr;
  bool flag_to_send_ = true;

  std::mutex rcv_mu_;
  std::condition_variable rcv_cv_;
  bool rcv_open_ = true;
  int rcv_ing_cnt_;
  int rcv_done_cnt_;
  enum {
    RECV_IN_PROGRESS,
    RECV_DONE
  };
  std::map<int, int> rcv_status_;

  std::mutex rf_mu_;
  int receiving_from = -1;
  int received_from = -1;

  PeerSelectorInterface* peer_selector_ = nullptr;

  std::mutex mu_;
  std::vector<Tensor*> for_push_;
  bool ready_to_push_ = false;
  RdmaManager* rdma_manager_ = nullptr;
  //std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;
};

}  // namespace ptre


#endif  // PTRE_CM_CONSENSUS_MANAGR_H_
