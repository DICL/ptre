#ifndef PTRE_CM_CONSENSUS_MANAGR_H_
#define PTRE_CM_CONSENSUS_MANAGR_H_

#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <string>
#include <iostream>

#include "ptre/cm/peer_selector.h"
#include "ptre/cm/tensor_aggregator.h"
#include "ptre/communication/rdma/rdma_manager.h"
#include "ptre/communication/grpc/grpc_client_cache.h"
//#include "ptre/communication/tcp/tcp_manager.h"
#include "tensorflow/core/framework/tensor.h"

#define MAX_RECV_THRESHOLD 4

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
  /// NOT USED at least until 5f1352f07118881c8c5319e341fde8633905b42f
  void InitGlobalConsensus(std::vector<const Tensor*>& vars);
  int InitGlobalConsensusV2(const std::vector<string>& names,
                            const std::vector<const Tensor*>& vars);
  void InitBufTensor(const std::string& name, const Tensor& tensor);
  void InitBufParam();
  bool IsInitialized() { return is_initialized_; }
  void SetRdmaManager(RdmaManager* rdma_manager);
  void EnqueuePushList(std::vector<const Tensor*>& vars);

  void CopyTensorSend(const std::string& name, const Tensor& tensor);
  void PushModel(int dst_rank);
  void PushTensors(int dst_rank);
  void PushTensors2(int dst_rank);
  void PushTensorsV3(int dst_rank);
  void TcpPushTensors(int dst_rank);
  void SetPushReady() { ready_to_push_ = true; }
  bool IsPushReady() { return ready_to_push_; }
  void UnsetPushReady() { ready_to_push_ = false; }
  int GetRandomTarget();
  int GetIncNeighbor();
  int get_peer();
  int get_peers(int num_peer, int* peers);

  void InitPeerSelector(int strategy, int num_push);

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
  //Tensor* send_tensor(int index) { return send_tensors_list_[index]; }
  //Tensor* send_tensor(const string& name) { return send_tensors_[name]; }
  Tensor* send_tensor(int index);
  Tensor* send_tensor(const string& name);

  bool CanReceive(int src_rank, int src_vstep);
  int FinalizeRecv(int src_rank);

  /// Init recv_tensor buf and agg_done counts
  /// and all related states
  /// Must Open Receive after this preparation done
  int PrepareReceive();
  int OpenReceive();
  int CloseReceive();
  bool IsReceiveDone();
  int WaitAndGetNumIncomings();
  int CountReduceAndOpenRecv(std::string& name);
  bool IsInitNumApplyOps();
  int InitNumRecvTensors();
  int ProcessAggregation();

  void set_rcv_done_cnt(int cnt) { rcv_done_cnt_ = cnt; }

  /// Training Status
  void set_local_step(int step) { local_step_ = step; }
  void count_local_step() { local_step_++; }
  void set_virtual_step(int step) { virtual_step_ = step; }
  void count_virtual_step() { virtual_step_++; }

  /// ...
  /// V2 Element Access Functions
  void* buf_ptr(const BufType type, const string& name);
  int rcv_ing_cnt() { return rcv_ing_cnt_; }
  int rcv_steps_sum() { return rcv_steps_sum_; }
  int num_apply_ops() { return num_rcv_tensors_; }
  TensorAggregator* tensor_aggregator() { return tensor_aggregator_; }

  std::mutex send_mu_;
  std::condition_variable send_cv_;
  enum {
    SEND_IDLE,
    SEND_IN_PROGRESS
  };
  int send_status_ = SEND_IDLE;

 private:
  int ptre_size_;
  int ptre_rank_;

  /// Training Status
  int local_step_;
  int virtual_step_ = 1;

  int num_vars_;
  //std::map<string, int> name_to_index_;
  std::vector<Tensor*> global_consensus_;
  std::map<std::string, Tensor*> recv_tensors_;
  std::map<std::string, Tensor*> send_tensors_;
  std::vector<Tensor*> send_tensors_list_;
  std::vector<std::string> tensor_names_;
  std::vector<std::string> actual_comm_tensors_;
  bool is_initialized_ = false;
  bool* is_new_incoming_ = nullptr;
  bool flag_to_send_ = true;

  /// Buffers
  int num_bufs_ = 0;
  std::map<string, int> buf_name_to_index_;
  std::map<BufType, std::map<string, int>> buf_type_name_index_map_;
  std::vector<BufType> buf_types_;
  std::vector<void*> bufs_;
  std::vector<size_t> buf_lengths_;
  std::vector<uint64_t*> agg_buf_states_;
  // To be deprecated.
  std::vector<string> buf_names_;

  std::mutex rcv_mu_;
  std::condition_variable rcv_cv_;
  bool rcv_open_ = false;
  int rcv_ing_cnt_ = 0;  // num peers
  int rcv_done_cnt_ = 0;  // num peers
  int rcv_steps_sum_ = 0;
  enum {
    RECV_IN_PROGRESS,
    RECV_DONE
  };
  std::map<int, int> rcv_status_;
  int num_rcv_tensors_;
  bool is_init_num_rcv_tensors_ = false;
  int reduce_cnt_ = 0;  // num tensors
  int non_reduce_cnt_ = 0;

  std::mutex rf_mu_;
  int receiving_from = -1;
  int received_from = -1;

  PeerSelectorInterface* peer_selector_ = nullptr;

  std::mutex mu_;
  std::vector<Tensor*> for_push_;
  bool ready_to_push_ = false;
  RdmaManager* rdma_manager_ = nullptr;
  //std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;

  TensorAggregator* tensor_aggregator_ = nullptr;
};

}  // namespace ptre


#endif  // PTRE_CM_CONSENSUS_MANAGR_H_
