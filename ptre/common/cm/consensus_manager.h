#ifndef PTRE_COMMON_CM_CONSENSUS_MANAGR_H_
#define PTRE_COMMON_CM_CONSENSUS_MANAGR_H_

#include <atomic>
#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <string>
#include <iostream>

#include "ptre/common/cm/peer_selector.h"
#include "ptre/common/cm/ready_tensor.h"
//#include "ptre/common/cm/remote_variable.h"
//#include "ptre/common/cm/tensor_aggregator.h"
#include "ptre/common/communication/rdma/rdma_mgr.h"
#include "ptre/common/communication/grpc/grpc_client_cache.h"
//#include "ptre/common/communication/tcp/tcp_manager.h"
#include "ptre/core/allocator.h"

#include "tensorflow/core/framework/tensor.h"

#define MAX_RECV_THRESHOLD 4

namespace ptre {
namespace common {

using std::string;
using std::cout;
using std::endl;

using ::tensorflow::Tensor;

enum CommbufState {
  COMMBUF_STATE_IDLE,
  COMMBUF_STATE_BUSY
};

class ConsensusManager {
 public:
  ConsensusManager(int ptre_size, int ptre_rank,
    const std::vector<const Tensor*>& vars, const std::vector<string>& names);
  ~ConsensusManager();
  /// NOT USED at least until 5f1352f07118881c8c5319e341fde8633905b42f
  void InitGlobalConsensus(std::vector<const Tensor*>& vars);
#if 0
  int InitGlobalConsensusV2(const std::vector<string>& names,
                            const std::vector<const Tensor*>& vars);
  void InitBufTensor(const std::string& name, const Tensor& tensor);
  void InitBufParam();
#endif
  bool IsInitialized() { return is_initialized_; }
  void SetRdmaMgr(RdmaMgr* rdma_mgr);
  void EnqueuePushList(std::vector<const Tensor*>& vars);

  void CopyTensorSend(const std::string& name, const Tensor& tensor);
#if 0
  void PushModel(int dst_rank);
  void PushTensors(int dst_rank);
  void PushTensors2(int dst_rank);
  void PushTensorsV3(int dst_rank);
#endif
  void TcpPushTensors(int dst_rank);
  void SetPushReady() { ready_to_push_ = true; }
  bool IsPushReady() { return ready_to_push_; }
  void UnsetPushReady() { ready_to_push_ = false; }
  int get_peer();
  void next_peer();
  int get_peers(int num_peer, int* peers);

  void InitPeerSelector(int strategy, int num_push);

  const std::vector<Tensor*>& GetGlobalConsensusList();
  const std::vector<Tensor*>& GetSendTensorsList();
  const std::vector<string>& GetGlcNameList();
  int GetGlcTensor(const int& idx, Tensor*& out);
  int GetGlcTensor(const string& var_name, Tensor*& out);

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
#if 0
  int PrepareReceive();
  int OpenReceive();
#endif
  void OpenReceive(int idx);
  void OpenReceive(const string& var_name);
  int CloseReceive();
  void CloseReceive(int idx);
  void CloseReceive(const string& var_name);
  bool IsReceiveDone();
#if 0
  int WaitAndGetNumIncomings();
#endif
  int GetNumIncomings(int idx);
  int GetNumIncomings(const string& var_name);
  int GetNumIncomings();
#if 0
  int CountReduceAndOpenRecv(std::string& name);
  void CountReduce(int idx);
  void CountReduce(const string& var_name);
  bool IsInitNumApplyOps();
  int InitNumRecvTensors();
  int ProcessAggregation();
#endif
  void ReceivePushNotify(int dst);
  void ProcessAggregation(int idx);
  int ProcessReceive();

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
  //TensorAggregator* tensor_aggregator() { return tensor_aggregator_; }
  //RemoteVariable* remote_variable(int idx);
  //RemoteVariable* remote_variable(const string& var_name);
  //std::vector<RemoteVariable*>& remote_variables();
  ReadyTensor* ready_tensor(int idx);
  ReadyTensor* ready_tensor(const string& var_name);
  const std::vector<string>& variable_names();
  int var_name_to_index(const string& var_name);

  std::mutex send_mu_;
  std::condition_variable send_cv_;
  enum {
    SEND_IDLE,
    SEND_IN_PROGRESS
  };
  int send_status_ = SEND_IDLE;
  std::atomic<int> commbuf_state_;

 private:
  int ptre_size_;
  int ptre_rank_;

  // Initialization Status

  /// Training Infos
  int local_step_;
  int virtual_step_ = 1;

  // Allocator
  Allocator* allocator_;

  // Variables
  int num_vars_;
  std::vector<string> var_names_;
  std::map<string, int> var_name_to_index_;
  //std::vector<RemoteVariable*> remote_variables_;
  std::vector<ReadyTensor*> ready_tensors_;
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

  /// Data Buffers
  int num_bufs_ = 0;
  std::map<string, int> buf_name_to_index_;
  std::map<BufType, std::map<string, int>> buf_type_name_index_map_;
  std::vector<BufType> buf_types_;
  std::vector<void*> bufs_;
  std::vector<size_t> buf_lengths_;
  std::vector<uint64_t*> agg_buf_states_;
  // To be deprecated.
  std::vector<string> buf_names_;

  // Receive lock with variable granularity
  std::vector<std::mutex> var_rcv_mus_;
  std::vector<std::condition_variable> var_rcv_cvs_;
  /// 0: Closed
  /// 1: Open
  int* var_rcv_doors_;
  int* var_rcv_ing_cnts_;
  int* var_rcv_done_cnts_;
  int* var_reduce_dones_;

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
  RdmaMgr* rdma_mgr_ = nullptr;
  //std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;

  //TensorAggregator* tensor_aggregator_ = nullptr;
  /// size = num_trainable_vars
#if 0
  std::vector<Flat*> glc_flats_;
  std::vector<Flat*> agg_flats_;
  std::vector<int> var_agg_done_cnts_;
  std::vector<int> recv_status_;
#endif
  /// size = comm_size
};

}  // namespace common
}  // namespace ptre


#endif  // PTRE_COMMON_CM_CONSENSUS_MANAGR_H_
