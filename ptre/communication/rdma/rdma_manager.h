#ifndef PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <iostream>

#include "ptre/communication/rdma/rdma.h"
//#include "ptre/communication/rdma/rdma_agg_writer.h"
//#include "ptre/communication/grpc/grpc_client_cache.h"
#include "ptre/protobuf/rdma_service.pb.h"
#include "ptre/cm/remote_variable.h"
#include "ptre/communication/push_variable.h"

#include "tensorflow/core/framework/tensor.h"

namespace ptre {

namespace {
using std::string;
using std::cout;
using std::endl;
using tensorflow::Tensor;
}  // namespace

/// RdmaManager
///
/// ibv_context
/// set memory region
/// create cq
/// create qp
class RdmaManager {
 public:
  RdmaManager(int ptre_size, int ptre_rank);
  ~RdmaManager();

  // Queue Pair Modification Functions
  void INITQP(int dst);
  void RTRQP(int dst, uint16_t remote_lid, uint32_t remote_qpn,
      uint32_t remote_psn);
  void RTSQP(int dst, uint32_t my_psn);
  void RESETQP(int dst);
  void ConnectQP(int dst, uint32_t remote_qpn);

  // Memory Region Init Functions
  void SetTrainableVariables(std::vector<RemoteVariable*>& vars,
      const std::vector<string>& names);

  /// MR management V2
  void RegisterMR(const BufType buf_type, const string& name, void* buf,
                  size_t length, int access);
  struct ibv_mr* GetMR(const BufType buf_type, const string& name);
  void SetRemoteAddress(int dst_rank, const BufType buf_type,
      const string& name, const uint64_t remote_addr, const uint32_t rkey);
  void GetRemoteAddress(int dst_rank, const BufType buf_type,
      const string& name, uint64_t* out_addr, uint32_t* out_rkey);
  int GetRemoteAccessBufInfos(std::vector<BufType>* out_buf_types,
                              std::vector<string>* out_names);
  bool IsRemoteMRSetV2(const int dst_rank, const BufType buf_type,
                       const string& name);
  void SetRemoteMRV2(const int dst_rank, const BufType buf_type,
      const string& name, const uint64_t remote_addr, const uint32_t rkey);

  // Send Buffer Management
  void InitPush(int idx);
  void SetPushReady(int idx);
  void SetPushReady(const string& var_name);
  bool IsPushReady(int idx);
  bool IsPushReady(const string& var_name);

  int RdmaRead(int dst, const BufType buf_type, const string& name,
      struct ibv_mr* read_mr, size_t read_length);
  int RdmaWrite(int dst, const BufType buf_type, const string& var_name,
      struct ibv_mr* send_mr, size_t send_length, uint32_t* imm_data = nullptr);

  void ProcessCQ();
  void Poll(int num_comps);
/// message GetRemoteAddressResponse {
///   int32 rank = 1;
///   string tensor_name = 2;
///   Channel channel = 3;
///   repeated MemoryRegion mr = 4;
/// }
  bool IsRemoteMRSet(int rank, const std::string& name);
  bool IsDlidSet(int rank) { return (dlids_.find(rank) != dlids_.end()); }
  void SetRemoteMR(int rank, const std::string& name, uint64_t remote_addr,
                   uint32_t rkey);
  void SetDlid(int rank, uint32_t lid) { dlids_.emplace(rank, lid); }
  void set_qpn(int rank, uint32_t qpn) { qpns_.emplace(rank, qpn); }
  void set_snp(int rank, uint64_t snp) { snps_.emplace(rank, snp); }
  void set_iid(int rank, uint64_t iid) { iids_.emplace(rank, iid); }
  /// Local MR with rkey
  RemoteMR GetRemoteMR(const std::string& name);
  RemoteMR GetRemoteParamMR();
  int RdmaWriteTensor(int dst_id, const std::string& name,
                      const Tensor& tensor, bool atomic_add);
#if 0
  int PushTensorAtomicAdd(int dst_rank, const std::string& name,
                          const Tensor& tensor);
  int PushTensorAtomicAddBatch(int dst_rank, const std::string& name,
                               const Tensor& tensor);
  //void PushTensor(int dst_id, const string& name, const Tensor& tensor);
  void RdmaWriteIncomingFlag(int dst_rank, bool* flag);
#endif
  bool AttemptPush(int dst_rank);

  int PushAndNotify(int dst, const string& var_name);
  int ReceivePushNotify(int dst);

  int PushTensor(int dst_rank, string name, const Tensor& tensor);
  int NotifyPushDone(int dst_rank);

  int rank() { return ptre_rank_; }
  struct ibv_context* ctx();
  struct ibv_port_attr port_attr();
  struct ibv_pd* pd();
  struct ibv_qp* qp(int dst);
  struct ibv_cq* send_cq(int dst) { return send_cqs_[dst]; }
  struct ibv_cq* recv_cq(int dst) { return recv_cqs_[dst]; }
#if 0
  uint16_t lid() { return rdma_env_.port_attr.lid; }
#endif
  void set_remote_lid(int dst, uint16_t lid);
  uint16_t remote_lid(int dst);

 private:
  // PTRE Attributes
  int ptre_size_;
  int ptre_rank_;

  // RDMA Attributes
  struct ibv_device** device_list_;
  struct ibv_context* ctx_;
  struct ibv_pd* pd_;
  struct ibv_port_attr port_attr_;
  // RDMA Completion Queues
  std::vector<struct ibv_cq*> send_cqs_;
  std::vector<struct ibv_cq*> recv_cqs_;
  // RDMA Queue Pairs
  std::vector<struct ibv_qp*> qps_;
  // RDMA Remote LIDs
  std::vector<uint16_t> remote_lids_;
  // RDMA Receive Work Request Array
  std::vector<struct ibv_recv_wr*> recv_wrs_;

  // Variables
  std::map<string, int> var_name_to_index_;
  std::vector<PushVariable*> push_variables_;
  std::mutex mu_;

  std::map<RemoteTensorId, RemoteMR> tensor_rmrs_;  // remote tensor data memory regions
  std::map<int, RemoteMR> rpmrs_;  // remote parameter memory regions
  ibv_mr* recv_in_flag_mr_;  // is_new_incoming_
  ibv_mr* send_in_flag_mr_;  // is_new_incoming_

  /// All MRs must be registered to this map, so remote client can find
  /// MRs for all type of variables.
  std::vector<string> recv_tensor_names_;
  std::map<string, int> buf_name_to_index_;
  std::vector<string> buf_names_;
  std::vector<struct ibv_mr*> buf_mrs_;
  //std::vector<struct ibv_mr*> mrs_for_remote_;
  std::map<BufType, std::map<string, struct ibv_mr*>> mrs_;
  std::map<BufType, std::map<string, int>> access_flags_;
  //std::map<string, struct ibv_mr*> mrs_;
  /// rank, buf_name
  std::map<int, std::map<string, RemoteMR>> remote_buf_addrs_;
  std::map<int, std::map<BufType, std::map<string, RemoteMR>>> rmrs_;
  //std::map<int, RdmaAggWriter*> agg_writers_;  // owned.

  std::map<std::string, ibv_mr*> recv_mrs_;
  std::map<std::string, ibv_mr*> send_mrs_;
  std::map<int, uint32_t> dlids_;
  std::map<int, uint32_t> qpns_;
  std::map<int, uint64_t> snps_;
  std::map<int, uint64_t> iids_;
  ibv_comp_channel* event_channel_;
  //ibv_cq* cq_;
  std::map<int, ibv_comp_channel*> event_channels_;
  std::map<int, bool> connected_;
  //ibv_wc wc_[MAX_CONCURRENT_WRITES * 2];
  ibv_wc wc_[QUEUE_DEPTH_DEFAULT * 2];
  //std::unordered_map<int, RdmaChannel> remotes_;
  //std::vector<Channel> channels_;
  //std::vector<MemoryRegion> mrs_;
  bool atomic_add_ = false;
  //std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;

};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_
