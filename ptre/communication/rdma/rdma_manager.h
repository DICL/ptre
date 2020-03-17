#ifndef PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <iostream>

#include "ptre/communication/rdma/rdma.h"
//#include "ptre/communication/grpc/grpc_client_cache.h"
#include "ptre/protobuf/rdma_service.pb.h"

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
  RdmaManager(int ptre_size, int ptre_rank, bool add);
  ~RdmaManager();
  /// The input tensor's buffer must be fixed.
  void InitTensorMR(int dst_id, const std::string& name,
                    Tensor* recv, Tensor* send);
  void InitParamMR(bool* is_new_incoming, bool* send_in_flag);
  void MarkMRInitialized();
  bool IsMRInitialized();
  void CreateCQs();
  void CreateQPs();
  int ConnectQP(int dst_rank);
  void ProcessCQ();
  void Poll(int num_comps);
/// message GetRemoteAddressResponse {
///   int32 rank = 1;
///   string tensor_name = 2;
///   Channel channel = 3;
///   repeated MemoryRegion mr = 4;
/// }
  bool IsRemoteMRSet(int rank, const std::string& name);
  bool IsRemoteParamMRSet(int rank);
  bool IsDlidSet(int rank) { return (dlids_.find(rank) != dlids_.end()); }
  void SetRemoteMR(int rank, const std::string& name, uint64_t remote_addr,
                   uint32_t rkey);
  void SetRemoteParamMR(int rank, uint64_t remote_addr, uint32_t rkey);
  void SetDlid(int rank, uint32_t lid) { dlids_.emplace(rank, lid); }
  void set_qpn(int rank, uint32_t qpn) { qpns_.emplace(rank, qpn); }
  void set_snp(int rank, uint64_t snp) { snps_.emplace(rank, snp); }
  void set_iid(int rank, uint64_t iid) { iids_.emplace(rank, iid); }
  RemoteMR GetRemoteMR(const std::string& name);
  RemoteMR GetRemoteParamMR();
  int RdmaWriteTensor(int dst_id, const std::string& name,
                      const Tensor& tensor, bool atomic_add);
  int PushTensorAtomicAdd(int dst_rank, const std::string& name,
                          const Tensor& tensor);
  int PushTensorAtomicAddBatch(int dst_rank, const std::string& name,
                               const Tensor& tensor);
  //void PushTensor(int dst_id, const string& name, const Tensor& tensor);
  void RdmaWriteIncomingFlag(int dst_rank, bool* flag);

  bool AttemptPush(int dst_rank);
  int PushTensor(int dst_rank, string name, const Tensor& tensor);
  int AckPushDone(int dst_rank);

  int rank() { return ptre_rank_; }
  ibv_cq* cq() { return cq_; }
  ibv_qp* qp(int dest_rank) { return qps_[dest_rank]; }
  RdmaEnv* rdma_env() { return &rdma_env_; }

 private:
  std::thread polling_thread_;
  std::mutex mu_;
  bool is_mr_initialized_ = false;

  int ptre_size_;
  int ptre_rank_;
  RdmaEnv rdma_env_;
  std::map<RemoteTensorId, RemoteMR> rmrs_;  // remote tensor data memory regions
  std::map<int, RemoteMR> rpmrs_;  // remote parameter memory regions
  ibv_mr* recv_in_flag_mr_;  // is_new_incoming_
  ibv_mr* send_in_flag_mr_;  // is_new_incoming_
  std::map<std::string, ibv_mr*> recv_mrs_;
  std::map<std::string, ibv_mr*> send_mrs_;
  std::map<int, uint32_t> dlids_;
  std::map<int, uint32_t> qpns_;
  std::map<int, uint64_t> snps_;
  std::map<int, uint64_t> iids_;
  //std::map<int, std::map<std::string, RemoteMR>> rmrs_;
  ibv_comp_channel* event_channel_;
  ibv_cq* cq_;
  std::map<int, ibv_comp_channel*> event_channels_;
  std::map<int, ibv_cq*> cqs_;
  std::map<int, ibv_qp*> qps_;
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
