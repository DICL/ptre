#include "ptre/communication/rdma/rdma_manager.h"

#include <iostream>

#include "ptre/communication/rdma/rdma.h"

namespace ptre {

RdmaManager::RdmaManager(int ptre_size, int ptre_rank)
    : ptre_size_(ptre_size), ptre_rank_(ptre_rank) {
  int ret = 0;
  ret = init_rdma_env(rdma_env_);
  if (ret < 0) {
    std::cout << "init_rdma_env failed. ret=" << ret << std::endl;
  } else {
    std::cout << "init_rdma_env done." << std::endl;
  }
}

void RdmaManager::InitTensorMR(int dst_id, const std::string& name,
                               const Tensor& recv, const Tensor& send) {
  tensorflow::StringPiece data;
  size_t length;
  void* addr;
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  ibv_mr* mr;
  /// Set tensor MR for recv buf
  /// Remote nodes use this MR as their own RemoteMR to perform rdma write
  /// operations.
  data = recv.tensor_data();
  length = data.size();
  addr = (void*) data.begin();
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  recv_mrs_.emplace(name, mr);
  std::cout << "RecvMR is set for name=" << name << ", addr=" << addr <<
            ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;

  /// Set tensor MR for send buf to perform rdma write operations on remote
  /// nodes.
  data = send.tensor_data();
  length = data.size();
  addr = (void*) data.begin();
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  send_mrs_.emplace(name, mr);
  std::cout << "SendMR is set for name=" << name << ", addr=" << addr <<
            ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;
}

//void RdmaManager::InitTensorMRs(int dst_id, const std::string& name,
//                                const Tensor& recv, const Tensor& send) {

void RdmaManager::MarkMRInitialized() {
  std::lock_guard<std::mutex> guard(mu_);
  is_mr_initialized_ = true;
}

bool RdmaManager::IsMRInitialized() {
  std::lock_guard<std::mutex> guard(mu_);
  return is_mr_initialized_;
}

bool RdmaManager::IsRemoteMRSet(int rank, const std::string& name) {
  RemoteTensorId id{ rank, name };
  return (rmrs_.find(id) != rmrs_.end());
}

void RdmaManager::SetRemoteMR(int rank, const std::string& name,
                              uint64_t remote_addr, uint32_t rkey) {
  rmrs_.emplace(RemoteTensorId{ rank, name }, RemoteMR { remote_addr, rkey });
  std::cout << "RemoteMR is set for rank=" << rank << ", name=" << name <<
            ", remote_addr=" << (void*) remote_addr << ", rkey=" << rkey << std::endl;
}

RemoteMR RdmaManager::GetRemoteMR(const std::string& name) {
  auto mr = recv_mrs_[name];
  uint64_t remote_addr = (uint64_t) mr->addr;
  uint32_t rkey = mr->rkey;
  return RemoteMR{ remote_addr, rkey };
}

void RdmaManager::RdmaWriteTensor(int dst_id, const std::string& name,
                                  const Tensor& tensor) {
  auto data = tensor.tensor_data();
  size_t buffer_size = (size_t) tensor.TotalBytes();
  size_t buf_size_from_stringview = data.size();
  uint64_t src_addr = (uint64_t) data.begin();
  struct ibv_mr *mr = send_mrs_[name];
  uint32_t lkey = mr->lkey;

  RemoteMR rmr = rmrs_[RemoteTensorId{ dst_id, name }];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_id];
  uint64_t wr_id = (uint64_t) new RdmaWriteID(RDMA_WRITE_ID_TENSOR_WRITE,
                                              nullptr);
  int ret = post_write(buffer_size, src_addr, lkey, remote_addr, rkey, wr_id,
                       qp);
  if (!ret) {
    std::cerr << "post_write failed." << std::endl;
  }
}

}  // namespace ptre
