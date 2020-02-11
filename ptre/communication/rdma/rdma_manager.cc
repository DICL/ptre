#include "ptre/communication/rdma/rdma_manager.h"

#include <iostream>

#include "ptre/communication/rdma/rdma.h"

namespace ptre {

RdmaManager::RdmaManager(int ptre_size, int ptre_rank, bool add)
    : ptre_size_(ptre_size), ptre_rank_(ptre_rank), atomic_add_(add) {
  int ret = 0;
  ret = init_rdma_env(rdma_env_);
  if (ret < 0) {
    std::cout << "init_rdma_env failed. ret=" << ret << std::endl;
  } else {
    std::cout << "init_rdma_env done." << std::endl;
  }
  CreateCQs();
  CreateQPs();
  //polling_thread_ = std::thread([this] { ProcessCQ(); });
}

RdmaManager::~RdmaManager() {
  if (polling_thread_.joinable()) {
    polling_thread_.join();
  }
}

void RdmaManager::InitTensorMR(int dst_rank, const std::string& name,
                               Tensor* recv, Tensor* send) {
  tensorflow::StringPiece strpc;
  size_t length;
  void* addr;
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  ibv_mr* mr;
  /// Set tensor MR for recv buf
  /// Remote nodes use this MR as their own RemoteMR to perform rdma write
  /// operations.
  strpc = recv->tensor_data();
  length = strpc.size();
  addr = (void*) strpc.data();
  //*((float*) addr) = 0;
  //*((float*) const_cast<char*>(addr)) = 0;
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  recv_mrs_.emplace(name, mr);
  std::cout << "RecvMR is set for name=" << name << ", addr=" << addr <<
            ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;

  /// Set tensor MR for send buf to perform rdma write operations on remote
  /// nodes.
  strpc = send->tensor_data();
  length = strpc.size();
  addr = (void*) strpc.data();
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  send_mrs_.emplace(name, mr);
  std::cout << "SendMR is set for name=" << name << ", addr=" << addr <<
            ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;
}

void RdmaManager::InitParamMR(bool* is_new_incoming,
                              bool* send_in_flag) {
  size_t length;
  void* addr;
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  ibv_mr* mr;
  length = sizeof(bool);
  addr = (void*) is_new_incoming;
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  recv_in_flag_mr_ = mr;
  std::cout << "RecvParamMR is set for addr=" << addr <<
            ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;

  /// Set tensor MR for send buf to perform rdma write operations on remote
  /// nodes.
  addr = (void*) send_in_flag;
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  send_in_flag_mr_ = mr;
  std::cout << "SendParamMR is set for addr=" << addr <<
            ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;
}

void RdmaManager::CreateCQs() {
  event_channel_ = ibv_create_comp_channel(rdma_env_.context);
  if (!event_channel_) {
    std::cout << "Failed to create completion channel" << std::endl;
  }
  cq_ = ibv_create_cq(rdma_env_.context, MAX_CONCURRENT_WRITES * 2,
                             nullptr, event_channel_, 0);
  if (cq_ == nullptr) {
    std::cout << "Failed to create CQ" << std::endl;
  }
  //for (int i = 0; i < ptre_size_; i++) {
  //  if (i == ptre_rank_) {
  //    continue;
  //  }
  //  ibv_cq* cq = ibv_create_cq(rdma_env_.context, rdma_env_.dev_attr.max_cqe,
  //                             nullptr, nullptr, 0);
  //  if (cq == nullptr) {
  //    std::cout << "Failed to create CQ for rank=" << i << std::endl;
  //  }
  //  cqs_.emplace(i, cq);
  //}
}

void RdmaManager::CreateQPs() {
  for (int i = 0; i < ptre_size_; i++) {
    if (i == ptre_rank_) {
      continue;
    }

    /// Create QP
    ibv_qp* qp;
    {
      struct ibv_qp_init_attr qp_init_attr;
      memset(&qp_init_attr, 0, sizeof(ibv_qp_init_attr));
      qp_init_attr.send_cq = cq_;
      qp_init_attr.recv_cq = cq_;
      qp_init_attr.cap.max_send_wr = QUEUE_DEPTH_DEFAULT;
      qp_init_attr.cap.max_recv_wr = QUEUE_DEPTH_DEFAULT;
      qp_init_attr.cap.max_send_sge = 1;
      qp_init_attr.cap.max_recv_sge = 1;
      qp_init_attr.qp_type = IBV_QPT_RC;

      qp = ibv_create_qp(rdma_env_.pd, &qp_init_attr);
      if (qp == nullptr) {
        std::cout << "Failed to create QP" << std::endl;
      }
    }

    /// Init QP
    {
      struct ibv_qp_attr attr;
      memset(&attr, 0, sizeof(ibv_qp_attr));
      attr.qp_state = IBV_QPS_INIT;
      attr.pkey_index = 0;
      attr.port_num = IB_PORT;
      attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

      int attr_mask =
          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
      int r = ibv_modify_qp(qp, &attr, attr_mask);
      if (r < 0) {
        std::cout << "Failed to set QP to INIT" << std::endl;
      }
    }

    qps_.emplace(i, qp);
    connected_.emplace(i, false);
  }
}

int RdmaManager::ConnectQP(int dst_rank) {
  ibv_qp* qp = qps_[dst_rank];
  if (!connected_[dst_rank]) {
    /// RTR
    {
      struct ibv_qp_attr attr;
      memset(&attr, 0, sizeof(ibv_qp_attr));
      attr.qp_state = IBV_QPS_RTR;

      attr.path_mtu = IBV_MTU_4096;
      attr.dest_qp_num = qpns_[dst_rank];
      attr.rq_psn = 0;
      attr.max_dest_rd_atomic = 1;
      attr.min_rnr_timer = 12;
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.dgid.global.subnet_prefix = snps_[dst_rank];
      attr.ah_attr.grh.dgid.global.interface_id = iids_[dst_rank];
      attr.ah_attr.grh.flow_label = 0;
      attr.ah_attr.grh.hop_limit = 255;
      attr.ah_attr.dlid = dlids_[dst_rank];
      attr.ah_attr.sl = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.port_num = IB_PORT;

      int r = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC |
                            IBV_QP_MIN_RNR_TIMER);
      if (r < 0) {
        std::cout << "Failed to ibv_modify_qp to RTR " << r << std::endl;
        return r;
      }
    }

    /// RTS
    {
      struct ibv_qp_attr attr;
      memset(&attr, 0, sizeof(ibv_qp_attr));
      attr.qp_state = IBV_QPS_RTS;
      attr.sq_psn = 0;
      attr.timeout = TIMEOUT_DEFAULT;
      attr.retry_cnt = RETRY_CNT_DEFAULT;
      attr.rnr_retry = 7; /* infinite */
      attr.max_rd_atomic = 1;

      int r = ibv_modify_qp(qp, &attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC);
      if (r < 0) {
        std::cout << "Failed to ibv_modify_qp to RTS " << r << std::endl;
        return r;
      }
    }
    connected_[dst_rank] = true;
    std::cout << "ConnectQP done." << std::endl;
  }
  return 0;
}

void RdmaManager::ProcessCQ() {
  std::cout << "Start ProcessCQ()" << std::endl;
  while (true) {
    ibv_cq* cq;
    void* cq_context;
    int r = ibv_get_cq_event(event_channel_, &cq, &cq_context);
    if (r < 0) {
      std::cout << "Failed to ibv_get_cq_event" << std::endl;
    }
    ibv_ack_cq_events(cq, 1);
    r = ibv_req_notify_cq(cq_, 0);
    if (r < 0) {
      std::cout << "Failed to ibv_req_notify_cq" << std::endl;
    }
    int ne = ibv_poll_cq(cq_, MAX_CONCURRENT_WRITES * 2,
                         static_cast<ibv_wc*>(wc_));
    for (int i = 0; i < ne; i++) {
      if (wc_[i].status != IBV_WC_SUCCESS) {
        std::cout << "Failed status \n"
          << ibv_wc_status_str(wc_[i].status) << " " << wc_[i].status << " "
          << static_cast<int>(wc_[i].wr_id) << " " << wc_[i].vendor_err;
      } else {
        std::cout << "work completion for opcode=" << wc_[i].opcode << std::endl;
      }
      //if (wc_[i].opcode == IBV_WC_RDMA_WRITE) {
      //}
      if (true) {
        RdmaWriteID* wr_id = reinterpret_cast<RdmaWriteID*>(wc_[i].wr_id);
        delete wr_id;
      }
      //std::cout << "wc opcode=" << wc_[i].opcode << std::endl;
    }
  }
}

void RdmaManager::Poll(int num_comps) {
  struct ibv_wc wcs[num_comps];
  ptre_poll_cq(cq_, num_comps, wcs);
}

//void RdmaManager::InitTensorMRs(int dst_rank, const std::string& name,
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

bool RdmaManager::IsRemoteParamMRSet(int rank) {
  return (rpmrs_.find(rank) != rpmrs_.end());
}

void RdmaManager::SetRemoteParamMR(int rank, uint64_t remote_addr,
                                   uint32_t rkey) {
  rpmrs_.emplace(rank, RemoteMR { remote_addr, rkey });
  std::cout << "RemoteMR is set for rank=" << rank << ", name=is_new_incoming" <<
            ", remote_addr=" << (void*) remote_addr << ", rkey=" << rkey << std::endl;
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

RemoteMR RdmaManager::GetRemoteParamMR() {
  auto mr = recv_in_flag_mr_;
  uint64_t remote_addr = (uint64_t) mr->addr;
  uint32_t rkey = mr->rkey;
  return RemoteMR{ remote_addr, rkey };
}

void RdmaManager::RdmaWriteTensor(int dst_rank, const std::string& name,
                                  const Tensor& tensor) {
  auto data = tensor.tensor_data();
  size_t buffer_size = (size_t) tensor.TotalBytes();
  //size_t buf_size_from_stringview = data.size();
  uint64_t src_addr = (uint64_t) data.begin();
  struct ibv_mr *mr = send_mrs_[name];
  uint32_t lkey = mr->lkey;

  RemoteMR rmr = rmrs_[RemoteTensorId{ dst_rank, name }];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];
  uint64_t wr_id = (uint64_t) new RdmaWriteID(RDMA_WRITE_ID_TENSOR_WRITE,
                                              nullptr);
  int ret = post_write(buffer_size, src_addr, lkey, remote_addr, rkey, wr_id,
                       qp);
  if (ret < 0) {
    std::cout << "post_write failed." << std::endl;
  }
  //struct ibv_wc wc;
  //ptre_poll_cq(cq_, 1, &wc);
}

void RdmaManager::RdmaWriteIncomingFlag(int dst_rank, bool* flag) {
  size_t buffer_size = sizeof(bool);
  uint64_t src_addr = (uint64_t) flag;
  struct ibv_mr *mr = send_in_flag_mr_;
  uint32_t lkey = mr->lkey;

  RemoteMR rmr = rpmrs_[dst_rank];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];
  uint64_t wr_id = (uint64_t) new RdmaWriteID(RDMA_WRITE_ID_INCOMING_FLAG_WRITE,
                                              nullptr);
  //std::cout << "   buffer_size=" << buffer_size << ", flag=" << flag << "(" << *flag << ")" << ", src_addr=" << src_addr << ", remote_addr=" << remote_addr << std::endl;
  int ret = post_write(buffer_size, src_addr, lkey, remote_addr, rkey, wr_id,
                       qp);
  if (ret < 0) {
    std::cout << "post_write failed." << std::endl;
  }
  //struct ibv_wc wc;
  //ptre_poll_cq(cq_, 1, &wc);
}

bool RdmaManager::AttemptPush(int dst_rank) {
  //auto client = grpc_client_cache_->GetClient(dst_rank);
  //bool ret = client->AttemptPush();
  //return ret;
}

int RdmaManager::PushTensor(int dst_rank, string name, const Tensor& tensor) {
  auto data = tensor.tensor_data();
  size_t buffer_size = (size_t) tensor.TotalBytes();
  uint64_t src_addr = (uint64_t) data.begin();
  struct ibv_mr *mr = send_mrs_[name];
  uint32_t lkey = mr->lkey;

  RemoteMR rmr = rmrs_[RemoteTensorId{ dst_rank, name }];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];
  uint64_t wr_id = (uint64_t) new RdmaWriteID(RDMA_WRITE_ID_TENSOR_WRITE,
                                              nullptr);
  int ret = -1;
  if (atomic_add_) {
    ret = post_fetch_and_add(buffer_size, src_addr, lkey, remote_addr, rkey,
                             wr_id, qp);
  } else {
    ret = post_write(buffer_size, src_addr, lkey, remote_addr, rkey, wr_id, qp);
  }
  if (ret < 0) {
    std::cout << "post_write failed." << std::endl;
  }
}
int RdmaManager::AckPushDone(int dst_rank) {
}
}  // namespace ptre
