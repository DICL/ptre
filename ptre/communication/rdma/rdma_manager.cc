#include "ptre/communication/rdma/rdma_manager.h"

#include <iostream>
#include <cstdlib>
#include <cerrno>

#include "ptre/communication/rdma/rdma.h"

#define DDLOG DLOG(INFO) << "RANK:" << ptre_rank_ << " "

namespace ptre {

RdmaManager::RdmaManager(int ptre_size, int ptre_rank, bool add)
    : ptre_size_(ptre_size), ptre_rank_(ptre_rank), atomic_add_(add) {
  int ret = 0;
  ret = init_rdma_env(rdma_env_);
  if (ret < 0) {
    std::cout << "init_rdma_env failed. ret=" << ret << std::endl;
  } else {
    //std::cout << "init_rdma_env done." << std::endl;
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
      attr.qp_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

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
    //std::cout << "ConnectQP done." << std::endl;
  }
  return 0;
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
  /// TODO: Implement PaddingAllocator instead of this workaround.
  if (length % sizeof(uint64_t) != 0) {
    length += sizeof(float);
  }
  addr = (void*) strpc.data();
  //*((float*) addr) = 0;
  //*((float*) const_cast<char*>(addr)) = 0;
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  if (mr == NULL) {
    std::cerr << "ibv_reg_mr failed. name=" << name << ", addr=" << addr << ", length=" << length << std::endl;
    exit(EXIT_FAILURE);
  }
  recv_mrs_.emplace(name, mr);
  //std::cout << "RecvMR is set for name=" << name << ", addr=" << addr <<
  //          ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;

  /// Set tensor MR for send buf to perform rdma write operations on remote
  /// nodes.
  strpc = send->tensor_data();
  length = strpc.size();
  /// TODO: Implement PaddingAllocator instead of this workaround.
  if (length % sizeof(uint64_t) != 0) {
    length += sizeof(float);
  }
  addr = (void*) strpc.data();
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  send_mrs_.emplace(name, mr);
  //std::cout << "SendMR is set for name=" << name << ", addr=" << addr <<
  //          ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;
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
  //std::cout << "RecvParamMR is set for addr=" << addr <<
  //          ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;

  /// Set tensor MR for send buf to perform rdma write operations on remote
  /// nodes.
  addr = (void*) send_in_flag;
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  send_in_flag_mr_ = mr;
  //std::cout << "SendParamMR is set for addr=" << addr <<
  //          ", lkey=" << mr->lkey << ", rkey=" << mr->rkey << std::endl;
}

/// MR management V2
void RdmaManager::RegisterMR(const BufType buf_type, const string& name,
                             void* buf, size_t length, bool remote) {
  int access = IBV_ACCESS_LOCAL_WRITE;
  if (remote) {
    access |= IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        | IBV_ACCESS_REMOTE_ATOMIC;
  }
  struct ibv_mr* mr = ibv_reg_mr(rdma_env_.pd, buf, length, access);
  if (mr == NULL) {
    LOG(INFO) << "ibv_reg_mr failed : (type, name, buf, length, remote)=("
        << buf_type << ", " << name << ", " << buf << ", " << length << ", " << remote
        << "), errno=" << errno;
    exit(EXIT_FAILURE);
  }
  mrs_[buf_type][name] = mr;
  access_flags_[buf_type][name] = access;
  //if (remote) {
  //  mrs_for_remote_.push_back(mr);
  //}
  if (buf_type == BUF_TYPE_RECV_BUF) {
    recv_tensor_names_.push_back(name);
  }
}

/// Return value is the number of buffers remotely writable.
int RdmaManager::GetRemoteAccessBufInfos(std::vector<BufType>* out_buf_types,
                                         std::vector<string>* out_names) {
  int cnt = 0;
  for (auto it : access_flags_) {
    BufType type = it.first;
    auto& name_to_access = it.second;
    for (auto inner_it : name_to_access) {
      string name = inner_it.first;
      int access = inner_it.second;
      if (access & (IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
            | IBV_ACCESS_REMOTE_ATOMIC)) {
        out_buf_types->push_back(type);
        out_names->push_back(name);
        cnt++;
      }
    }
  }
  return cnt;
}

bool RdmaManager::IsRemoteMRSetV2(const int dst_rank, const BufType buf_type,
                                  const string& name) {
  auto rank_search = rmrs_.find(dst_rank);
  if (rank_search != rmrs_.end()) {
    auto& buf_types = rank_search->second;
    auto type_search = buf_types.find(buf_type);
    if (type_search != buf_types.end()) {
      auto& names = type_search->second;
      auto name_search = names.find(name);
      if (name_search != names.end()) {
        return true;
      }
    }
  }
  return false;
}

void RdmaManager::SetRemoteMRV2(const int dst_rank, const BufType buf_type,
    const string& name, const uint64_t remote_addr, const uint32_t rkey) {
  rmrs_[dst_rank][buf_type].emplace(name, RemoteMR { remote_addr, rkey });
}

/// Must be called after all remote mrs initialized.
void RdmaManager::InitAggWriter() {
  // Init vector send_buf_mrs
  std::map<string, struct ibv_mr*>& send_buf_mr_map = mrs_[BUF_TYPE_SEND_BUF];
  std::vector<struct ibv_mr*> send_buf_mrs;
  for (int i = 0; i < recv_tensor_names_.size(); i++) {
    send_buf_mrs.push_back(send_buf_mr_map[recv_tensor_names_[i]]);
  }
  for (int dst = 0; dst < ptre_size_; dst++) {
    if (dst != ptre_rank_) {
      // Init vector agg_buf_state_rmrs
      std::map<string, RemoteMR>& agg_buf_state_rmr_map =
          rmrs_[dst][BUF_TYPE_AGG_BUF_STATE];
      std::vector<RemoteMR> agg_buf_state_rmrs;
      for (int i = 0; i < recv_tensor_names_.size(); i++) {
        agg_buf_state_rmrs.push_back(
            agg_buf_state_rmr_map[recv_tensor_names_[i]]);
      }
      // Init vector agg_buf_rmrs
      std::map<string, RemoteMR>& agg_buf_rmr_map = rmrs_[dst][BUF_TYPE_AGG_BUF];
      std::vector<RemoteMR> agg_buf_rmrs;
      for (int i = 0; i < recv_tensor_names_.size(); i++) {
        agg_buf_rmrs.push_back(agg_buf_rmr_map[recv_tensor_names_[i]]);
      }
      RdmaAggWriter* writer = new RdmaAggWriter(dst, rdma_env_.pd,
          cq_, qps_[dst],
          recv_tensor_names_,
          agg_buf_state_rmrs,
          agg_buf_rmrs,
          send_buf_mrs);
      agg_writers_.emplace(dst, writer);
    }
  }
}

int RdmaManager::PushTensorBufferedAggregation(const int dst_rank,
                                               const string& name) {

  return agg_writers_[dst_rank]->WriteToAggBuf(name);
}

struct ibv_mr* RdmaManager::GetMR(const BufType buf_type, const string& name) {
  return mrs_[buf_type][name];
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
        RdmaWrId* wr_id = reinterpret_cast<RdmaWrId*>(wc_[i].wr_id);
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
  return (tensor_rmrs_.find(id) != tensor_rmrs_.end());
}

bool RdmaManager::IsRemoteParamMRSet(int rank) {
  return (rpmrs_.find(rank) != rpmrs_.end());
}

void RdmaManager::SetRemoteParamMR(int rank, uint64_t remote_addr,
                                   uint32_t rkey) {
  rpmrs_.emplace(rank, RemoteMR { remote_addr, rkey });
  //std::cout << "RemoteMR is set for rank=" << rank << ", name=is_new_incoming" <<
  //          ", remote_addr=" << (void*) remote_addr << ", rkey=" << rkey << std::endl;
}

void RdmaManager::SetRemoteMR(int rank, const std::string& name,
                              uint64_t remote_addr, uint32_t rkey) {
  tensor_rmrs_.emplace(RemoteTensorId{ rank, name }, RemoteMR { remote_addr, rkey });
  //std::cout << "RemoteMR is set for rank=" << rank << ", name=" << name <<
  //          ", remote_addr=" << (void*) remote_addr << ", rkey=" << rkey << std::endl;
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

int RdmaManager::RdmaWriteTensor(int dst_rank, const std::string& name,
                                 const Tensor& tensor, bool atomic_add) {
  int num_wrs = 0;
  auto data = tensor.tensor_data();
  size_t buffer_size = (size_t) tensor.TotalBytes();
  uint64_t src_addr = (uint64_t) data.begin();
  //struct ibv_mr *mr = send_mrs_[name];
  struct ibv_mr *mr = mrs_[BUF_TYPE_SEND_BUF][name];
  uint32_t lkey = mr->lkey;

  //RemoteMR rmr = tensor_rmrs_[RemoteTensorId{ dst_rank, name }];
  RemoteMR rmr = rmrs_[dst_rank][BUF_TYPE_RECV_BUF][name];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];
  uint64_t wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_TENSOR_WRITE,
                                              nullptr);
  int ret = -1;
  if (atomic_add) {
    ret = post_fetch_and_add(buffer_size, src_addr, lkey, remote_addr, rkey,
                             wr_id, qp);
  } else {
    ret = post_write(buffer_size, src_addr, lkey, remote_addr, rkey, wr_id, qp);
    num_wrs++;
  }
  if (ret < 0) {
    std::cout << "post_write failed." << std::endl;
  }
  return num_wrs;
  //struct ibv_wc wc;
  //ptre_poll_cq(cq_, 1, &wc);
}

int RdmaManager::PushTensorAtomicAdd(int dst_rank, const std::string& name,
                                     const Tensor& tensor) {
  int num_wrs = 0;
  auto data = tensor.tensor_data();
  auto flat = tensor.flat<float>();
  //size_t buffer_size = (size_t) tensor.TotalBytes();
  int num_elem = tensor.NumElements();

  RemoteMR rmr = tensor_rmrs_[RemoteTensorId{ dst_rank, name }];
  uint64_t remote_addr_base = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];

  //uint64_t read_buf;
  uint64_t read_buf_base;
  float* read_buf = reinterpret_cast<float*>(&read_buf_base);
  size_t length;
  void* addr;
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  ibv_mr* mr;
  length = sizeof(float) * 2;
  addr = (void*) read_buf;
  mr = ibv_reg_mr(rdma_env_.pd, addr, length, ibv_access_flags);
  uint64_t wr_id;
  for (int i = 0; i < (num_elem + 1) / 2; i++) {
    uint64_t offset32 = 2 * i;
    uint64_t remote_addr = remote_addr_base + sizeof(float) * offset32;
    struct ibv_wc wc;
    int ret = -1;
    //wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_READ_TWO, nullptr);
    //int ret = post_read(sizeof(uint64_t), (uint64_t) &read_buf_base, mr->lkey,
    //    remote_addr, rkey, wr_id, qp);
    //if (ret < 0) {
    //  std::cout << "post_read failed." << std::endl;
    //}
    //ptre_poll_cq(cq_, 1, &wc);

    while (true) {
      //float sums[2];
      //sums[0] = read_buf[0] + flat(offset32);
      //sums[1] = read_buf[1]
      //    + ((offset32 + 1 < num_elem) ? flat(offset32 + 1) : 0);
      uint64_t swap;
      float* sums = reinterpret_cast<float*>(&swap);
      sums[0] = read_buf[0] + flat(offset32);
      if (offset32 + 1 < num_elem) {
        sums[1] = read_buf[1] + flat(offset32 + 1);
      } else {
        sums[1] = read_buf[1];
      }
      std::cout << read_buf[0] << " " << read_buf[1] << " " << sums[0] << " " << sums[1] << std::endl;
      uint64_t compare_add = read_buf_base;
      //memcpy(reinterpret_cast<float*>(&swap), sums, sizeof(float) * 2);
      struct ibv_send_wr wr;
      memset(&wr, 0, sizeof(wr));
      wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TWO, nullptr);
      ret = post_atomic_cmp_and_swp(sizeof(uint64_t),
          (uint64_t) &read_buf_base, mr->lkey,
          remote_addr,
          rkey, wr, wr_id, qp, compare_add, swap);
      if (ret < 0) {
        std::cout << "post_atomic_cmp_and_swp failed." << std::endl;
      }
      ptre_poll_cq(cq_, 1, &wc);
      if (compare_add == read_buf_base) {
        break;
      } else {
        std::cout << "compare_add=" << compare_add << ", read_buf_base=" << read_buf_base << std::endl;
      }
    }
  }
  int ret = ibv_dereg_mr(mr);
  if (ret < 0) {
    std::cout << "ibv_dereg_mr failed." << std::endl;
  }
  return num_wrs;
  //struct ibv_wc wc;
  //ptre_poll_cq(cq_, 1, &wc);
}

int RdmaManager::PushTensorAtomicAddBatch(int dst_rank, const std::string& name,
                                          const Tensor& tensor) {
  int num_wrs = 0;
  auto data = tensor.tensor_data();
  auto flat = tensor.flat<float>();
  //size_t buffer_size = (size_t) tensor.TotalBytes();
  int num_elem = tensor.NumElements();
  //if (num_elem > 589824) {
  //  return 0;
  //}
  //DDLOG << "PushTensorAtomicAddBatch(" << dst_rank << ", " << name << ", ..), num_elem=" << num_elem;

  RemoteMR rmr = tensor_rmrs_[RemoteTensorId{ dst_rank, name }];
  uint64_t remote_addr_base = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];


  int batch_size_tot = (num_elem + 0) / 2;
  //size_t length = batch_size_tot * 2 * sizeof(float);
  //DDLOG << "TotalBytes=" << tensor.TotalBytes();
  size_t length = tensor.TotalBytes();
  //DDLOG << "size_t length=" << length;
  //float read_buf[batch_size_tot * 2];
  float* read_buf = (float*) malloc(length);
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  //void* vaddr = (void*) read_buf;
  //DDLOG << "readbuf=" << read_buf << ", (uint64_t)=" << (uint64_t) read_buf << ", (void*) addr=" << (void*) read_buf << ", vaddr=" << vaddr;
  struct ibv_mr* mr = ibv_reg_mr(rdma_env_.pd, (void*) read_buf, length,
                                 ibv_access_flags);
  if (mr == NULL) {
    std::cerr << "ibv_reg_mr failed. length=" << length << ": " << std::strerror(errno);
    //if (errno == EINVAL) {
    //  DDLOG << "EINVAL";
    //} else if (errno == ENOMEM) {
    //  DDLOG << "ENOMEM";
    //}
    exit(EXIT_FAILURE);
  } else {
    //DDLOG << "ibv_reg_mr done. length=" << length;
  }
  uint64_t read_buf_addr = (uint64_t) read_buf;
  //DDLOG << "read_buf_addr=" << read_buf_addr;
  uint64_t wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TWO, nullptr);
  post_read(length, read_buf_addr, mr->lkey, remote_addr_base, rkey, wr_id, qp);
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc);
  int cnt = 0;
  bool checker[batch_size_tot] = {};
  while (true) {
    int batch_size = batch_size_tot - cnt;
    if (batch_size > 1024) {
      batch_size = 1024;
    }
    //DDLOG << "batch_size=" << batch_size;
    struct ibv_sge sg[batch_size];
    struct ibv_send_wr wr[batch_size];
    //DDLOG << "sg and wr array set for " << batch_size;
    int j = 0;
    wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TWO, nullptr);
    //DDLOG << "wr_id = " << wr_id;
    for (uint64_t i = 0; i < batch_size_tot; i++) {
      if (checker[i]) {
        continue;
      }
      memset(&sg[j], 0, sizeof(struct ibv_sge));
      //if (j == 0) DDLOG << "memset sg done for " << j;
      sg[j].addr = (uint64_t) (read_buf_addr + i * sizeof(uint64_t));
      //if (j == 0) DDLOG << "sg.addr set " << sg[j].addr;
      sg[j].length = sizeof(uint64_t);
      //if (j == 0) DDLOG << "sg.length set " << sg[j].length;
      sg[j].lkey = mr->lkey;
      //if (j == 0) DDLOG << "sg.lkey set " << sg[j].lkey;
      //if (j == 0) DDLOG << "sg set for " << j;

      memset(&wr[j], 0, sizeof(struct ibv_send_wr));
      wr[j].wr_id = wr_id;
      wr[j].sg_list = &sg[j];
      wr[j].num_sge = 1;
      wr[j].next = (j == batch_size - 1) ? NULL : &wr[j + 1];
      wr[j].opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
      wr[j].send_flags = (j == batch_size - 1) ? IBV_SEND_SIGNALED : 0;
      //if (j == 0) DDLOG << "wr set for " << j;
      //wr[j].send_flags = IBV_SEND_SIGNALED;
      uint64_t offset32 = 2 * i;
      //if (j == 0) DDLOG << "offset32=" << offset32 << " for " << j;
      uint64_t remote_addr = remote_addr_base + sizeof(float) * offset32;
      //if (j == 0) DDLOG << "remote_addr=" << remote_addr << " for " << j;
      wr[j].wr.atomic.remote_addr = remote_addr;
      wr[j].wr.atomic.compare_add = *(((uint64_t*) read_buf) + i);
      wr[j].wr.atomic.rkey = rkey;
      j++;
      if (j == batch_size) {
        break;
      }
    }
    //DDLOG << "wr set done for " << batch_size;
    j = 0;
    for (int i = 0; i < batch_size_tot; i++) {
      if (checker[i]) {
        continue;
      }
      uint64_t offset32 = 2 * i;
      uint64_t swap;
      float* sums = reinterpret_cast<float*>(&swap);
      if (0) {
        sums[0] = read_buf[offset32] + flat(offset32);
        if (offset32 + 1 < num_elem) {
          sums[1] = read_buf[offset32 + 1] + flat(offset32 + 1);
        } else {
          sums[1] = read_buf[offset32 + 1];
        }
      } else {
        sums[0] = read_buf[offset32];
        sums[1] = read_buf[offset32 + 1];
      }
      wr[j].wr.atomic.swap = swap;
      j++;
      if (j == batch_size) {
        break;
      }
    }
    //DDLOG << "set wr.atomic.swap done.";
    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp, &wr[0], &bad_wr);
    //DDLOG << "ibv_post_send done. batch_size=" << batch_size;
    ptre_poll_cq(cq_, 1, &wc);
    //DDLOG << "poll_cq done.";
    j = 0;
    for (int i = 0; i < batch_size_tot; i++) {
      if (checker[i]) {
        continue;
      }
      uint64_t compare_add = wr[j].wr.atomic.compare_add;
      if (compare_add == *(((uint64_t*) read_buf) + i)) {
        checker[i] = 1;
        //std::cout << "batch=" << i << ", ca=" << compare_add << ", rb=" << *(((uint64_t*) read_buf) + i * 2) << std::endl;
        cnt++;
      }
      j++;
      if (j == batch_size) {
        break;
      }
    }
    if (cnt > 0 && cnt < batch_size_tot) {
      //LOG_EVERY_N(INFO, 10000) << "RANK:" << ptre_rank_ << " match=[" << cnt << " / " << batch_size_tot << "]";
    }
    if (cnt == batch_size_tot) {
      break;
    } else {
      //if (cnt > 0) {
      //  break;
      //}
    }
  }
  int ret = ibv_dereg_mr(mr);
  if (ret < 0) {
    std::cout << "ibv_dereg_mr failed." << std::endl;
  }
  free(read_buf);
  return num_wrs;
}

void RdmaManager::RdmaWriteIncomingFlag(int dst_rank, bool* flag) {
  size_t buffer_size = sizeof(bool);
  uint64_t src_addr = (uint64_t) flag;
  struct ibv_mr *mr = send_in_flag_mr_;
  uint32_t lkey = mr->lkey;

  //RemoteMR rmr = rpmrs_[dst_rank];
  RemoteMR rmr = rmrs_[dst_rank][BUF_TYPE_FLAG_RECV]["is_new_incoming"];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];
  uint64_t wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_INCOMING_FLAG_WRITE,
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
  //GrpcClient* client = grpc_client_cache_->GetClient(dst_rank, &client);
  //bool ret = client->AttemptPush();
  //return ret;
}

int RdmaManager::PushTensor(int dst_rank, string name, const Tensor& tensor) {
  auto data = tensor.tensor_data();
  size_t buffer_size = (size_t) tensor.TotalBytes();
  uint64_t src_addr = (uint64_t) data.begin();
  struct ibv_mr *mr = send_mrs_[name];
  uint32_t lkey = mr->lkey;

  RemoteMR rmr = tensor_rmrs_[RemoteTensorId{ dst_rank, name }];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];
  uint64_t wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_TENSOR_WRITE,
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
