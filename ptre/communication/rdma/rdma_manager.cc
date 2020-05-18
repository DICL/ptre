#include "ptre/communication/rdma/rdma_manager.h"

#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <chrono>

#include "ptre/communication/rdma/rdma.h"
//#include "ptre/cm/tensor_aggregator.h"

namespace ptre {

RdmaManager::RdmaManager(int ptre_size, int ptre_rank) {
  ptre_size_ = ptre_size;
  ptre_rank_ = ptre_rank;

  // Init RDMA Environment
  int ret = 0;
  device_list_ = ibv_get_device_list(NULL);
  ctx_ = ibv_open_device(device_list_[0]);
  pd_ = ibv_alloc_pd(ctx_);
  ret = ibv_query_port(ctx_, IB_PORT, &port_attr_);

  // Create Completion Queues and Queue Pairs
  for (int i = 0; i < ptre_size_; i++) {
    // Send CQ
    struct ibv_cq* send_cq = ibv_create_cq(ctx_, 100, NULL, NULL, 0);
    send_cqs_.push_back(send_cq);
    // Recv CQ
    struct ibv_cq* recv_cq = ibv_create_cq(ctx_, 100, NULL, NULL, 0);
    recv_cqs_.push_back(recv_cq);
    // QP
    struct ibv_qp* qp;
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(ibv_qp_init_attr));
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = QP_MAX_WR_DEFAULT;
    qp_init_attr.cap.max_recv_wr = QP_MAX_WR_DEFAULT;
    qp_init_attr.cap.max_send_sge = 16;
    qp_init_attr.cap.max_recv_sge = 16;
    qp = ibv_create_qp(pd_, &qp_init_attr);
    if (!qp) {
      LOG(ERROR) << "Failed to create QP for rank=" << i;
      exit(1);
    }
    qps_.push_back(qp);
    INITQP(i);
  }

  // Init Remote Attributes Arrays
  remote_lids_.resize(ptre_size_);

  // Init Receive Work Request Array
  for (int i = 0; i < ptre_size_; i++) {
    void* buf = malloc(1);
    struct ibv_mr* mr = ibv_reg_mr(pd_, buf, 1, IBV_ACCESS_LOCAL_WRITE);
    struct ibv_sge* sge = (struct ibv_sge*) calloc(1, sizeof(struct ibv_sge));
    sge->addr = (uint64_t) mr->addr;
    sge->length = mr->length;
    sge->lkey = mr->lkey;
    struct ibv_recv_wr* wr =
        (struct ibv_recv_wr*) calloc(1, sizeof(struct ibv_recv_wr));
    wr->wr_id = 0;
    wr->sg_list = sge;
    wr->num_sge = 1;
    recv_wrs_.push_back(wr);
  }
}

RdmaManager::~RdmaManager() { }

void RdmaManager::INITQP(int dst) {
  // INIT QP
  int ret;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = IB_PORT;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
      | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  ret = ibv_modify_qp(qps_[dst], &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX
      | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to INIT state: " << std::strerror(ret)
        << "(code=" << ret << ")";
  }
}

void RdmaManager::RTRQP(int dst, uint16_t remote_lid, uint32_t remote_qpn,
    uint32_t remote_psn = 0) {
  // INIT -> RTR
  int ret;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = remote_psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = remote_lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num  = IB_PORT;
  ret = ibv_modify_qp(qps_[dst], &attr,
        IBV_QP_STATE
      | IBV_QP_AV
      | IBV_QP_PATH_MTU
      | IBV_QP_DEST_QPN
      | IBV_QP_RQ_PSN
      | IBV_QP_MAX_DEST_RD_ATOMIC
      | IBV_QP_MIN_RNR_TIMER);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RTR errno=" << ret << ": "
        << std::strerror(ret);
  }
}

void RdmaManager::RTSQP(int dst, uint32_t my_psn = 0) {
  // RTR -> RTS
  int ret;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = my_psn;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  ret = ibv_modify_qp(qps_[dst], &attr,
        IBV_QP_STATE
      | IBV_QP_TIMEOUT
      | IBV_QP_RETRY_CNT
      | IBV_QP_RNR_RETRY
      | IBV_QP_SQ_PSN
      | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RTS errno=" << ret << ": "
        << std::strerror(ret);
  }
}

void RdmaManager::RESETQP(int dst) {
  // Reset QP
  int ret;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  ret = ibv_modify_qp(qps_[dst], &attr, IBV_QP_STATE);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RESET state: " << std::strerror(ret)
        << "(code=" << ret << ")";
  }
}

void RdmaManager::ConnectQP(int dst, uint32_t remote_qpn) {
  RTRQP(dst, remote_lids_[dst], remote_qpn);
  RTSQP(dst);
}

void RdmaManager::SetTrainableVariables(std::vector<RemoteVariable*>& vars,
    const std::vector<string>& names) {
  for (int i = 0; i < vars.size(); i++) {
    var_name_to_index_[names[i]] = i;
    // Send/Recv Buffer for Variables
    size_t length = vars[i]->rcv_length();
    RegisterMR(BUF_TYPE_RECV_BUF, names[i], vars[i]->rcv_data(), length,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    PushVariable* pvar = new PushVariable(length);
    push_variables_.push_back(pvar);
    void* send_buf = push_variables_.back()->data();
    RegisterMR(BUF_TYPE_SEND_BUF, names[i], send_buf, length,
        IBV_ACCESS_LOCAL_WRITE);
    // Permit
    RegisterMR(BUF_TYPE_PUSH_PERMIT, names[i], vars[i]->permit_data(),
        sizeof(int), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    void* permit_read_buf = malloc(sizeof(int));
    RegisterMR(BUF_TYPE_PUSH_PERMIT_READ, names[i], permit_read_buf,
        sizeof(int), IBV_ACCESS_LOCAL_WRITE);
  }
}

void RdmaManager::set_remote_lid(int dst, uint16_t lid) {
  remote_lids_[dst] = lid;
}

uint16_t RdmaManager::remote_lid(int dst) {
  return remote_lids_[dst];
}

#if 0
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
    exit(1);
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
#endif

/// MR management V2
void RdmaManager::RegisterMR(const BufType buf_type, const string& name,
    void* buf, size_t length, int access = IBV_ACCESS_LOCAL_WRITE) {
  struct ibv_mr* mr = ibv_reg_mr(pd_, buf, length, access);
  if (mr == NULL) {
    LOG(ERROR) << "ibv_reg_mr failed : (type, name, buf, length, remote)=("
        << buf_type << ", " << name << ", " << buf << ", " << length
        << "), errno=" << errno;
    exit(1);
  }
  mrs_[buf_type][name] = mr;
  access_flags_[buf_type][name] = access;
  /*
  if (buf_type == BUF_TYPE_RECV_BUF) {
    recv_tensor_names_.push_back(name);
  }
  */
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

#if 0
int RdmaManager::RdmaWriteBufRemote(const int dst_rank, const BufType src_type,
    const BufType dst_type, const string& name, const bool polling) {
  struct ibv_qp* qp = qps_[dst_rank];
  struct ibv_mr* mr = mrs_[src_type][name];
  RemoteMR rmr = rmrs_[dst_rank][dst_type][name];
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) mr->addr;
  sge.length = mr->length;
  sge.lkey = mr->lkey;
  struct ibv_send_wr wr;
  int ret;
  // Try Write
  while (true) {
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_TENSOR_WRITE, nullptr);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = rmr.remote_addr;
    wr.wr.rdma.rkey = rmr.rkey;
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "Failed to RDMA WRITE: " << std::strerror(ret) << "(code=" << ret << ")";
      exit(1);
    }
#if 0
    if (polling) {
      struct ibv_wc wc;
      ptre_poll_cq(cq_, 1, &wc);
    }
#else
    struct ibv_wc wc;
    ptre_poll_cq(send_cqs_[dst_rank], 1, &wc);
    if (!wc.status) {
      break;
    } else {
      LOG(ERROR) << "Bad WC status=" << wc.status << ", Starting to reset QP";
    }
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    ret = ibv_query_qp(qp, &attr,
          IBV_QP_STATE
        | IBV_QP_AV
        | IBV_QP_DEST_QPN,
        &init_attr);
    if (ret) {
      LOG(ERROR) << "Failed to query QP state: " << std::strerror(ret);
      exit(1);
    }
    if (attr.qp_state != IBV_QPS_RTS) {
      uint32_t dest_qp_num = attr.dest_qp_num;
      uint16_t dlid = attr.ah_attr.dlid;
      LOG(ERROR) << "QP num=" << qp->qp_num << ", state=" << attr.qp_state << ", dest_qp_num=" << dest_qp_num;
      rdma_qp_reset_to_rts(qp, dest_qp_num, dlid);
    }
#endif
  }
}
#endif

/// Must be called after all remote mrs initialized.
#if 0
void RdmaManager::InitAggWriter() {
  // Init vector send_buf_mrs
  std::map<string, struct ibv_mr*>& send_buf_mr_map = mrs_[BUF_TYPE_SEND_BUF];
  std::vector<struct ibv_mr*> send_buf_mrs;
  for (int i = 0; i < recv_tensor_names_.size(); i++) {
    send_buf_mrs.push_back(send_buf_mr_map[recv_tensor_names_[i]]);
  }
  for (int dst = 0; dst < ptre_size_; dst++) {
#if 0
    bool condition = dst != ptre_rank_;
#else
    bool condition = true;
#endif
    if (condition) {
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
          send_cqs_[dst], qps_[dst],
          recv_tensor_names_,
          agg_buf_state_rmrs,
          agg_buf_rmrs,
          send_buf_mrs);
      agg_writers_.emplace(dst, writer);
    }
  }
}
#endif

#if 0
int RdmaManager::PushTensorBufferedAggregation(const int dst_rank,
                                               const string& name) {

  return agg_writers_[dst_rank]->WriteToAggBuf(name);
}

int RdmaManager::PushTensorBufferedAggregation(const int dst_rank,
    const std::vector<string>& names) {
  int done_cnt = 0;
  int n = names.size();
  bool done_flags[n] = { };
  uint64_t read_states[n] = { };
  bool all_done = false;
  LOG(INFO) << "[DEBUG] Write Tensors to rank " << dst_rank;
  auto start_time = std::chrono::system_clock::now();
  auto last_time = start_time;
  while (!all_done) {
    auto curr_time = std::chrono::system_clock::now();
    std::chrono::duration<double> time_diff = curr_time - last_time;
    if (time_diff.count() > 10) {
      LOG(INFO) << "[DEBUG] Hangs on writing tensors to rank " << dst_rank;
      int written_cnt = 0;
      for (int i = 0; i < n; i++) {
        if (!done_flags[i]) {
          LOG(INFO) << "[DEBUG] not written: " << names[i] << ", state = " << read_states[i];
        } else {
          written_cnt++;
        }
      }
      LOG(INFO) << "[DEBUG] Written count = " << written_cnt << " to rank " << dst_rank;
      last_time = curr_time;
    }
    all_done = true;
    for (int i = 0; i < n; i++) {
      if (!done_flags[i]) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //LOG(INFO) << "[DEBUG] TransitStateV2(" << names[i] << ")";
        uint64_t remote_state = agg_writers_[dst_rank]->TransitStateV2(names[i],
            StatefulAggBuf::kRecvReady, StatefulAggBuf::kRecvInProgress);
        read_states[i] = remote_state;
        if (remote_state == StatefulAggBuf::kRecvReady) {
          //LOG(INFO) << "[DEBUG] WriteToAggBufV2(" << names[i] << ")";
          agg_writers_[dst_rank]->WriteToAggBufV2(names[i]);
          while (true) {
            uint64_t inner = agg_writers_[dst_rank]->TransitStateV2(names[i],
                2, 3);
            if (inner == 2) {
              break;
            } else {
              LOG(ERROR) << "inner=" << inner;
            }
          }
          done_flags[i] = true;
          last_time = curr_time;
          //LOG(INFO) << "[DEBUG] Write Done for dst=" << dst_rank << " (" << ++done_cnt << "/" << n << ")";
        } else {
          //LOG(INFO) << "[DEBUG] TransitState Failed for dst=" << dst_rank << ", name=" << names[i];
          all_done = false;
        }
      }
    }
    //usleep(1000 * 1000 * 1);
  }
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> time_diff = end_time - start_time;
  LOG(INFO) << "Done pushing all tensors to rank " << dst_rank << ": elapsed " << time_diff.count() << " s";
}

uint64_t RdmaManager::RdmaFetchAndAdd(const int dst_rank,
    const BufType dst_type, const string& name, const uint64_t add,
    struct ibv_mr* read_mr) {
  LOG(ERROR) << "THIS FUNCTION IS NOT READY YET.";
  exit(1);
  uint64_t* read_buf;
  bool dereg_mr;
  if (read_mr == nullptr) {
    dereg_mr = true;
    read_buf = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
    read_mr = ibv_reg_mr(rdma_env_.pd, (void*) read_buf, sizeof(uint64_t),
        IBV_ACCESS_LOCAL_WRITE);
  } else {
    dereg_mr = false;
    read_buf = (uint64_t*) read_mr->addr;
  }

  struct ibv_sge sge;
  sge.addr = (uint64_t) read_buf;
  sge.length = sizeof(uint64_t);
  sge.lkey = read_mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS, nullptr);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr.send_flags = IBV_SEND_SIGNALED;
  RemoteMR rmr = rmrs_[dst_rank][dst_type][name];
  wr.wr.atomic.remote_addr = rmr.remote_addr;
  wr.wr.atomic.compare_add = add;
  wr.wr.atomic.rkey = rmr.rkey;

  struct ibv_send_wr* bad_wr;
  struct ibv_qp* qp = qps_[dst_rank];
  int ret = ibv_post_send(qp, &wr, &bad_wr);
}
#endif

int RdmaManager::RdmaRead(int dst, const BufType buf_type, const string& name,
    struct ibv_mr* read_mr, size_t read_length) {
  int ret;
  // Retrieve remote address
  RemoteMR rmr = rmrs_[dst][buf_type][name];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  // Init SGE
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) read_mr->addr;
  sge.length = read_length;
  sge.lkey = read_mr->lkey;
  // Init send WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;
  // QP
  struct ibv_qp* qp = qps_[dst];
  // Try RDMA read
  while (true) {
    // WR ID
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_READ, nullptr);
    // Post send
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "Failed to ibv_post_send for read " << name << ":"
          << buf_type << " for rank " << dst;
      exit(1);
    }
    struct ibv_wc wc;
    ptre_poll_cq(send_cqs_[dst], 1, &wc);
    if (!wc.status) {
      break;
    } else {
      LOG(ERROR) << "Bad WC status=" << wc.status << ", Starting to reset QP";
    }
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    ret = ibv_query_qp(qp, &attr,
          IBV_QP_STATE
        | IBV_QP_AV
        | IBV_QP_DEST_QPN,
        &init_attr);
    if (ret) {
      LOG(ERROR) << "Failed to query QP state: " << std::strerror(ret);
      exit(1);
    }
    if (attr.qp_state != IBV_QPS_RTS) {
      uint32_t dest_qp_num = attr.dest_qp_num;
      uint16_t dlid = attr.ah_attr.dlid;
      LOG(ERROR) << "QP num=" << qp->qp_num << ", state=" << attr.qp_state << ", dest_qp_num=" << dest_qp_num;
      rdma_qp_reset_to_rts(qp, dest_qp_num, dlid);
    }
  }
  return ret;
}

int RdmaManager::RdmaWrite(int dst, const BufType buf_type,
    const string& var_name, struct ibv_mr* send_mr, size_t send_length,
    uint32_t* imm_data) {
  int ret;
  // Retrieve remote address
  RemoteMR rmr = rmrs_[dst][buf_type][var_name];
  uint64_t remote_addr = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  // Init SGE
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) send_mr->addr;
  sge.length = send_length;
  sge.lkey = send_mr->lkey;
  // Init send WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (imm_data == nullptr) {
    wr.opcode = IBV_WR_RDMA_WRITE;
  } else {
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = *imm_data;
  }
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;
  // QP
  struct ibv_qp* qp = qps_[dst];
  // Try RDMA read
  while (true) {
    // WR ID
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_READ, nullptr);
    // Post send
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "Failed to ibv_post_send for write " << var_name << ":"
          << buf_type << " for rank " << dst;
      exit(1);
    }
    struct ibv_wc wc;
    ptre_poll_cq(send_cqs_[dst], 1, &wc);
    if (!wc.status) {
      break;
    } else {
      LOG(ERROR) << "Bad WC status=" << wc.status << ", Starting to reset QP";
    }
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    ret = ibv_query_qp(qp, &attr,
          IBV_QP_STATE
        | IBV_QP_AV
        | IBV_QP_DEST_QPN,
        &init_attr);
    if (ret) {
      LOG(ERROR) << "Failed to query QP state: " << std::strerror(ret);
      exit(1);
    }
    if (attr.qp_state != IBV_QPS_RTS) {
      uint32_t dest_qp_num = attr.dest_qp_num;
      uint16_t dlid = attr.ah_attr.dlid;
      LOG(ERROR) << "QP num=" << qp->qp_num << ", state=" << attr.qp_state << ", dest_qp_num=" << dest_qp_num;
      rdma_qp_reset_to_rts(qp, dest_qp_num, dlid);
    }
  }
  return ret;
}

struct ibv_mr* RdmaManager::GetMR(const BufType buf_type, const string& name) {
  return mrs_[buf_type][name];
}

void RdmaManager::SetRemoteAddress(int dst_rank, const BufType buf_type,
      const string& name, const uint64_t remote_addr, const uint32_t rkey) {
  rmrs_[dst_rank][buf_type].emplace(name, RemoteMR { remote_addr, rkey });
}

void RdmaManager::GetRemoteAddress(int dst_rank, const BufType buf_type,
      const string& name, uint64_t* out_addr, uint32_t* out_rkey) {
  RemoteMR rmr = rmrs_[dst_rank][buf_type][name];
  *out_addr = rmr.remote_addr;
  *out_rkey = rmr.rkey;
}

void RdmaManager::ProcessCQ() {
#if 0
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
#endif
}

void RdmaManager::Poll(int num_comps) {
  LOG(ERROR) << "Not Implemented.";
  exit(1);
  struct ibv_wc wcs[num_comps];
  //ptre_poll_cq(cq_, num_comps, wcs);
}

//void RdmaManager::InitTensorMRs(int dst_rank, const std::string& name,
//                                const Tensor& recv, const Tensor& send) {

bool RdmaManager::IsRemoteMRSet(int rank, const std::string& name) {
  RemoteTensorId id{ rank, name };
  return (tensor_rmrs_.find(id) != tensor_rmrs_.end());
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
  if (ret) {
    std::cout << "post_write failed." << std::endl;
    exit(1);
  }
  return num_wrs;
  //struct ibv_wc wc;
  //ptre_poll_cq(cq_, 1, &wc);
}

#if 0
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
    //if (ret) {
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
      if (ret) {
        std::cout << "post_atomic_cmp_and_swp failed." << std::endl;
      }
      ptre_poll_cq(send_cqs_[dst_rank], 1, &wc);
      if (compare_add == read_buf_base) {
        break;
      } else {
        std::cout << "compare_add=" << compare_add << ", read_buf_base=" << read_buf_base << std::endl;
      }
    }
  }
  int ret = ibv_dereg_mr(mr);
  if (ret) {
    std::cout << "ibv_dereg_mr failed." << std::endl;
    exit(1);
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

  RemoteMR rmr = tensor_rmrs_[RemoteTensorId{ dst_rank, name }];
  uint64_t remote_addr_base = rmr.remote_addr;
  uint32_t rkey = rmr.rkey;
  struct ibv_qp *qp = qps_[dst_rank];


  int batch_size_tot = (num_elem + 0) / 2;
  //size_t length = batch_size_tot * 2 * sizeof(float);
  size_t length = tensor.TotalBytes();
  //float read_buf[batch_size_tot * 2];
  float* read_buf = (float*) malloc(length);
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  //void* vaddr = (void*) read_buf;
  struct ibv_mr* mr = ibv_reg_mr(rdma_env_.pd, (void*) read_buf, length,
                                 ibv_access_flags);
  if (mr == NULL) {
    std::cerr << "ibv_reg_mr failed. length=" << length << ": " << std::strerror(errno);
    //if (errno == EINVAL) {
    //} else if (errno == ENOMEM) {
    //}
    exit(1);
  } else {
  }
  uint64_t read_buf_addr = (uint64_t) read_buf;
  uint64_t wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TWO, nullptr);
  post_read(length, read_buf_addr, mr->lkey, remote_addr_base, rkey, wr_id, qp);
  struct ibv_wc wc;
  ptre_poll_cq(send_cqs_[dst_rank], 1, &wc);
  int cnt = 0;
  bool checker[batch_size_tot] = {};
  while (true) {
    int batch_size = batch_size_tot - cnt;
    if (batch_size > 1024) {
      batch_size = 1024;
    }
    struct ibv_sge sg[batch_size];
    struct ibv_send_wr wr[batch_size];
    int j = 0;
    wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TWO, nullptr);
    for (uint64_t i = 0; i < batch_size_tot; i++) {
      if (checker[i]) {
        continue;
      }
      memset(&sg[j], 0, sizeof(struct ibv_sge));
      sg[j].addr = (uint64_t) (read_buf_addr + i * sizeof(uint64_t));
      sg[j].length = sizeof(uint64_t);
      sg[j].lkey = mr->lkey;

      memset(&wr[j], 0, sizeof(struct ibv_send_wr));
      wr[j].wr_id = wr_id;
      wr[j].sg_list = &sg[j];
      wr[j].num_sge = 1;
      wr[j].next = (j == batch_size - 1) ? NULL : &wr[j + 1];
      wr[j].opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
      wr[j].send_flags = (j == batch_size - 1) ? IBV_SEND_SIGNALED : 0;
      //wr[j].send_flags = IBV_SEND_SIGNALED;
      uint64_t offset32 = 2 * i;
      uint64_t remote_addr = remote_addr_base + sizeof(float) * offset32;
      wr[j].wr.atomic.remote_addr = remote_addr;
      wr[j].wr.atomic.compare_add = *(((uint64_t*) read_buf) + i);
      wr[j].wr.atomic.rkey = rkey;
      j++;
      if (j == batch_size) {
        break;
      }
    }
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
    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp, &wr[0], &bad_wr);
    ptre_poll_cq(send_cqs_[dst_rank], 1, &wc);
    j = 0;
    for (int i = 0; i < batch_size_tot; i++) {
      if (checker[i]) {
        continue;
      }
      uint64_t compare_add = wr[j].wr.atomic.compare_add;
      if (compare_add == *(((uint64_t*) read_buf) + i)) {
        checker[i] = 1;
        cnt++;
      }
      j++;
      if (j == batch_size) {
        break;
      }
    }
    if (cnt == batch_size_tot) {
      break;
    }
  }
  int ret = ibv_dereg_mr(mr);
  if (ret) {
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
  if (ret) {
    std::cout << "post_write failed." << std::endl;
  }
  //struct ibv_wc wc;
  //ptre_poll_cq(cq_, 1, &wc);
}
#endif

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
  if (ret) {
    std::cout << "post_write failed." << std::endl;
  }
}
int RdmaManager::NotifyPushDone(int dst_rank) {
}

int RdmaManager::PushAndNotify(int dst, const string& var_name) {
  int ret;
  // 1. Read Permit Table
  struct ibv_mr* read_mr = mrs_[BUF_TYPE_PUSH_PERMIT_READ][var_name];
  ret = RdmaRead(dst, BUF_TYPE_PUSH_PERMIT, var_name, read_mr, sizeof(int));
  ret = *((int*) read_mr->addr);
  // 2. RDMA Write with IMM Data
  if (ret != ptre_rank_) {
    return 1;
  } else {
    ibv_mr* send_mr = mrs_[BUF_TYPE_SEND_BUF][var_name];
    uint32_t imm_data = var_name_to_index_[var_name];
    ret = RdmaWrite(dst, BUF_TYPE_RECV_BUF, var_name, send_mr, send_mr->length,
        &imm_data);
    return ret;
  }
}

int RdmaManager::ReceivePushNotify(int dst) {
  int ret;
  struct ibv_qp* qp = qps_[dst];
  struct ibv_cq* cq = recv_cqs_[dst];
  struct ibv_recv_wr* wr = recv_wrs_[dst];
  struct ibv_recv_wr* bad_wr;
  ret = ibv_post_recv(qp, wr, &bad_wr);
  struct ibv_wc wc;
  ret = ibv_poll_cq(cq, 1, &wc);
  if (ret > 0) {
    if (wc.status == IBV_WC_SUCCESS) {
      // Write done
      uint32_t idx = wc.imm_data;
      return idx;
    } else {
      LOG(ERROR) << "Failed to poll recv CQ for rank=" << dst << ": status="
          << wc.status;
      exit(EXIT_FAILURE);
    }
  }
  return -1;
}


void RdmaManager::InitPush(int idx) {
  auto&& var = push_variables_[idx];
  var->StopPush();
}
void RdmaManager::SetPushReady(int idx) {
  auto&& var = push_variables_[idx];
  var->StartPush();
}
void RdmaManager::SetPushReady(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  SetPushReady(idx);
}
bool RdmaManager::IsPushReady(int idx) {
  auto&& var = push_variables_[idx];
  if (var->GetState() == 1) {
    return true;
  } else {
    return false;
  }
}
bool RdmaManager::IsPushReady(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  return IsPushReady(idx);
}

struct ibv_context* RdmaManager::ctx() {
  return ctx_;
}
struct ibv_pd* RdmaManager::pd() {
  return pd_;
}
//struct ibv_qp* RdmaManager::qp(int dest_rank) { return qps_[dest_rank]; }
//struct ibv_cq* RdmaManager::send_cq(int dst) { return send_cqs_[dst]; }
//struct ibv_cq* RdmaManager::recv_cq(int dst) { return recv_cqs_[dst]; }

}  // namespace ptre
