#include "ptre/common/communication/rdma/rdma_agg_writer.h"

#include "tensorflow/core/platform/logging.h"

#include <iostream>
#include <chrono>

namespace ptre {

void cas() {
}

RdmaAggWriter::RdmaAggWriter(int dst_rank, struct ibv_pd* pd,
                struct ibv_cq* cq, struct ibv_qp* qp,
                const std::vector<string>& names,
                const std::vector<RemoteMR>& agg_buf_state_rmrs,
                const std::vector<RemoteMR>& agg_buf_rmrs,
                const std::vector<struct ibv_mr*>& send_buf_mrs)
    : dst_rank_(dst_rank), n_(names.size()), pd_(pd), cq_(cq), qp_(qp) {
  LOG(INFO) << "dst=" << dst_rank << ", cq=" << cq_;
  // Init names
  for (int i = 0; i < names.size(); i++) {
    const string& name = names[i];
    //names_.push_back(name);
    name_to_index_.emplace(name, i);
  }
  /// Init State Read Buf and its MR for CAS
  state_read_bufs_ = (uint64_t*) malloc(sizeof(uint64_t) * n_);
  for (int i = 0; i < n_; i++) {
    state_read_bufs_[i] = StatefulAggBuf::kRecvReady;
  }
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  for (int i = 0; i < n_; i++) {
    struct ibv_mr* state_read_buf_mr = ibv_reg_mr(pd_,
        (void*) (state_read_bufs_ + i), sizeof(uint64_t), ibv_access_flags);
    state_read_buf_mrs_.push_back(state_read_buf_mr);
  }
  /// Init Remote MRs
  for (int i = 0; i < n_; i++) {
    agg_buf_state_rmrs_.push_back(agg_buf_state_rmrs[i]);
    agg_buf_rmrs_.push_back(agg_buf_rmrs[i]);
    send_buf_mrs_.push_back(send_buf_mrs[i]);
  }
}

int RdmaAggWriter::TransitState(const string& name) {
  int idx = name_to_index_[name];
  struct ibv_mr* state_read_buf_mr = state_read_buf_mrs_[idx];
  uint64_t* state_read_buf = (uint64_t*) state_read_buf_mr->addr;
  /// Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  /// Get Remote AggBuf State Mr
  RemoteMR state_rmr = agg_buf_state_rmrs_[idx];
  /// Init WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(struct ibv_send_wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.atomic.remote_addr = (uint64_t) state_rmr.remote_addr;
  wr.wr.atomic.compare_add = StatefulAggBuf::kRecvReady;
  wr.wr.atomic.swap = StatefulAggBuf::kRecvInProgress;
  wr.wr.atomic.rkey = state_rmr.rkey;
  /// Try CAS
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  if (ret) {
    LOG(ERROR) << "Failed to post send CAS: " << std::strerror(ret);
    exit(1);
  }
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc, 2);  // delete RdmaWrId
  usleep(1);
#if 0
  if (*state_read_buf == StatefulAggBuf::kRecvReady) {
    return 0;
  }
  return -1;
#else
  return *state_read_buf;
#endif
}

uint64_t RdmaAggWriter::TransitStateV2(const string& name, const uint64_t from,
    const uint64_t to) {
  int idx = name_to_index_[name];
  uint64_t state_read;
  uint64_t* state_read_buf = &state_read;
  int access = IBV_ACCESS_LOCAL_WRITE;
  struct ibv_mr* state_read_buf_mr = ibv_reg_mr(pd_, (void*) state_read_buf,
      sizeof(uint64_t), access);
  // Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  // Get Remote AggBuf State Mr
  RemoteMR state_rmr = agg_buf_state_rmrs_[idx];
  struct ibv_send_wr wr;
  int ret;
  // Try CAS
  while (true) {
    // Init WR
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.atomic.remote_addr = (uint64_t) state_rmr.remote_addr;
    wr.wr.atomic.compare_add = from;
    wr.wr.atomic.swap = to;
    wr.wr.atomic.rkey = state_rmr.rkey;
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "Failed to post send CAS: " << std::strerror(ret);
      exit(1);
    }
    struct ibv_wc wc;
    //LOG(INFO) << "[DEBUG] Starting poll";
    ptre_poll_cq(cq_, 1, &wc, 3);  // delete RdmaWrId
    //LOG(INFO) << "[DEBUG] Poll Done.";
    if (!wc.status) {
      break;
    }
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    ret = ibv_query_qp(qp_, &attr,
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
      LOG(ERROR) << "QP num=" << qp_->qp_num << ", state=" << attr.qp_state << ", dest_qp_num=" << dest_qp_num;
      rdma_qp_reset_to_rts(qp_, dest_qp_num, dlid);
    }
  }
  usleep(1);
  uint64_t read_state = *state_read_buf;
  ret = ibv_dereg_mr(state_read_buf_mr);
  return read_state;
}


int RdmaAggWriter::WriteToAggBuf(const string& name) {
  // TODO: Check send buf state
  int idx = name_to_index_[name];

  //LOG(INFO) << "[DEBUG] Writing Target=" << dst_rank_ << ", Name=" << name;
  //LOG(INFO) << "[DEBUG] STEP 1: CAS";
  /// STEP 1: CAS
  /// Get State Read Buf MR
  struct ibv_mr* state_read_buf_mr = state_read_buf_mrs_[idx];
  uint64_t* state_read_buf = (uint64_t*) state_read_buf_mr->addr;
  /// Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  /// Get Remote AggBuf State Mr
  RemoteMR state_rmr = agg_buf_state_rmrs_[idx];
  /// Init WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.atomic.remote_addr = (uint64_t) state_rmr.remote_addr;
  wr.wr.atomic.compare_add = StatefulAggBuf::kRecvReady;
  wr.wr.atomic.swap = StatefulAggBuf::kRecvInProgress;
  wr.wr.atomic.rkey = state_rmr.rkey;
  /// Repeat CAS until we get kRecvReady
  bool cas_done = false;
  int agg_ip_cnt = 0;
  while (!cas_done) {
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "[DEBUG] Failed to ibv_post_send: " << ret;
      exit(1);
    }
    struct ibv_wc wc;
    ptre_poll_cq(cq_, 1, &wc, 2);  // delete RdmaWrId
    usleep(1);
    if (*state_read_buf == StatefulAggBuf::kRecvReady) {
      cas_done = true;
    } else if (*state_read_buf == StatefulAggBuf::kAggInProgress) {
#if 0
      /// NOTE: This is workaround of resolving CAS read problem
      /// TODO: Solve the problem in a correct way.
      agg_ip_cnt++;
      if (agg_ip_cnt > 1000) {
        wr.wr.atomic.compare_add = *state_read_buf;
      }
#endif
    } else {
      //LOG(INFO) << "[DEBUG] state_read_buf=" << *state_read_buf << ", kRecvReady=" << StatefulAggBuf::kRecvReady;
    }
  }

  //LOG(INFO) << "[DEBUG] STEP 2: Write";
  /// STEP 2: Write
  /// Get Local Send Buf MR
  struct ibv_mr* send_buf_mr = send_buf_mrs_[idx];
  /// Init Scatter & Gather List
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) send_buf_mr->addr;
  sge.length = send_buf_mr->length;
  sge.lkey = send_buf_mr->lkey;
  /// Get Remote AggBuf MR
  RemoteMR rmr = agg_buf_rmrs_[idx];
  /// Init WR
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_TENSOR_WRITE, nullptr);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uint64_t) rmr.remote_addr;
  wr.wr.rdma.rkey = rmr.rkey;
  /// Write Send Buf to Remote AggBuf
  struct ibv_send_wr* bad_wr_1;
  int ret = ibv_post_send(qp_, &wr, &bad_wr_1);
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc, 2);  // delete RdmaWrId
  //LOG(INFO) << "[DEBUG] STEP 3: CAS: to AggReady";
  // Transit State
  // Init Scatter & Gather List
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.atomic.remote_addr = (uint64_t) state_rmr.remote_addr;
  wr.wr.atomic.compare_add = StatefulAggBuf::kRecvInProgress;
  wr.wr.atomic.swap = StatefulAggBuf::kAggReady;
  wr.wr.atomic.rkey = state_rmr.rkey;
  // CAS State: RecvInProgress -> AggReady
#if 0
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
  struct ibv_send_wr* bad_wr_2;
  ret = ibv_post_send(qp_, &wr, &bad_wr_2);
  ptre_poll_cq(cq_, 1, &wc, 2);  // delete RdmaWrId
  usleep(1);
  if (*state_read_buf != StatefulAggBuf::kRecvInProgress) {
    LOG(INFO) << "[DEBUG] Remote AggBuf state has been manipulated by another thread: state_read_buf=" << *state_read_buf
        << ", kRecvInProgress=" << StatefulAggBuf::kRecvInProgress;
    exit(EXIT_FAILURE);
  }
#else
  cas_done = false;
  int wrong_state_cnt = 0;
  while (!cas_done) {
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "[DEBUG] Failed to ibv_post_send: " << ret;
      exit(1);
    }
    struct ibv_wc wc;
    ptre_poll_cq(cq_, 1, &wc, 2);  // delete RdmaWrId
    usleep(1);
    if (*state_read_buf == StatefulAggBuf::kRecvInProgress) {
      cas_done = true;
    } else {
      /// NOTE: This is workaround of resolving CAS read problem
      /// TODO: Solve the problem in a correct way.
      LOG(ERROR) << "[DEBUG] Remote AggBuf state has been manipulated by another thread: state_read_buf=" << *state_read_buf
         << ", kRecvInProgress=" << StatefulAggBuf::kRecvInProgress;
#if 0
      wrong_state_cnt++;
      if (wrong_state_cnt > 1000) {
        wr.wr.atomic.compare_add = *state_read_buf;
      }
#endif
    }
  }
#endif
}

int RdmaAggWriter::WriteToAggBufV2(const string& name) {
  int idx = name_to_index_[name];
#if 1
  //std::this_thread::sleep_for(std::chrono::seconds(1));
  //std::cout << "[DEBUG] WriteToAggBufV2 name=" << name << std::endl;
  /// STEP 2: Write
  /// Get Local Send Buf MR
  struct ibv_mr* send_buf_mr = send_buf_mrs_[idx];
  /// Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) send_buf_mr->addr;
  sge.length = send_buf_mr->length;
  sge.lkey = send_buf_mr->lkey;
  /// Get Remote AggBuf MR
  RemoteMR rmr = agg_buf_rmrs_[idx];
  /// Init WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(struct ibv_send_wr));
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_TENSOR_WRITE, nullptr);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uint64_t) rmr.remote_addr;
  wr.wr.rdma.rkey = rmr.rkey;
  /// Write Send Buf to Remote AggBuf
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  if (ret) {
    LOG(ERROR) << "[DEBUG] Failed to ibv_post_send: " << ret;
    exit(1);
  }
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc, 2);  // delete RdmaWrId
#endif
  // STEP 3: Transit State: RecvInProgress -> AggReady
  /// Get Remote AggBuf State Mr
  RemoteMR state_rmr = agg_buf_state_rmrs_[idx];
#if 0
  bool cas_done = false;
#else
  bool cas_done = true;
#endif
  auto start_time = std::chrono::system_clock::now();
  auto last_time = start_time;
  while (!cas_done) {
    uint64_t remote_state = TransitStateV2(name, StatefulAggBuf::kRecvInProgress,
        StatefulAggBuf::kAggReady);
    if (remote_state == StatefulAggBuf::kRecvInProgress) {
      cas_done = true;
    } else {
      auto curr_time = std::chrono::system_clock::now();
      std::chrono::duration<double> since_last = curr_time - last_time;
      if (since_last.count() > 5) {
        LOG(INFO) << "[DEBUG] The Last CAS not done for rank=" << dst_rank_ << ", read=" << remote_state << ", should be " << StatefulAggBuf::kRecvInProgress;
        last_time = curr_time;
      }
    }
  }
}

}  // namespcae ptre
