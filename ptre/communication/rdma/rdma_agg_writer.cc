#include "ptre/communication/rdma/rdma_agg_writer.h"

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
  if (ret < 0) {
  }
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
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

int RdmaAggWriter::TransitStateV2(const string& name, const uint64_t from,
    const uint64_t to) {
  int idx = name_to_index_[name];
  //uint64_t* state_read_buf = (uint64_t*) malloc(sizeof(uint64_t));
  uint64_t* state_read_buf = (uint64_t*) aligned_alloc(64, sizeof(uint64_t));
  int access = IBV_ACCESS_LOCAL_WRITE;
  struct ibv_mr* state_read_buf_mr = ibv_reg_mr(pd_, (void*) state_read_buf,
      sizeof(uint64_t), access);
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
  //wr.wr.atomic.compare_add = StatefulAggBuf::kRecvReady;
  wr.wr.atomic.compare_add = from;
  //wr.wr.atomic.swap = StatefulAggBuf::kRecvInProgress;
  wr.wr.atomic.swap = to;
  wr.wr.atomic.rkey = state_rmr.rkey;
  /// Try CAS
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  usleep(1);
  if (ret < 0) {
  }
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
  usleep(1);
  uint64_t read_state = *state_read_buf;
  ret = ibv_dereg_mr(state_read_buf_mr);
  free(state_read_buf);
#if 0
  if (*state_read_buf == StatefulAggBuf::kRecvReady) {
    return 0;
  }
  return -1;
#else
  return read_state;
#endif
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
    if (ret < 0) {
    }
    struct ibv_wc wc;
    ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
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
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
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
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
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
    if (ret < 0) {
    }
    struct ibv_wc wc;
    ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
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
  //std::cout << "[DEBUG] WriteToAggBufV2 name=" << name << std::endl;
  int idx = name_to_index_[name];
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
  usleep(1);
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
  usleep(1);
  // STEP 3: Transit State: RecvInProgress -> AggReady
  /// Get Remote AggBuf State Mr
  RemoteMR state_rmr = agg_buf_state_rmrs_[idx];
#if 0
  // Init Scatter & Gather List
  struct ibv_sge sge1;
  memset(&sge1, 0, sizeof(struct ibv_sge));
  struct ibv_mr* state_read_buf_mr = state_read_buf_mrs_[idx];
  uint64_t* state_read_buf = (uint64_t*) state_read_buf_mr->addr;
  *state_read_buf = StatefulAggBuf::kAggReady;
  sge1.addr = (uint64_t) state_read_buf_mr->addr;
  sge1.length = state_read_buf_mr->length;
  sge1.lkey = state_read_buf_mr->lkey;
  struct ibv_send_wr wr1;
  memset(&wr1, 0, sizeof(struct ibv_send_wr));
  wr1.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_WRITE_TENSOR_AGG_STATE,
                                      nullptr);
  wr1.sg_list = &sge;
  wr1.num_sge = 1;
  wr1.opcode = IBV_WR_RDMA_WRITE;
  wr1.send_flags = IBV_SEND_SIGNALED;
  wr1.wr.rdma.remote_addr = (uint64_t) state_rmr.remote_addr;
  wr1.wr.rdma.rkey = state_rmr.rkey;
  /// Write Send Buf to Remote AggBuf
  struct ibv_send_wr* bad_wr1;
  if (idx == 0) {
    LOG(INFO) << "[DEBUG] ibv_post_send RDMA_WRITE STATE WR";
  }
  int ret1 = ibv_post_send(qp_, &wr1, &bad_wr1);
  struct ibv_wc wc1;
  if (idx == 0) {
    LOG(INFO) << "[DEBUG] Polling RDMA_WRITE STATE WR";
  }
  ptre_poll_cq(cq_, 1, &wc1);  // delete RdmaWrId
  if (idx == 0) {
    LOG(INFO) << "[DEBUG] Poll done.";
  }
#elif 0
  // Init Scatter & Gather List
  struct ibv_sge sge1;
  memset(&sge1, 0, sizeof(struct ibv_sge));
  struct ibv_mr* state_read_buf_mr = state_read_buf_mrs_[idx];
  sge1.addr = (uint64_t) state_read_buf_mr->addr;
  sge1.length = state_read_buf_mr->length;
  sge1.lkey = state_read_buf_mr->lkey;
  struct ibv_send_wr wr1;
  memset(&wr1, 0, sizeof(struct ibv_send_wr));
  wr1.sg_list = &sge;
  wr1.num_sge = 1;
  wr1.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr1.send_flags = IBV_SEND_SIGNALED;
  wr1.wr.atomic.remote_addr = (uint64_t) state_rmr.remote_addr;
  wr1.wr.atomic.compare_add = StatefulAggBuf::kRecvInProgress;
  wr1.wr.atomic.swap = StatefulAggBuf::kAggReady;
  wr1.wr.atomic.rkey = state_rmr.rkey;
  /// Write Send Buf to Remote AggBuf
  struct ibv_send_wr* bad_wr1;
  bool cas_done = false;
  auto start_time = std::chrono::system_clock::now();
  auto last_time = start_time;
  while (cas_done) {
    wr1.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
    int ret1 = ibv_post_send(qp_, &wr1, &bad_wr1);
    struct ibv_wc wc1;
    ptre_poll_cq(cq_, 1, &wc1);  // delete RdmaWrId
    usleep(1);
    uint64_t* state_read_buf = (uint64_t*) state_read_buf_mr->addr;
    if (*state_read_buf != StatefulAggBuf::kRecvInProgress) {
      auto curr_time = std::chrono::system_clock::now();
      std::chrono::duration<double> since_last = curr_time - last_time;
      if (since_last.count() > 5) {
        LOG(INFO) << "[DEBUG] Last CAS not done read=" << *state_read_buf << ", should be " << StatefulAggBuf::kRecvInProgress;
        last_time = curr_time;
      }
      //LOG(INFO) << "[DEBUG] Another thread manipulated buf state!!!! read=" << *state_read_buf << ", should be " << StatefulAggBuf::kRecvInProgress;
    } else {
      cas_done = true;
    }
  }
#else
  bool cas_done = false;
  auto start_time = std::chrono::system_clock::now();
  auto last_time = start_time;
  while (!cas_done) {
    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    int remote_state = TransitStateV2(name, StatefulAggBuf::kRecvInProgress,
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
#endif
}

}  // namespcae ptre
