#include "ptre/common/cm/tensor_aggregator.h"
#include "ptre/lib/cache_ctl.h"

#include "tensorflow/core/platform/logging.h"

#include <cstdlib>
#include <chrono>
//#include <asm/cachectl.h>

namespace ptre {

void AggregateSum(const Eigen::ThreadPoolDevice& d,
                  Flat target,
                  Flat buf) {
  target.device(d) = target + buf;
}

void AggregateSumSingle(Flat target, Flat buf) {
  target = target + buf;
}

/// Constructor
TensorAggregator::TensorAggregator(Eigen::ThreadPool* pool, int pool_size,
      RdmaEnv* rdma_env,
      struct ibv_cq* cq, struct ibv_qp* qp,
      const std::vector<string>& names,
      const std::vector<Flat>& flats)
    : rdma_env_(rdma_env), pool_(pool), pool_size_(pool_size),
      n_(flats.size()) {
  //cq_ = cq;
  //LOG(INFO) << "cq=" << cq_;
  //qp_ = qp;
  if (pool_ == nullptr) {
    pool_ = new Eigen::ThreadPool(DEFAULT_THREAD_POOL_SIZE);
  }
  if (pool_size_ == 0) {
    pool_size_ = DEFAULT_THREAD_POOL_SIZE;
  }
  // Init names
  for (int i = 0; i < names.size(); i++) {
    names_.push_back(names[i]);
    name_to_index_.emplace(names[i], i);
  }
  // Init StatefulAggBuf
#if 1
  for (int i = 0; i < n_; i++) {
    size_t num_bytes = sizeof(float) * flats[i].size();
    float* buf = (float*) malloc(num_bytes);
    memset(buf, 0, num_bytes);
    Flat* buf_flat = new Flat(buf, flats[i].size());
    buf_flats_.push_back(buf_flat);
    StatefulAggBuf* agg_buf = new StatefulAggBuf();
    uint64_t* state = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
    *state = 1;
    cache_ctl::clflush((char*) state, 8);
    //std::atomic<uint64_t>* state = (std::atomic<uint64_t>*) aligned_alloc(8,
    //    sizeof(std::atomic<uint64_t>));
    buf_states_.push_back(state);
    agg_buf->state = state;
    agg_buf->flat = buf_flat;
    target_buf_pairs_.emplace_back(flats[i], agg_buf);
    Flat* glc_flat = new Flat(flats[i].data(), flats[i].size());
    glc_flats_.push_back(glc_flat);
  }
#endif

  // Init state.
  for (int i = 0; i < n_; i++) {
    memset(buf_states_[i], 1, 1);
    cache_ctl::clflush((char*) buf_states_[i], 8);
  }
  state_ = kReady;

  for (int i = 0; i < n_; i++) {
    buf_state_mrs_.push_back(nullptr);
  }

}

TensorAggregator::~TensorAggregator() {
  // TODO: free recv bufs
}

void TensorAggregator::InitReceive() {
  for (int i = 0; i < comm_size_; i++) {
    done_tensor_cnts_[i] = 0;
  }
}

void TensorAggregator::SetStateMR(const string& name, struct ibv_mr* state_mr) {
  int idx = name_to_index_[name];
  buf_state_mrs_[idx] = state_mr;
}

float* TensorAggregator::buf_ptr(int i) {
  return agg_buf_ptr(i)->flat->data();
}

float* TensorAggregator::buf_ptr(const string& name) {
  int idx = name_to_index_[name];
  return agg_buf_ptr(idx)->flat->data();
}

size_t TensorAggregator::buf_length(const string& name) {
  int i = name_to_index_[name];
  Flat* buf_flat = buf_flats_[i];
  return buf_flat->size() * sizeof(float);
}

uint64_t* TensorAggregator::state_ptr(int i) {
  return (uint64_t*) target_buf_pairs_[i].second->state;
}

uint64_t* TensorAggregator::state_ptr(const string& name) {
  int i = name_to_index_.find(name)->second;
  return (uint64_t*) target_buf_pairs_[i].second->state;
}

StatefulAggBuf* TensorAggregator::agg_buf_ptr(int i) {
  StatefulAggBuf* agg_buf = target_buf_pairs_[i].second;
  if (agg_buf->flat->data() == (void*) 0x1) {
    LOG(INFO) << buf_flats_[i] << ", " << buf_flats_[i]->size() << ", " << buf_flats_[i]->data();
    LOG(INFO) << agg_buf << ", " << agg_buf->flat->size() << ", " << agg_buf->flat->data();
    exit(EXIT_FAILURE);
  }
  return target_buf_pairs_[i].second;
}

int TensorAggregator::agg_done_cnt(const string& name) {
  int idx = name_to_index_[name];
  return target_buf_pairs_[idx].second->agg_done_cnt;
}

void TensorAggregator::InitQp(struct ibv_context* ctx, struct ibv_pd* pd) {
  int ret;
  // Query Port
  struct ibv_port_attr port_attr;
  ret = ibv_query_port(ctx, 1, &port_attr);
  if (ret) {
    LOG(ERROR) << "Failed to query port";
    exit(1);
  }
  // Create local CQ
  cq_ = ibv_create_cq(ctx, 100, NULL, NULL, 0);
  if (!cq_) {
    LOG(ERROR) << "Failed to create local CQ";
    exit(1);
  }
  // Create local QP with local CQ
  struct ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(ibv_qp_init_attr));
  qp_init_attr.send_cq = cq_;
  qp_init_attr.recv_cq = cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.cap.max_send_wr = 16;
  qp_init_attr.cap.max_recv_wr = 16;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_ = ibv_create_qp(pd, &qp_init_attr);
  if (!qp_) {
    LOG(ERROR) << "Failed to create local QP";
    exit(1);
  }
  // Init QP
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE
      | IBV_ACCESS_REMOTE_WRITE
      | IBV_ACCESS_REMOTE_READ
      | IBV_ACCESS_REMOTE_ATOMIC;
  ret = ibv_modify_qp(qp_, &attr,
        IBV_QP_STATE
      | IBV_QP_PKEY_INDEX
      | IBV_QP_PORT
      | IBV_QP_ACCESS_FLAGS);
  if (ret) {
    LOG(ERROR) << "Failed to modify local QP to INIT: " << std::strerror(ret);
    exit(1);
  }
  // INIT -> RTR
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = qp_->qp_num;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = port_attr.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num  = IB_PORT;
  ret = ibv_modify_qp(qp_, &attr,
        IBV_QP_STATE
      | IBV_QP_AV
      | IBV_QP_PATH_MTU
      | IBV_QP_DEST_QPN
      | IBV_QP_RQ_PSN
      | IBV_QP_MAX_DEST_RD_ATOMIC
      | IBV_QP_MIN_RNR_TIMER);
  if (ret) {
    LOG(ERROR) << "Failed to modify local QP to RTR: " << std::strerror(ret);
    exit(1);
  }
  /// RTR -> RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  ret = ibv_modify_qp(qp_, &attr,
        IBV_QP_STATE
      | IBV_QP_TIMEOUT
      | IBV_QP_RETRY_CNT
      | IBV_QP_RNR_RETRY
      | IBV_QP_SQ_PSN
      | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret) {
    LOG(ERROR) << "Failed to modify local QP to RTS: " << std::strerror(ret);
    exit(1);
  }
}

uint64_t TensorAggregator::TransitState(const string& name, const uint64_t from,
    const uint64_t to) {
  if (qp_ == nullptr) {
    LOG(ERROR) << "Initialize QP first. Terminating";
    exit(1);
  }
  int idx = name_to_index_[name];
  uint64_t read_state;
  uint64_t* state_read_buf = &read_state;
  struct ibv_mr* state_read_buf_mr = ibv_reg_mr(rdma_env_->pd,
      (void*) state_read_buf, sizeof(uint64_t), IBV_ACCESS_LOCAL_WRITE);
  /// Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  /// Init WR
  struct ibv_send_wr wr;
  /// Try CAS
  int ret;
  while (true) {
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.atomic.remote_addr = (uint64_t) buf_state_mrs_[idx]->addr;
    wr.wr.atomic.compare_add = from;
    wr.wr.atomic.swap = to;
    wr.wr.atomic.rkey = buf_state_mrs_[idx]->rkey;
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp_, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "Failed to post send CAS: " << std::strerror(ret);
      exit(1);
    }
    struct ibv_wc wc;
    //LOG(INFO) << "[DEBUG] Starting poll";
    ptre_poll_cq(cq_, 1, &wc, 1);  // delete RdmaWrId
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
  ret = ibv_dereg_mr(state_read_buf_mr);
  return read_state;
}

void TensorAggregator::InitAggBufStates() {
  for (const auto& p : target_buf_pairs_) {
    p.second->agg_done_cnt = 0;
    //*p.second->state = StatefulAggBuf::kRecvReady;
  }
}

void TensorAggregator::PrintDebug(int compare) {
  for (int i = 0; i < names_.size(); i++) {
    if (*target_buf_pairs_[i].second->state != compare) {
    LOG(INFO) << "[DEBUG] TensorAggregator: name=" << names_[i] << ", state=" << *target_buf_pairs_[i].second->state;
    }
  }
}

int TensorAggregator::ProcessAggregationNoVerbs() {
  int agg_cnt = 0;
  Eigen::ThreadPoolDevice d(pool_, pool_size_);
  for (int i = 0; i < n_; i++) {
    if (*buf_states_[i] == 3) {
      *buf_states_[i] = 4;
      cache_ctl::clflush((char*) buf_states_[i], 8);
      AggregateSum(d, *glc_flats_[i], *buf_flats_[i]);
      target_buf_pairs_[i].second->agg_done_cnt++;
      *buf_states_[i] = 1;
      cache_ctl::clflush((char*) buf_states_[i], 8);
      agg_cnt++;
    }
  }
  return agg_cnt;
}

int TensorAggregator::ProcessAggregation() {
  int agg_cnt = 0;
  Eigen::ThreadPoolDevice d(pool_, pool_size_);
  for (int i = 0; i < n_; i++) {
    uint64_t read_state = TransitState(names_[i], 3, 4);
    if (read_state == 3) {
      AggregateSum(d, *glc_flats_[i], *buf_flats_[i]);
      target_buf_pairs_[i].second->agg_done_cnt++;
      while (true) {
        uint64_t read_state_inner = TransitState(names_[i], 4, 1);
        if (read_state_inner == 4) {
          break;
        } else {
          LOG(INFO) << "[DEBUG] read_state_inner = " << read_state_inner;
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }
      agg_cnt++;
    }
  }
  return agg_cnt;
}

void TensorAggregator::EnqueuePeer(int src_rank) {
  peer_q_mu_.lock();
  for (int i = 0; i < n_; i++) {
    peer_q_[i].push(src_rank);
  }
  peer_q_mu_.unlock();
}

int TensorAggregator::NextPeerToReceive(int idx) {
  int ret;
  peer_q_mu_.lock();
  std::queue<int>& q = peer_q_[idx];
  if (q.size() == 0) {
    ret = -1;
  } else {
    ret = q.front();
    q.pop();
  }
  peer_q_mu_.unlock();
  return ret;
}

void TensorAggregator::StartReceive() {
  // Shift queue elements for load balancing
  peer_q_mu_.lock();
  for (int i = 0; i < n_; i++) {
    std::queue<int>& q = peer_q_[i];
    for (int j = 0; j < (i % comm_size_); j++) {
      int front = q.front();
      q.pop();
      q.push(front);
    }
  }
  peer_q_mu_.unlock();
  // Init done_arr_
  // Remote peer will read done_arr_ to decide whether it will write or not
  done_arr_mu_.lock();
  for (int i = 0; i < n_; i++) {
    done_arr_[idx] = NextPeerToReceive(idx);
  }
  done_arr_mu_.unlock();
  state_mu_.lock();
  state_ = kInProgress;
  state_mu_.unlock();
}

void TensorAggregator::ReceiveWriteDone(int dst) {
  if (state_ != kInProgress) {
    return;
  }
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
      AggregateSum(d, *glc_flats_[idx], *buf_flats_[idx]);
      target_buf_pairs_[i].second->agg_done_cnt++;
      done_arr_mu_.lock();
      done_arr_[idx] = NextPeerToReceive(idx);
      cache_ctl::clflush((char*) done_arr_[idx], sizeof(int));

      done_tensor_cnts_[dst]++;
      if (done_tensor_cnts_[dst] == n_) {
        done_peer_cnt_++;
      }
      done_arr_mu_.unlock();
    } else {
      LOG(ERROR) << "Failed to poll recv CQ for rank=" << dst << ": status="
          << wc.status;
      exit(EXIT_FAILURE);
    }
  }
}

}  // namespace ptre
