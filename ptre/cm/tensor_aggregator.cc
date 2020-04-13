#include "ptre/cm/tensor_aggregator.h"
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

/// Constructor
TensorAggregator::TensorAggregator(Eigen::ThreadPool* pool, int pool_size,
      RdmaEnv* rdma_env,
      const std::vector<string>& names,
      const std::vector<Flat>& flats)
    : rdma_env_(rdma_env), pool_(pool), pool_size_(pool_size),
      n_(flats.size()) {
  if (pool_ == nullptr) {
    pool_ = new Eigen::ThreadPool(DEFAULT_THREAD_POOL_SIZE);
  }
  if (pool_size_ == 0) {
    pool_size_ = DEFAULT_THREAD_POOL_SIZE;
    //d_ = new Eigen::ThreadPoolDevice(pool_, pool_size_);
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
    //std::atomic<uint64_t>* state = (std::atomic<uint64_t>*) aligned_alloc(8,
    //    sizeof(std::atomic<uint64_t>));
    buf_states_.push_back(state);
    agg_buf->state = state;
    agg_buf->flat = buf_flat;
    target_buf_pairs_.emplace_back(flats[i], agg_buf);
    Flat* glc_flat = new Flat(flats[i].data(), flats[i].size());
    glc_flats_.push_back(glc_flat);
  }
#elif 0
  buf_states_ = (uint64_t*) malloc(sizeof(uint64_t) * n_);
  for (int i = 0; i < n_; i++) {
    size_t num_bytes = sizeof(float) * flats[i].size();
    float* buf = (float*) malloc(num_bytes);
    memset(buf, 0, num_bytes);
    Flat* buf_flat = new Flat(buf, flats[i].size());
    buf_flats_.push_back(buf_flat);
    StatefulAggBuf* agg_buf = new StatefulAggBuf();
    agg_buf->state = buf_states_ + i;
    agg_buf->flat = buf_flat;
    target_buf_pairs_.emplace_back(flats[i], agg_buf);
  }
#else
  for (auto& f : flats) {
    uint64_t* state = (uint64_t*) malloc(sizeof(uint64_t));
    size_t num_bytes = sizeof(float) * f.size();
    float* buf = (float*) malloc(num_bytes);
    memset(buf, 0, num_bytes);
    Flat* buf_flat = new Flat(buf, f.size());
    buf_flats_.push_back(buf_flat);
    //buf_flats_.emplace_back(buf, f.size());
    StatefulAggBuf* agg_buf = new StatefulAggBuf();
    agg_buf->state = state;
    agg_buf->flat = buf_flat;
    //agg_buf->flat = buf_flats_.data() + buf_flats_.size() - 1;
    if (agg_buf->flat->data() == (void*) 0x1) {
      LOG(INFO) << buf_flats_.back()->size() << ", " << buf_flats_.back()->data();
      LOG(INFO) << agg_buf->flat;
      exit(EXIT_FAILURE);
    }
    target_buf_pairs_.emplace_back(f, agg_buf);
  }
#endif

  // Init aggregation count array.
  counts_ = (int*) malloc(sizeof(int) * n_);
  memset(counts_, 0, sizeof(int) * n_);

  // Init state.
  for (auto& p : target_buf_pairs_) {
    *p.second->state = StatefulAggBuf::kRecvReady;
  }
  state_ = kReady;

  for (int i = 0; i < n_; i++) {
    buf_state_mrs_.push_back(nullptr);
  }

  /// Create CQ and QP
#if 0
  cq_ = ptre_rdma_create_cq(rdma_env_, 2);
  qp_ = ptre_rdma_create_qp(rdma_env_, cq_, cq_);
  int conn_ret = ptre_rdma_connect_qp(qp_, qp_->qp_num,
      rdma_env_->gid.global.subnet_prefix, rdma_env_->gid.global.interface_id,
      rdma_env_->port_attr.lid);
#endif
}

TensorAggregator::~TensorAggregator() {
  if (background_thread_.joinable()) {
    background_thread_.join();
  }
  // TODO: free recv bufs
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

int TensorAggregator::TransitState(const string& name, const uint64_t from,
    const uint64_t to) {
  int idx = name_to_index_[name];
  uint64_t* state_read_buf = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
  int access = IBV_ACCESS_LOCAL_WRITE;
  struct ibv_mr* state_read_buf_mr = ibv_reg_mr(rdma_env_->pd,
      (void*) state_read_buf, sizeof(uint64_t), access);
  /// Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  /// Init WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(struct ibv_send_wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.atomic.remote_addr = (uint64_t) buf_state_mrs_[idx]->addr;
  wr.wr.atomic.compare_add = from;
  wr.wr.atomic.swap = to;
  wr.wr.atomic.rkey = buf_state_mrs_[idx]->rkey;
  /// Try CAS
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  if (ret < 0) {
    LOG(INFO) << "[DEBUG] ibv_post_send failed: " << ret;
  }
  struct ibv_wc wc;
#if 1
  LOG(INFO) << "[DEBUG] Starting poll";
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
  LOG(INFO) << "[DEBUG] Poll Done.";
#else
  delete reinterpret_cast<RdmaWrId*>(wr.wr_id);
#endif
  usleep(1);
  ret = ibv_dereg_mr(state_read_buf_mr);
  uint64_t read_state = *state_read_buf;
  free(state_read_buf);
  return read_state;
}

void TensorAggregator::InitAggBufStates() {
  for (const auto& p : target_buf_pairs_) {
    p.second->agg_done_cnt = 0;
    *p.second->state = StatefulAggBuf::kRecvReady;
  }
}

void TensorAggregator::Start() {
  // Init background aggregation thread.
  background_thread_ = std::thread([this] { BackgroundThreadLoop(); });
}

void TensorAggregator::PrintDebug(int compare) {
  for (int i = 0; i < names_.size(); i++) {
    if (*target_buf_pairs_[i].second->state != compare) {
    LOG(INFO) << "[DEBUG] TensorAggregator: name=" << names_[i] << ", state=" << *target_buf_pairs_[i].second->state;
    }
  }
}

int TensorAggregator::ProcessAggregation() {
  int agg_cnt = 0;
#if 0
  Eigen::ThreadPoolDevice d(pool_, pool_size_);
  for (int i = 0; i < n_; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    uint64_t expected;
    uint64_t val;
    bool ret;
    expected = 3;
    val = 4;
    ret = buf_states_[i]->compare_exchange_weak(expected, val, std::memory_order_relaxed);
    if (ret) {
      //LOG(INFO) << "[DEBUG] aggregating " << names_[i] << ", state=" << *buf_states_[i];
      //memset(buf_states_[i], 4, 1);
      AggregateSum(d, *glc_flats_[i], *buf_flats_[i]);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    expected = 4;
    val = 5;
    ret = buf_states_[i]->compare_exchange_weak(expected, val, std::memory_order_relaxed);
    if (ret) {
      //memset(buf_states_[i], 5, 1);
      agg_cnt++;
      target_buf_pairs_[i].second->agg_done_cnt++;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    expected = 5;
    val = 1;
    ret = buf_states_[i]->compare_exchange_weak(expected, val, std::memory_order_relaxed);
    if (ret) {
      //memset(buf_states_[i], 1, 1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
#else
  Eigen::ThreadPoolDevice d(pool_, pool_size_);
  uint64_t read;
  for (int i = 0; i < n_; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    //read = buf_states_[i]->load();
    read = *buf_states_[i];
    if (read == 3) {
      //buf_states_[i]->store(4);
      //memset(buf_states_[i], 4, 1);
      *buf_states_[i] = 4;
      //cache_ctl::clflush((char*) buf_states_[i], sizeof(uint64_t));
      cache_ctl::clflush((char*) buf_states_[i], 64);
      AggregateSum(d, *glc_flats_[i], *buf_flats_[i]);
      target_buf_pairs_[i].second->agg_done_cnt++;
      //buf_states_[i]->store(1);
      cache_ctl::mfence();
      //memset(buf_states_[i], 1, 1);
      *buf_states_[i] = 1;
      //cache_ctl::clflush((char*) buf_states_[i], sizeof(uint64_t));
      cache_ctl::clflush((char*) buf_states_[i], 64);
      agg_cnt++;
    } else if (read == 4) {
      //memset(buf_states_[i], 1, 1);
    }
    /*
    else if (read == 4) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      buf_states_[i]->store(1);
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    */
  }
#endif
  return agg_cnt;
}

void TensorAggregator::BackgroundThreadLoop() {
  cq_ = ptre_rdma_create_cq(rdma_env_, 2);
  qp_ = ptre_rdma_create_qp(rdma_env_, cq_, cq_);
  int conn_ret = ptre_rdma_connect_qp(qp_, qp_->qp_num,
      rdma_env_->gid.global.subnet_prefix, rdma_env_->gid.global.interface_id,
      rdma_env_->port_attr.lid);
  if (conn_ret < 0) {
    LOG(INFO) << "[DEBUG] failed to connect qp: " << conn_ret;
  }
  usleep (15 * 1000 * 1000);
  void* dummy = aligned_alloc(64, 64);
  auto start_time = std::chrono::system_clock::now();
  auto last_time = start_time;
  uint64_t read_state[n_] = { };
  while (state_ != kTerminate) {
    auto curr_time = std::chrono::system_clock::now();
    std::chrono::duration<double> since_last = curr_time - last_time;
    if (since_last.count() > 30) {
      LOG(INFO) << "[DEBUG]: Agg not performed for a while";
      for (int i = 0; i < names_.size(); i++) {
        if (*target_buf_pairs_[i].second->state != 1) {
        LOG(INFO) << "[DEBUG] TensorAggregator: name=" << names_[i] << ", state=" << *target_buf_pairs_[i].second->state << ", read_state=" << read_state[i];
        }
      }
      last_time = curr_time;
    }
#if 0
    int i = 0;
    for (auto p : target_buf_pairs_) {
      memset(dummy, 2, 64);
      usleep(1);
      //LOG(INFO) << "[DEBUG] buf name=" << names_[i] << ", state=" << *p.second->state;
      if (*p.second->state == StatefulAggBuf::kAggReady) {
        //*p.second->state = StatefulAggBuf::kAggInProgress;
        while (*p.second->state == StatefulAggBuf::kAggReady) {
          memset(p.second->state, 4, 1);
        }
        //memset(p.second->state, 4, 1);
        memset(dummy, 0, 64);
#if 1
        // TODO: Check target buf state
        //AggregateSum(*d_, p.first, *p.second->flat);
        Eigen::ThreadPoolDevice d(pool_, pool_size_);
        AggregateSum(d, p.first, *p.second->flat);
#elif 0
        for (int i = 0; i < p.first.size(); i++) {
          p.first.data()[i] += p.second->flat->data()[i];
        }
#else
        //Flat b(p.second->flat->data(), p.first.size());
        p.first = p.first + *p.second->flat;
        //Eigen::ThreadPool pool(4);
        //Eigen::ThreadPoolDevice d(&pool, 4);
        //AggregateSum(d, p.first, *p.second->flat);
#endif
        p.second->agg_done_cnt++;
        //*p.second->state = StatefulAggBuf::kRecvReady;
        //cacheflush((char*) p.second->state, 8, DCACHE);
        usleep(1);
        while (*p.second->state == StatefulAggBuf::kAggInProgress) {
          memset(p.second->state, 1, 1);
        }
        //memset(dummy, 1, 64);
        //char* addr = (char*) p.second->state + 1;
        //memset(addr, 0, 7);
      }
      i++;
    }
    //usleep(1000 * 1000);
#elif 0
    for (int i = 0; i < n_; i++) {
      //uint64_t read_state = TransitState(names_[i], 3, 4);
      read_state[i] = TransitState(names_[i], 3, 4);
      if (read_state[i] == 3) {
        Eigen::ThreadPoolDevice d(pool_, pool_size_);
        AggregateSum(d, *glc_flats_[i], *buf_flats_[i]);
        target_buf_pairs_[i].second->agg_done_cnt++;
        //agg_done_cnts_[i]++;
        uint64_t read_state_inner = TransitState(names_[i], 4, 1);
        //while (true) {
        //  uint64_t read_state_inner = TransitState(names_[i], 4, 1);
        //  if (read_state_inner == 4) {
        //    break;
        //  } else {
        //    LOG(INFO) << "[DEBUG] read_state_inner = " << read_state_inner;
        //    usleep(5 * 1000 * 1000);
        //  }
        //}
        last_time = curr_time;
      }
    }
#elif 0
    for (int i = 0; i < n_; i++) {
      usleep(1);
      uint64_t old_val = *buf_states_[i];
      /*
      if (old_val == 3) {
        memset(buf_states_[i], 4, 1);
        //*buf_states_[i] = 4;
      }
      */
      if (old_val == 3) {
        memset(buf_states_[i], 4, 1);
      } else if (old_val == 4) {
        Eigen::ThreadPoolDevice d(pool_, pool_size_);
        AggregateSum(d, *glc_flats_[i], *buf_flats_[i]);
        /*
        uint64_t old_val_inner = *buf_states_[i];
        if (old_val_inner == 4) {
          memset(buf_states_[i], 1, 1);
        } else {
          LOG(ERROR) << "Something's wrong: old_val_inner=" << old_val_inner;
          exit(EXIT_FAILURE);
        }
        */
        memset(buf_states_[i], 5, 1);
      } else if (old_val == 5) {
        target_buf_pairs_[i].second->agg_done_cnt++;
        memset(buf_states_[i], 1, 1);
        /*
        LOG(ERROR) << "Something's wrong: old_val=" << old_val;
        exit(EXIT_FAILURE);
        */
        last_time = curr_time;
      }
    }
#endif
  }
}

}  // namespace ptre
