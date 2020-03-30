#include "ptre/cm/tensor_aggregator.h"

#include "tensorflow/core/platform/logging.h"

#include <cstdlib>

namespace ptre {

void AggregateSum(const Eigen::ThreadPoolDevice& d,
                  Flat target,
                  Flat buf) {
  target.device(d) = target + buf;
}

/// Constructor
TensorAggregator::TensorAggregator(Eigen::ThreadPool* pool, int pool_size,
      const std::vector<string>& names,
      const std::vector<Flat>& flats)
    : pool_(pool), pool_size_(pool_size), n_(flats.size()) {
  if (pool_ == nullptr) {
    pool_ = new Eigen::ThreadPool(32);
  }
  if (pool_size_ == 0) {
    pool_size_ = 32;
  }
  // Init names
  for (int i = 0; i < names.size(); i++) {
    names_.push_back(names[i]);
    name_to_index_.emplace(names[i], i);
  }
  // Init StatefulAggBuf
  for (auto& f : flats) {
    size_t num_bytes = sizeof(float) * f.size();
    float* buf = (float*) malloc(num_bytes);
    memset(buf, 0, num_bytes);
    Flat* buf_flat = new Flat(buf, f.size());
    buf_flats_.push_back(buf_flat);
    //buf_flats_.emplace_back(buf, f.size());
    StatefulAggBuf* agg_buf = new StatefulAggBuf();
    agg_buf->flat = buf_flat;
    //agg_buf->flat = buf_flats_.data() + buf_flats_.size() - 1;
    if (agg_buf->flat->data() == (void*) 0x1) {
      LOG(INFO) << buf_flats_.back()->size() << ", " << buf_flats_.back()->data();
      LOG(INFO) << agg_buf->flat;
      exit(EXIT_FAILURE);
    }
    target_buf_pairs_.emplace_back(f, agg_buf);
  }

  // Init aggregation count array.
  counts_ = (int*) malloc(sizeof(int) * n_);
  memset(counts_, 0, sizeof(int) * n_);

  // Init background aggregation thread.
  background_thread_ = std::thread([this] { BackgroundThreadLoop(); });

  // Init state.
  for (const auto& p : target_buf_pairs_) {
    p.second->state = StatefulAggBuf::kRecvReady;
  }
  state_ = kReady;
}

TensorAggregator::~TensorAggregator() {
  if (background_thread_.joinable()) {
    background_thread_.join();
  }
  // TODO: free recv bufs
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
  return &target_buf_pairs_[i].second->state;
}

uint64_t* TensorAggregator::state_ptr(const string& name) {
  int i = name_to_index_.find(name)->second;
  return &target_buf_pairs_[i].second->state;
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

void TensorAggregator::InitAggBufStates() {
  for (const auto& p : target_buf_pairs_) {
    p.second->state = StatefulAggBuf::kRecvReady;
    p.second->agg_done_cnt = 0;
  }
}

void TensorAggregator::BackgroundThreadLoop() {
  while (state_ != kTerminate) {
    for (const auto& p : target_buf_pairs_) {
      if (p.second->state == StatefulAggBuf::kAggReady) {
        p.second->state = StatefulAggBuf::kAggInProgress;
#if 1
        Eigen::ThreadPoolDevice d(pool_, pool_size_);
        // TODO: Check target buf state
        AggregateSum(d, p.first, *p.second->flat);
#elif 0
        for (int i = 0; i < p.first.size(); i++) {
          p.first.data()[i] += p.second->flat->data()[i];
        }
#else
        Eigen::ThreadPool pool(4);
        Eigen::ThreadPoolDevice d(&pool, 4);
        AggregateSum(d, p.first, *p.second->flat);
#endif
        p.second->agg_done_cnt++;
        p.second->state = StatefulAggBuf::kRecvReady;
      }
    }
  }
}

}  // namespace ptre
