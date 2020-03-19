#include "ptre/cm/tensor_aggregator.h"

#include <cstdlib>

namespace ptre {

void AggregateSum(const Eigen::ThreadPoolDevice& d,
                  Flat target,
                  Flat buf) {
  target.device(d) = target + buf;
}

float* TensorAggregator::buf_ptr(int i) {
  return agg_buf_ptr(i)->flat->data();
}

float* TensorAggregator::buf_ptr(const string& name) {
  auto it = name_to_index_.find(name);
  return agg_buf_ptr(it->second)->flat->data();
}

StatefulAggBuf* TensorAggregator::agg_buf_ptr(int i) {
  return target_buf_pairs_[i].second;
}

/// Constructor
TensorAggregator::TensorAggregator(Eigen::ThreadPool* pool, int pool_size,
      const std::vector<string>& names,
      const std::vector<Flat>& flats)
    : pool_(pool), pool_size_(pool_size), n_(flats.size()) {
  // Init names
  for (int i = 0; i < names.size(); i++) {
    const string& name = names[i];
    names_.push_back(name);
    name_to_index_.emplace(name, i);
  }
  // Init StatefulAggBuf
  for (const auto& f : flats) {
    size_t num_bytes = sizeof(float) * f.size();
    float* buf = (float*) malloc(num_bytes);
    memset(buf, 0, num_bytes);
    buf_flats_.emplace_back(buf, f.size());
    auto agg_buf = new StatefulAggBuf();
    agg_buf->flat = &buf_flats_.back();
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

void TensorAggregator::BackgroundThreadLoop() {
  while (state_ != kTerminate) {
    for (const auto& p : target_buf_pairs_) {
      if (p.second->state == StatefulAggBuf::kAggReady) {
        p.second->state = StatefulAggBuf::kAggInProgress;
        Eigen::ThreadPoolDevice d(pool_, pool_size_);
        AggregateSum(d, p.first, *p.second->flat);
        p.second->count++;
        p.second->state = StatefulAggBuf::kRecvReady;
      }
    }
  }
}

}  // namespace ptre
