#ifndef PTRE_CM_TENSOR_AGGREGATOR_H_
#define PTRE_CM_TENSOR_AGGREGATOR_H_

#define EIGEN_USE_THREADS

#include <string>
#include <thread>
#include <memory>
#include <utility>
#include <map>

#include "ptre/lib/types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

/// alias for Tensor::flat<T>()
/// template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
/// typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
/// Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>

namespace ptre {

using std::string;
typedef Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> Flat;

struct StatefulAggBuf {
  /// Remote can change this state
  /// kRecvReady -> kRecvInProgress
  /// kRecvInProgress -> kAggReady
  enum State {
    kInit,
    kRecvReady,
    kRecvInProgress,
    kAggReady,
    kAggInProgress
  };
  uint64_t state = kInit;
  Flat* flat;
  int count = 0;
};

using TargetBufPair = std::pair<Flat, StatefulAggBuf*>;

class TensorAggregator {
 public:
  TensorAggregator(Eigen::ThreadPool* pool, int pool_size,
      const std::vector<string>& names,
      const std::vector<Flat>& flats);
  ~TensorAggregator();

  // State Transition
  void Terminate() { state_ = kTerminate; }

  // Element Access Functions
  float* buf_ptr(int i);
  float* buf_ptr(const string& name);
  StatefulAggBuf* agg_buf_ptr(int i);

 protected:
  void BackgroundThreadLoop();

  std::thread background_thread_;
  Eigen::ThreadPool* pool_;
  int pool_size_;  // number of threads for Eigen::ThreadPool

  enum State {
    kInit,
    kReady,
    kInProgress,
    kDone,
    kTerminate
  };
  uint64_t state_ = kInit;
  int n_;
  int* counts_;

  // Data
  std::map<string, int> name_to_index_;
  std::vector<TargetBufPair> target_buf_pairs_;
  std::vector<string> names_;
  std::vector<Flat> buf_flats_;
};

}  // namespace ptre

#endif  // PTRE_CM_TENSOR_AGGREGATOR_H_
