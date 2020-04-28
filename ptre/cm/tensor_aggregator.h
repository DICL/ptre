#ifndef PTRE_CM_TENSOR_AGGREGATOR_H_
#define PTRE_CM_TENSOR_AGGREGATOR_H_

#define EIGEN_USE_THREADS

#include <atomic>
#include <string>
#include <thread>
#include <memory>
#include <utility>
#include <map>

#include "ptre/lib/types.h"
#include "ptre/communication/rdma/rdma.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define DEFAULT_THREAD_POOL_SIZE 16

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
    kAggInProgress,
    kAggDone
  };
  uint64_t* state;
  //std::atomic<uint64_t>* state;
  Flat* flat;
  uint64_t agg_done_cnt = 0;
};

using TargetBufPair = std::pair<Flat, StatefulAggBuf*>;

class TensorAggregator {
 public:
  TensorAggregator(Eigen::ThreadPool* pool, int pool_size,
      RdmaEnv* rdma_env,
      struct ibv_cq* cq, struct ibv_qp* qp,
      const std::vector<string>& names,
      const std::vector<Flat>& flats);
  ~TensorAggregator();

  void SetStateMR(const string& name, struct ibv_mr* state_mr);

  // Element Access Functions
  float* buf_ptr(int i);
  float* buf_ptr(const string& name);
  size_t buf_length(const string& name);
  uint64_t* state_ptr(int i);
  uint64_t* state_ptr(const string& name);
  StatefulAggBuf* agg_buf_ptr(int i);
  int agg_done_cnt(const string& name);

  // State Transition
  void InitQp(struct ibv_context* ctx, struct ibv_pd* pd);
  void Start();
  void Terminate() { state_ = kTerminate; }
  // State Transition of StatefulAggBuf
  uint64_t TransitState(const string& name, const uint64_t from, const uint64_t to);
  void InitAggBufStates();

  int ProcessAggregationNoVerbs();
  int ProcessAggregation();

  // Utility Functions
  void PrintDebug(int compare = 1);

 protected:
  void BackgroundThreadLoop();

  std::thread background_thread_;
  Eigen::ThreadPool* pool_;
  //Eigen::ThreadPoolDevice* d_;
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

  // Data
  std::map<string, int> name_to_index_;
  std::vector<TargetBufPair> target_buf_pairs_;
  std::vector<string> names_;
  std::vector<Flat*> glc_flats_;
  std::vector<Flat*> buf_flats_;
#if 0
  uint64_t* buf_states_;
#else
  std::vector<uint64_t*> buf_states_;
  //std::vector<std::atomic<uint64_t>*> buf_states_;
#endif
  std::vector<struct ibv_mr*> buf_state_mrs_;

  /// RdmaEnv
  RdmaEnv* rdma_env_;
  /// Completion Queue
  /// TODO: const?
  //struct ibv_cq* cq0_;
  struct ibv_cq* cq_ = nullptr;
  /// QP
  /// TODO: const?
  //struct ibv_qp* qp0_;
  struct ibv_qp* qp_ = nullptr;
};

}  // namespace ptre

#endif  // PTRE_CM_TENSOR_AGGREGATOR_H_
