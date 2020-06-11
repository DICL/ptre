#ifndef PTRE_COMMON_CM_REMOTE_VARIABLE_H_
#define PTRE_COMMON_CM_REMOTE_VARIABLE_H_

#define EIGEN_USE_THREADS

#include <mutex>

#include "ptre/core/allocator.h"
#include "ptre/tensorflow/types.h"
#include "tensorflow/core/framework/tensor.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace ptre {
namespace common {

using std::string;
using ::tensorflow::Tensor;
using ::tensorflow::DataType;
using ::tensorflow::TensorShape;

class RemoteVariable {
 public:
  RemoteVariable(const Tensor& var, const string& name);
  RemoteVariable(const Tensor& var, const string& name, Allocator* a);
  void StartAggregation();
  void StopAggregation();
  void SetAggState(int state);
  void Aggregate(const Tensor& other);
  void Aggregate(const Tensor& other, const Eigen::ThreadPoolDevice& d);
  void Aggregate(const void* other);
  void Aggregate(const void* other, const Eigen::ThreadPoolDevice& d);
  int AggCount();
  void Reduce();
  void Reduce(const Eigen::ThreadPoolDevice& d);
  int agg_state();
  int agg_count();
  Tensor* tensor();
  const string& name();
  uint64_t last_key() { return last_key_; }
  void set_last_key(uint64_t key) { last_key_ = key; }

  DataType dtype() const { return tensor_->dtype(); }
  const TensorShape& shape() const { return tensor_->shape(); }

 private:
  std::mutex mu_;
  string name_;
  // Storage
  Tensor* tensor_;

  // State Member Variables
  int agg_state_;
  int agg_cnt_;

  uint64_t last_key_;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_CM_REMOTE_VARIABLE_H_
