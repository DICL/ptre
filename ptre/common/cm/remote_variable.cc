#include "ptre/common/cm/remote_variable.h"

#include <thread>
#include <chrono>
#include <sstream>

namespace ptre {
namespace common {

#if 0
// Tensor* to Flat
{
  std::vector<Flat> recv_flats;
  for (int i = 0; i < num_vars_; i++) {
    recv_flats.push_back(global_consensus_[i]->flat<float>());
  }
}
#endif

#if 0
// Tensor* to void* and size
{
  Tensor* recv_tensor = new Tensor(vars[i]->dtype(), vars[i]->shape());
  global_consensus_.push_back(recv_tensor);
  // Register Buf
  tensorflow::StringPiece strpc = recv_tensor->tensor_data();
  void* buf = (void*) strpc.data();
  size_t length = strpc.size();
}
#endif

RemoteVariable::RemoteVariable(const Tensor& var, const string& name) {
  /// \brief Creates a Tensor of the given `type` and `shape`.  If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  tensor_ = new Tensor(var.dtype(), var.shape());
  name_ = name;

  agg_state_ = 1;
  agg_cnt_ = 0;
}

RemoteVariable::RemoteVariable(const Tensor& var, const string& name,
                               Allocator* a) {
  tensor_ = new Tensor(var.dtype(), var.shape());
  name_ = name;

  agg_state_ = 1;
  agg_cnt_ = 0;
}

void RemoteVariable::StartAggregation() {
  std::lock_guard<std::mutex> guard(mu_);
  //tensor_->flat<float>().setZero();
  agg_cnt_ = 0;
  agg_state_ = 1;
}

void RemoteVariable::StopAggregation() {
  std::lock_guard<std::mutex> guard(mu_);
  agg_state_ = 0;
}

void RemoteVariable::SetAggState(int state) {
  std::lock_guard<std::mutex> guard(mu_);
  agg_state_ = state;
}

void RemoteVariable::Aggregate(const Tensor& other) {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_) {
    Flat var_flat = tensor_->flat<float>();
    ConstFlat other_flat = other.flat<float>();
    if (agg_cnt_ == 0) {
      var_flat = other_flat;
    } else {
      var_flat = var_flat + other_flat;
    }
    agg_cnt_++;
  }
}

void RemoteVariable::Aggregate(const Tensor& other,
                               const Eigen::ThreadPoolDevice& d) {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_) {
    Flat var_flat = tensor_->flat<float>();
    ConstFlat other_flat = other.flat<float>();
    if (agg_cnt_ == 0) {
      var_flat.device(d) = other_flat;
    } else {
      var_flat.device(d) = var_flat + other_flat;
    }
    agg_cnt_++;
  }
}

void RemoteVariable::Aggregate(const void* other) {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_) {
    Flat var_flat = tensor_->flat<float>();
    ConstFlat other_flat((const float*) other, var_flat.size());
    if (agg_cnt_ == 0) {
      var_flat = other_flat;
    } else {
      var_flat = var_flat + other_flat;
    }
    agg_cnt_++;
  }
}

void RemoteVariable::Aggregate(const void* other,
                               const Eigen::ThreadPoolDevice& d) {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_) {
    Flat var_flat = tensor_->flat<float>();
    ConstFlat other_flat((const float*) other, var_flat.size());
    if (agg_cnt_ == 0) {
      var_flat.device(d) = other_flat;
    } else {
      var_flat.device(d) = var_flat + other_flat;
    }
    agg_cnt_++;
  }
}

void RemoteVariable::Reduce() {
  std::lock_guard<std::mutex> guard(mu_);
//std::stringstream ss;  // DEBUG
//ss << "\nREDUCE BEFORE=" << tensor_->DebugString() << "\n";  // DEBUG
  Flat var_flat = tensor_->flat<float>();
  float m_float = agg_cnt_;
  ConstScalar m(&m_float, 1);
  var_flat = var_flat / m();
//ss << "\nREDUCE AFTER=" << tensor_->DebugString() << "\n";  // DEBUG
//LOG(INFO) << ss.str();  // DEBUG
}

void RemoteVariable::Reduce(const Eigen::ThreadPoolDevice& d) {
  std::lock_guard<std::mutex> guard(mu_);
  Flat var_flat = tensor_->flat<float>();
  float m_float = agg_cnt_;
  ConstScalar m(&m_float, 1);
  var_flat.device(d) = var_flat / m();
}

int RemoteVariable::AggCount() {
  std::lock_guard<std::mutex> guard(mu_);
  return agg_cnt_;
}

int RemoteVariable::agg_state() {
  return agg_state_;
}

int RemoteVariable::agg_count() {
  return agg_cnt_;
}

Tensor* RemoteVariable::tensor() {
  return tensor_;
}

const string& RemoteVariable::name() {
  return name_;
}

}  // namespace common
}  // namespace ptre
