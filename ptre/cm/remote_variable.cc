#include "ptre/cm/remote_variable.h"

#include <thread>
#include <chrono>

namespace ptre {

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

RemoteVariable::RemoteVariable(const Tensor& var) {
  /// \brief Creates a Tensor of the given `type` and `shape`.  If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  tensor_ = new Tensor(var.dtype(), var.shape());
  // Receive Buffer
  /*
  rcv_tensor_ = new Tensor(var.dtype(), var.shape());
  rcv_length_ = rcv_tensor_->AllocatedBytes();
  rcv_buf_ = (void*) rcv_tensor_->tensor_data().data();
  */
  rcv_length_ = tensor_->AllocatedBytes();
  //rcv_buf_ = malloc(rcv_length_);
  size_t alloc_size = (rcv_length_ + 63) / 64;
  alloc_size *= 64;
  rcv_buf_ = aligned_alloc(64, alloc_size);
  memset(rcv_buf_, 0, alloc_size);

  rcv_state_ = 0;
  agg_state_ = 0;
  agg_cnt_ = 0;

  permit_ = new Permit();
}

RemoteVariable::RemoteVariable(const Tensor& var, Allocator* a) {
  tensor_ = new Tensor(var.dtype(), var.shape());
  // Receive Buffer
  rcv_length_ = tensor_->AllocatedBytes();
  rcv_buf_ = a->Allocate(rcv_length_);
  //memset(rcv_buf_, 0, rcv_length_);

  rcv_state_ = 0;
  agg_state_ = 0;
  agg_cnt_ = 0;

  permit_ = new Permit(a);
}

void RemoteVariable::StartRecv() {
  std::lock_guard<std::mutex> guard(mu_);
  //tensor_->flat<float>().setZero();
  agg_cnt_ = 0;
  agg_state_ = 0;
  rcv_state_ = 1;
  permit_->SwapPendingQueue();
  permit_->Next();
}

int RemoteVariable::EnqueueSenderCandidate(int src_rank) {
  std::lock_guard<std::mutex> guard(mu_);
  int ret = permit_->Enqueue(src_rank, rcv_state_);
  if (!ret && rcv_state_ == 1 && permit_->value() == -1) {
    permit_->Next();
  }
  return ret;
}

void RemoteVariable::StopRecv() {
  std::lock_guard<std::mutex> guard(mu_);
  rcv_state_ = 0;
  permit_->SetValue(-1);
}

void RemoteVariable::NewIncoming(int src_rank) {
  std::lock_guard<std::mutex> guard(mu_);
  if (rcv_state_ == 1 && permit_->value() == src_rank) {
    permit_->SetValue(-1);
    agg_state_ = 1;
  }
}

void RemoteVariable::SetAggState(int state) {
  std::lock_guard<std::mutex> guard(mu_);
  agg_state_ = state;
}

void RemoteVariable::Aggregate() {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_ == 1 && rcv_state_ == 1) {
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Flat var_flat = tensor_->flat<float>();
    Flat rcv_flat((float*) rcv_buf_, var_flat.size());
    if (agg_cnt_ == 0) {
      //var_flat = rcv_flat;
      memcpy((void*) tensor_->tensor_data().data(), rcv_buf_, rcv_length_);
    } else {
      var_flat = var_flat + rcv_flat;
    }
    //AggregateSum(d, *glc_flats_[idx], *agg_flats_[idx]);
    agg_cnt_++;
    permit_->Next();
  }
  agg_state_ = 0;
}

void RemoteVariable::AggregateEigenDevice(const Eigen::ThreadPoolDevice& d) {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_ == 1 && rcv_state_ == 1) {
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Flat var_flat = tensor_->flat<float>();
    Flat rcv_flat((float*) rcv_buf_, var_flat.size());
    if (agg_cnt_ == 0) {
      //var_flat = rcv_flat;
      //memcpy((void*) tensor_->tensor_data().data(), rcv_buf_, rcv_length_);
      var_flat.device(d) = rcv_flat;
    } else {
      var_flat.device(d) = var_flat + rcv_flat;
    }
    agg_cnt_++;
    permit_->Next();
  }
  agg_state_ = 0;
}

int RemoteVariable::AggCount() {
  std::lock_guard<std::mutex> guard(mu_);
  return agg_cnt_;
}


int RemoteVariable::GetGlcTensor(Tensor*& out) {
  std::lock_guard<std::mutex> guard(mu_);
  rcv_state_ = 0;
  agg_state_ = 0;
  permit_->SetValue(-1);
  out = tensor_;
  return agg_cnt_;
}

void* RemoteVariable::rcv_data() {
  return rcv_buf_;
}

size_t RemoteVariable::rcv_length() {
  return rcv_length_;
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

int RemoteVariable::permit() {
  return permit_->value();
}

void* RemoteVariable::permit_data() {
  return permit_->data();
}

}  // namespace ptre
