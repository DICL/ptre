#include "ptre/cm/remote_variable.h"

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

void RemoteVariable::StartRecv() {
  std::lock_guard<std::mutex> guard(mu_);
  tensor_->flat<float>().setZero();
  agg_cnt_ = 0;
  agg_state_ = 0;
  rcv_state_ = 1;
  permit_->SwapPendingQueue();
  permit_->Next();
}

void RemoteVariable::EnqueueSenderCandidate(int src_rank) {
  std::lock_guard<std::mutex> guard(mu_);
  permit_->Enqueue(src_rank, rcv_state_);
  if (rcv_state_ == 1 && permit_->value() == -1) {
    permit_->Next();
  }
}

void RemoteVariable::StopRecv() {
  std::lock_guard<std::mutex> guard(mu_);
  rcv_state_ = 0;
  permit_->SetValue(-1);
}

void RemoteVariable::SetAggState(int state) {
  std::lock_guard<std::mutex> guard(mu_);
  agg_state_ = state;
}

void RemoteVariable::Aggregate() {
  std::lock_guard<std::mutex> guard(mu_);
  if (agg_state_ == 1 && rcv_state_ == 1) {
    Flat var_flat = tensor_->flat<float>();
    Flat rcv_flat(rcv_buf_, var_flat.size());
    var_flat = var_flat + rcv_flat;
    //AggregateSum(d, *glc_flats_[idx], *agg_flats_[idx]);
    agg_cnt_++;
    permit_->Next();
  }
  agg_state_ = 0;
}

int RemoteVariable::GetGlcTensor(Tensor*& out) {
  std::lock_guard<std::mutex> guard(mu_);
  rcv_state_ = 0;
  agg_state_ = 0;
  permit_->SetValue(-1);
  out = tensor_;
  return agg_cnt_;
}

}  // namespace ptre
