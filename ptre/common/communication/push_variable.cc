#include "ptre/common/communication/push_variable.h"

namespace ptre {

PushVariable::PushVariable(const Tensor& var) {
  //tensor_ = new Tensor(var.dtype(), var.shape());
  length_ = var.AllocatedBytes();
  buf_ = malloc(length_);
  state_ = 0;
}

PushVariable::PushVariable(const Tensor& var, Allocator* a) {
  length_ = var.AllocatedBytes();
  buf_ = a->Allocate(length_);
  state_ = 0;
}

void PushVariable::StartPush() {
  std::lock_guard<std::mutex> guard(mu_);
  state_ = 1;
}

void PushVariable::StopPush() {
  std::lock_guard<std::mutex> guard(mu_);
  state_ = 0;
}

void* PushVariable::data() {
  return buf_;
}

size_t PushVariable::length() {
  return length_;
}

int PushVariable::GetState() {
  std::lock_guard<std::mutex> guard(mu_);
  return state_;
}

}  // namespace ptre
