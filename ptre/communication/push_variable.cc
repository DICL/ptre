#include "ptre/communication/push_variable.h"

namespace ptre {

PushVariable::PushVariable(size_t length) {
  buf_ = malloc(length);
  length_ = length;
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
