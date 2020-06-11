#ifndef PTRE_COMMON_COMMUNICATION_PUSH_VARIABLE_H_
#define PTRE_COMMON_COMMUNICATION_PUSH_VARIABLE_H_

#include <mutex>
#include "ptre/core/allocator.h"
#include "tensorflow/core/framework/tensor.h"

namespace ptre {

using ::tensorflow::Tensor;

class PushVariable {
 public:
  PushVariable(const Tensor& var);
  PushVariable(const Tensor& var, Allocator* a);

  void StartPush();
  void StopPush();
  int GetState();
  void* data();
  size_t length();

 private:
  std::mutex mu_;
  // Send Buffer
  //Tensor* tensor_;
  void* buf_;
  size_t length_;
  // State
  int state_;
};

}  // namespace ptre

#endif  // PTRE_COMMON_COMMUNICATION_PUSH_VARIABLE_H_
