#ifndef PTRE_COMMUNICATION_PULL_VARIABLE_H_
#define PTRE_COMMUNICATION_PULL_VARIABLE_H_

#include <string>

#include "ptre/communication/comm_types.h"
#include "ptre/core/allocator.h"

#include "tensorflow/core/framework/tensor.h"

namespace ptre {

using ::tensorflow::Tensor;
using ::tensorflow::DataType;
using ::tensorflow::TensorShape;

using std::string;

class PullVariable {
 public:
  PullVariable(const Tensor& var, const string& name, Allocator* a);
  void Switch();
  void SetNextKey(uint64_t key);
  void* data(int idx) { return data_ptrs_[idx]; }
  const void* curr_data() { return data(key_->curr); }
  void* next_data() { return data(!key_->curr); }
  size_t length() { return length_; }
  struct PullKey* key() { return key_; }
  const string& name() { return name_; }

 private:
  string name_;
  struct PullKey* key_;
  void* data_ptrs_[2];
  size_t length_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_PULL_VARIABLE_H_
