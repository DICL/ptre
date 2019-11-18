#include "ptre/cm/remote_store.h"

#include <string>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace tensorflow {

static Allocator* get_default_cpu_allocator() {
  static Allocator* default_cpu_allocator =
      cpu_allocator(port::kNUMANoAffinity);
  return default_cpu_allocator;
}

//RemoteStore::RemoteStore(const Tensor& other) {
//  tensor_ = new Tensor(get_default_cpu_allocator(),
//                       other.dtype(),
//                       other.shape());
//}

RemoteStore::RemoteStore() {
}

RemoteStore::~RemoteStore() {
  //delete tensor_;
}

void RemoteStore::AddVariable(const std::string& name, Tensor* var) {
  //TODO: manage memory leak
  //Tensor* var = new Tensor(tensor::DeepCopy(*in));
  //Tensor* var = new Tensor(*in);
  vars_.push_back(var);
  name_to_var_.emplace(name, var);
  LOG(INFO) << "Registered a new tensor: " << name << std::endl
            << var->dtype() << ", "
            << var->shape() << std::endl;
}

//void RemoteStore::AverageVariableCpu(const std::string& name,
//                                     const Tensor* other) {
//  Tensor* var = name_to_var_[name];
//  AverageVariableCpu(var, other);
//}
//
//void RemoteStore::AverageVariableCpu(Tensor* target, const Tensor* other) {
//  for(
//}

Tensor* RemoteStore::tensor(const std::string& name) {
  return name_to_var_[name];
}

string RemoteStore::DebugString(const std::string& name, int max_entries) {
  auto t = name_to_var_[name];
  return t->DebugString(max_entries);
}
//void RemoteStore::Write(const Tensor& other) {
//  CHECK_EQ(tensor_->shape(), other.shape());
//  TensorBuffer* buf = tensor_->buf_;
//  TensorBuffer* other_buf = other.buf_;
//  size_t size = other_buf->size();
//  for (size_t i = 0; i < size; i++) {
//    buf[i] = other_buf[i];
//  }
//}

}  // namespace tensorflow
