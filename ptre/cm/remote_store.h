#ifndef TENSORFLOW_PTRE_CM_REMOTE_STORE_H_
#define TENSORFLOW_PTRE_CM_REMOTE_STORE_H_

#include <vector>
#include <unordered_map>
#include <string>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class Tensor;

class RemoteStore {
  friend class Tensor;
  friend class TensorBuffer;

 public:
  RemoteStore();
  ~RemoteStore();
  void AddVariable(const std::string& name, Tensor* var);
  //void AverageVariableCpu(const std::string& name, const Tensor* other);
  //void AverageVariableCpu(Tensor* target, const Tensor* other);
  Tensor* tensor(const std::string& name);
  //void Write(const Tensor& tensor);
  //void Read(Tensor& tensor);
  string DebugString(const std::string& name, int max_entries);  // For debugging

 private:
  std::vector<Tensor*> vars_;
  std::map<std::string, Tensor*> name_to_var_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PTRE_CM_REMOTE_STORE_H_
