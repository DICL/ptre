#ifndef PTRE_CM_MODEL_H_
#define PTRE_CM_MODEL_H_

#include <map>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace ptre {

class Model {
 public:
  Model(const std::vector<const Tensor&>& tensors);

 private:
  void* blob_;
  std::map<std::string, uint64_t> bufs_;
  std::map<std::string, const Tensor&> tensors_;
};


}  // namespace ptre

#endif  // PTRE_CM_MODEL_H_
