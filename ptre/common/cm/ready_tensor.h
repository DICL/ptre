#ifndef PTRE_COMMON_CM_READY_TENSOR_H_
#define PTRE_COMMON_CM_READY_TENSOR_H_
#include <condition_variable>
#include <mutex>

#include "ptre/common/common.h"

namespace ptre {
namespace common {

class ReadyTensor : public Tensor {
 public:
  ReadyTensor(DataType type, const TensorShape& shape);
  ReadyTensor(const Tensor& other);
  void set_step(uint64_t step);
  uint64_t step() { return step_; }
  std::mutex& mu() { return mu_; }
  std::condition_variable& cv() { return cv_; }

 protected:
  std::mutex mu_;
  std::condition_variable cv_;
  uint64_t step_ = 0;
};

}  // namespace common
}  // namespace ptre
#endif  // PTRE_COMMON_CM_READY_TENSOR_H_
