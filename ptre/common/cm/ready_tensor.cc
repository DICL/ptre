#include "ptre/common/cm/ready_tensor.h"

#include <condition_variable>
#include <mutex>

#include "ptre/common/common.h"

namespace ptre {
namespace common {

ReadyTensor::ReadyTensor(DataType type, const TensorShape& shape)
    : Tensor(type, shape) { }

ReadyTensor::ReadyTensor(const Tensor& other) : Tensor(other) { }

void ReadyTensor::set_step(uint64_t step) {
  step_ = step;
}

}  // namespace common
}  // namespace ptre
