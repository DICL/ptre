#ifndef PTRE_COMMON_COMMON_H_
#define PTRE_COMMON_COMMON_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace ptre {
namespace common {

using OpContext = tensorflow::OpKernelContext;
using Status = tensorflow::Status;
using StatusCallback = std::function<void(const Status&)>;
using Tensor = tensorflow::Tensor;

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_COMMON_H_
