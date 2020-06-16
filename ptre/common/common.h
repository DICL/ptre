#ifndef PTRE_COMMON_COMMON_H_
#define PTRE_COMMON_COMMON_H_

#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

namespace ptre {
namespace common {

using OpContext = tensorflow::OpKernelContext;
using Status = tensorflow::Status;
using StatusCallback = std::function<void(const Status&)>;
using Tensor = tensorflow::Tensor;

enum ReduceOp {
  REDUCE_SUM = 0
};

struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
 // std::shared_ptr<OpContext> context;
  OpContext* context;
  // Input tensor.
  //std::shared_ptr<Tensor> tensor;
  Tensor* tensor;
  // Pre-allocated output tensor.
  //std::shared_ptr<Tensor> output;
  Tensor* output;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Event indicating that data is ready.
  //std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_COMMON_H_
