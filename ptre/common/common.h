#ifndef PTRE_COMMON_COMMON_H_
#define PTRE_COMMON_COMMON_H_

#include <memory>
#include <string>

#include "ptre/common/logging.h"
#include "ptre/common/communication/rdma/rdma.h"
#include "ptre/common/communication/rdma/rdma_channel.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#define DVLOGR(X, R) DVLOG(X) << "[" << R << "] "

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

#define RET_OK(X)                           \
  {                                         \
    int r = X;                              \
    if (r) {                                \
      LOG(ERROR) << #X << " returned " << r \
          << " @ " << __PRETTY_FUNCTION__;  \
      return 1;                             \
    }                                       \
  }

namespace ptre {
namespace common {

using std::string;

using OpContext = ::tensorflow::OpKernelContext;
using Status = ::tensorflow::Status;
using StatusCallback = std::function<void(const Status&)>;
using Tensor = ::tensorflow::Tensor;
using ::tensorflow::DataType;
using ::tensorflow::TensorShape;

enum ReduceOp {
  REDUCE_SUM = 0
};

enum ModelaverageOp {
  MODELAVERAGE_DEFAULT = 0
};

enum CommOp {
  COMM_ALLREDUCE = 0,
  COMM_P2P_PULL = 1
};

enum MemcpyOp {
  MEMCPY_DEVICE_TO_HOST = 0,
  MEMCPY_HOST_TO_DEVICE = 1
};

struct MemcpyRequest {
  OpContext* context;
  string key;
  std::shared_ptr<Tensor> tensor;
  MemcpyOp type;
  StatusCallback callback;
};

struct RvarRequest {
  OpContext* context;
  string var_name;
  std::shared_ptr<Tensor> tensor;
  StatusCallback callback;
};

enum SendbufState {
  SENDBUF_STATE_INIT,
  SENDBUF_STATE_READY
};

enum RecvbufState {
  RECVBUF_STATE_INIT,
  RECVBUF_STATE_READY,
  RECVBUF_STATE_RECV_DONE,
  RECVBUF_STATE_MEMCPY_READY
};

struct StateMutex {
  uint64_t dummy1;
  std::mutex mu;
  uint64_t dummy2;
  int state = 0;
};

enum RdmaOpState {
  RDMA_OP_STATE_WRITE_TENSOR,
  RDMA_OP_STATE_WRITE_STATE
};

struct RdmaEntry {
  string tensor_name;
  std::shared_ptr<Tensor> tensor;
  std::shared_ptr<StateMutex> tensor_state;
  struct ibv_mr* state_mr;
  struct ibv_mr* tensor_mr;
  int rank;
  uint32_t tensor_id;
  RemoteAddr state_addr;
  RemoteAddr tensor_addr;
  RdmaChannel* channel;
  enum RdmaOpState state;
};

struct RdmaRecvEntry {
  int rank;
  RdmaChannel* channel;
};

struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
 // std::shared_ptr<OpContext> context;
  OpContext* context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  //Tensor tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  //Tensor output;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Event indicating that data is ready.
  //std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  //int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
};

struct PtreNode {
  string hostname;
  int local_size;
  std::vector<int> grpc_ports;
};

struct PtreWorker {
  int rank;
  int local_rank;
  string grpc_host;
  int port;
  PtreNode host;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_COMMON_H_
