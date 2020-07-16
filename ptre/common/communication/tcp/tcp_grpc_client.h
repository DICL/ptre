#ifndef PTRE_COMMON_COMMUNICATION_TCP_TCP_GRPC_CLIENT_H_
#define PTRE_COMMON_COMMUNICATION_TCP_TCP_GRPC_CLIENT_H_

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "ptre/protobuf/tcp_service.grpc.pb.h"  // TODO: <--Implement

#include "tensorflow/core/framework/tensor.h"

namespace ptre {
using std::string;
namespace {
using tensorflow::Tensor;
}  // namespace

class TcpGrpcClient {
 public:
  TcpGrpcClient(int src_rank, int dst_rank, const string& hostname);
  int PullTensor(const string& tensor_name);
  int PushTensor(const string& tensor_name, const Tensor& tensor);

 private:
  int src_rank_;
  int dst_rank_;
  string hostname_;
  std::unique_ptr<Tcp::Stub> stub_;
};

}  // namespace ptre


#endif  // PTRE_COMMON_COMMUNICATION_TCP_TCP_GRPC_CLIENT_H_
