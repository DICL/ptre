#ifndef PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_
#define PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_

#include <memory>

#include <grpcpp/grpcpp.h>

#include "ptre/protobuf/rdma_service.grpc.pb.h"

namespace ptre {

class GrpcClient {
 public:
  GrpcClient(std::shared_ptr<::grpc::Channel> channel)
      : stub_(Rdma::NewStub(channel)) {}
  void GetRemoteAddress
}

#endif  // PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_
