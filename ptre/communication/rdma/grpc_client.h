#ifndef PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_
#define PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "ptre/protobuf/rdma_service.grpc.pb.h"
#include "ptre/communication/rdma/rdma_manager.h"

namespace ptre {

class GrpcClient {
 public:
  //GrpcClient(std::shared_ptr<::grpc::Channel> channel);
  GrpcClient(int src_rank, int dst_rank);
  int GetRemoteAddress(const std::string& name);
  int GetRemoteEnv();
  void SetRdmaManager(RdmaManager* rdma_manager);

 private:
  int src_rank_;
  int dst_rank_;
  std::unique_ptr<Rdma::Stub> stub_;
  RdmaManager* rdma_manager_ = nullptr;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_
