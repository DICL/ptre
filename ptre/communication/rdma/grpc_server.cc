#include "ptre/communication/rdma/grpc_server.h"

#include <string>

namespace ptre {

class RdmaServiceImpl final : public Rdma::Service {
  grpc::Status GetRemoteAddress(grpc::ServerContext* context,
                                const GetRemoteAddressRequest* request,
                                GetRemoteAddressResponse* response) override {
    response->set_host_name("hostname");
    return grpc::Status::OK;
  }
};

void GrpcServer::RunServer() {
  std::string server_address("0.0.0.0:50051");
  RdmaServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  server_ = std::move(std::unique_ptr<grpc::Server>(builder.BuildAndStart()));
  server_->Wait();
}

}  // namespace ptre
