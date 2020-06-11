#include "ptre/common/communication/grpc/grpc_client_cache.h"

#include <vector>
#include <string>
#include <thread>

void RunGrpcServer() {
  RdmaServiceImpl service;
  service.SetRdmaMgr(ptre_global.rdma_mgr);
  service.SetConsensusManager(&ptre_global.cm);
  std::string server_address("0.0.0.0:50051");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  ptre_global.grpc_server = builder.BuildAndStart();
  std::cout << "Grpc server listening on " << server_address << std::endl;
  ptre_global.grpc_server->Wait();
}

int main() {
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);
  std::vector<std::string> hosts;
  hosts.push_back("ib010:50051");
  GrpcClientCache cache(0, hosts);
  return 0;
}
