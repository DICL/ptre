#include <thread>

#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/rdma_manager.h"

int main(int argc, char* argv[]) {
  //RdmaManager* rdma_manager = new RdmaManager(  
  ptre::GrpcServer grpc_server;
  std::thread t1(grpc_server.RunServer, nullptr);



  t1.join();

  return 0;
}
