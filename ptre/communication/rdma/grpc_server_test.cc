#include <thread>

#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/rdma_mgr.h"

int main(int argc, char* argv[]) {
  //RdmaMgr* rdma_mgr = new RdmaMgr(  
  ptre::GrpcServer grpc_server;
  std::thread t1(grpc_server.RunServer, nullptr);



  t1.join();

  return 0;
}
