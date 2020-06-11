#include <thread>

#include "ptre/common/communication/rdma/grpc_server.h"
#include "ptre/common/communication/rdma/rdma_mgr.h"

int main(int argc, char* argv[]) {
  //RdmaMgr* rdma_mgr = new RdmaMgr(  
  ptre::common::GrpcServer grpc_server;
  std::thread t1(grpc_server.RunServer, nullptr);



  t1.join();

  return 0;
}
