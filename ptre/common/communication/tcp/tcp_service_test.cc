#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ptre/common/cm/consensus_manager.h"
#include "ptre/common/communication/rdma/grpc_server.h"
#include "ptre/common/communication/tcp/tcp_service_impl.h"
#include "ptre/common/communication/tcp/tcp_grpc_client.h"

using namespace std;
using namespace ptre::common;

const string kServerAddrs[2] = { "0.0.0.0:50051", "0.0.0.0:50052" };

void RunGrpcServer(int comm_size, int comm_rank) {
  // Init ConsensusManager
  std::vector<string> names = { "tensor_0" };
  std::vector<const Tensor*> inputs;
  inputs.push_back(new Tensor(tensorflow::DT_FLOAT, TensorShape({2, 2})));
  ConsensusManager cm(comm_size, comm_rank, inputs, names);

  // Init Ready Tensor
  Tensor* tensor = cm.ready_tensor(names[0]);
  for (int i = 0; i < tensor->NumElements(); i++) {
    tensor->flat<float>()(i) = comm_rank + 1;
  }
  printf("[RANK=%d]: Init ready tensor(name=%s): %s\n", comm_rank,
      names[0].c_str(), tensor->DebugString(4).c_str());

  // Init Grpc Services
  grpc::ServerBuilder builder;
  builder.AddListeningPort(kServerAddrs[comm_rank],
      grpc::InsecureServerCredentials());
  builder.SetMaxReceiveMessageSize(INT_MAX);

  // For multiple service registrations test
  RdmaServiceImpl rdma_service;
  builder.RegisterService(&rdma_service);

  // Register TCP Service
  TcpServiceImpl service;
  service.SetConsensusManager(&cm);
  builder.RegisterService(&service);

  // Run Grpc Server
  auto grpc_server = builder.BuildAndStart();
  printf("[RANK=%d]: Started Grpc Server: %s\n", comm_rank,
      kServerAddrs[comm_rank].c_str());
  grpc_server->Wait();
}

int main(int argc, char* argv[]) {
  std::thread t0(RunGrpcServer, 2, 0);
  std::thread t1(RunGrpcServer, 2, 1);

  std::this_thread::sleep_for(std::chrono::seconds(1));
  Tensor a(tensorflow::DT_FLOAT, TensorShape({2, 2}));
  printf("Before Pull: %s\n", a.DebugString(4).c_str());
  for (int i = 0; i < 2; i++) {
    TcpGrpcClient client(0, i, kServerAddrs[i]);
    client.PullTensor("tensor_0", 0, a);
    printf("After Pull from rank %d: %s\n", i, a.DebugString(4).c_str());
  }

  printf("Ctrl-C to terminate\n");

  t0.join();
  t1.join();
  return 0;
}
