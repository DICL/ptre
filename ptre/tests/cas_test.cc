#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "ptre/communication/rdma/rdma_manager.h"
#include "ptre/communication/grpc/grpc_client_cache.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/grpc_client.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

#define COMM_SIZE 10

using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::DT_FLOAT;

#define WKLOG(MSG) std::cout << "[RANK:" << ptre::rank << "] " << MSG << std::endl;

namespace ptre {


ConsensusManager cm;
RdmaManager* rdma_manager;
std::unique_ptr<grpc::Server> grpc_server = nullptr;
std::thread grpc_server_thread;
std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;
int rank;
int size;
std::vector<std::string> grpc_hosts;
std::vector<std::string> tensor_names = {"my_tensor_0"};

void RunGrpcServer() {
  RdmaServiceImpl service;
  service.SetRdmaManager(rdma_manager);
  service.SetConsensusManager(&cm);
  //service.SetBarrierVariable(barrier_variable);
  std::string server_address("0.0.0.0:50051");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  grpc_server = builder.BuildAndStart();
  //std::cout << "Grpc server listening on " << server_address << std::endl;
  grpc_server->Wait();
}

void InitGrpcService() {
  //std::cout << "Running grpc server" << std::endl;
  grpc_server_thread = std::thread(RunGrpcServer);
}

void init(int argc, char* argv[], Tensor tensor) {
  /// 0: rank
  size = atoi(argv[1]);
  rank = atoi(argv[2]);
  cm.set_size(size);
  cm.set_rank(rank);
  std::string in_line;
  std::ifstream in("/home/wkim/experiments/scripts/host_files/grpc_hosts_recent_ipoib");
  while (std::getline(in, in_line)) {
    if (in_line[0] == '#') continue;
    grpc_hosts.emplace_back(in_line);
  }
  in.close();
  //for (int i = 0; i < size; i++) {
  //  std::cout << grpc_hosts[i] << std::endl;
  //}
  //for (int i = 0; i < size; i++) {
  //  std::stringstream ss;
  //  ss << "ib" << std::setw(3) << std::setfill('0') << (i + 1) << ":50051";
  //  grpc_hosts.push_back(ss.str());
  //  ss.str("");
  //  //dd= {"ib001:50051", "172.30.1.2:50051", "ib003:50051"};
  //  //cout << grpc_hosts[i];
  //}
  grpc_client_cache = std::make_shared<GrpcClientCache>(rank, grpc_hosts);

  //std::cout << "Initializing RdmaManager" << std::endl;
  rdma_manager = new RdmaManager(size, rank, false);
  cm.SetRdmaManager(rdma_manager);
  //is_shutdown = false;
  //background_thread = std::thread(BackgroundThreadLoop);
  cm.InitBufTensor("my_tensor_0", tensor);
  cm.InitBufParam();

  InitGrpcService();
  /// Init remote MR
  bool peer_flag[size] = {};
  peer_flag[rank] = true;
  int done_flag = 0;
  while (!done_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    done_flag = 1;
    for (int i = 0; i < size; i++) {
      if (peer_flag[i]) {
        continue;
      }
      GrpcClient* grpc_client;
      grpc_client_cache->GetClient(i, &grpc_client);
      grpc_client->SetRdmaManager(rdma_manager);
      if (!rdma_manager->IsDlidSet(i)) {
        int ret = grpc_client->GetRemoteEnv();
        if (ret < 0) {
          done_flag = 0;
          continue;
        }
      }
      int client_status = 0;
      for (int j = 0; j < tensor_names.size(); j++) {
        if (rdma_manager->IsRemoteMRSet(i, tensor_names[j])) {
          continue;
        }
        int ret = grpc_client->GetRemoteAddress(tensor_names[j]);
        if (ret < 0) {
          client_status = -1;
          break;
        }
      }
      if (client_status < 0) {
        done_flag = 0;
        continue;
      }
      if (!rdma_manager->IsRemoteParamMRSet(i)) {
        int ret = grpc_client->GetRemoteParamAddress();
        if (ret < 0) {
          done_flag = 0;
          continue;
        }
      }
      peer_flag[i] = true;
    }
  }
  //std::cout << "Init RemoteMR done." << std::endl;
  /// Connect QPs
  done_flag = 0;
  while (!done_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    done_flag = 1;
    for (int i = 0; i < size; i++) {
      if (i == rank) {
        continue;
      }
      int r = rdma_manager->ConnectQP(i);
      if (r < 0) {
        done_flag = 0;
      }
    }
  }
  //WKLOG("Init done.");
}

}

int main(int argc, char* argv[]) {
  Tensor a(DT_FLOAT, TensorShape({7, 7, 5}));
  int rank = atoi(argv[2]);
  auto flat = a.flat<float>();
  for (int i = 0; i < flat.size(); i++) {
    //flat(i) = pow(-1, i) * (i + 1) / 10;
    //flat(i) = (rank + 1) / 10.0f;
    flat(i) = 0.1;
  }
  ptre::init(argc, argv, a);
  if (ptre::rank != 0) {
    ptre::rdma_manager->PushTensorAtomicAddBatch(0, ptre::tensor_names[0], a);
  } else {
    auto glc = ptre::cm.global_consensus(0);
    auto glc_flat = glc.flat<float>();
    std::string prev;
    while (true) {
      std::stringstream ss("");
      if (glc_flat.size() > 0) {
        ss << ", flat=[";
      }
      for (int i = 0; i < glc_flat.size(); i++) {
        ss << glc_flat(i);
        if (i < glc_flat.size() - 1) {
          ss << ", ";
        } else {
          ss << "]";
        }
      }
      ss << " " << glc_flat(glc_flat.size());
      std::string curr = glc.DebugString() + ss.str();
      if (curr.compare(prev) != 0) {
        WKLOG(curr);
      }
      prev = curr;
    }
  }
  if (ptre::grpc_server_thread.joinable()) {
    ptre::grpc_server_thread.join();
  }
  return 0;
}
