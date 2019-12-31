#ifndef PTRE_PTRE_GLOBAL_H_
#define PTRE_PTRE_GLOBAL_H_

#include <queue>
#include <thread>
#include <string>

#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/rdma_manager.h"

using ptre::ConsensusManager;
using ptre::RdmaManager;

struct PtreGlobal {
  ConsensusManager cm;
  RdmaManager* rdma_manager;
  //std::vector<Tensor*> remote_tensors;
  std::mutex mu;
  //std::queue<PtreRequest> request_queue;
  std::queue<int> q;

  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::thread background_thread;

  //bool new_incoming;

  int rank;
  int size;

  std::vector<std::string> grpc_hosts;

  ~PtreGlobal() {
    if (background_thread.joinable()) {
      //shut_down = true;
      background_thread.join();
    }
    if (grpc_server_thread.joinable()) {
      grpc_server_thread.join();
    }
  }
};

//extern "C" {
//
//int ptre_init(int size, int rank);
//
//}

#endif  // PTRE_PTRE_GLOBAL_H_
