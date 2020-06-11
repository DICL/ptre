#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include "ptre/tests/test_server_lib.h"
#include "ptre/common/cm/consensus_manager.h"
#include "ptre/common/cm/tensor_aggregator.h"
#include "ptre/common/communication/rdma/rdma_mgr.h"
#include "ptre/common/communication/rdma/grpc_server.h"
#include "ptre/common/communication/rdma/grpc_client.h"
#include "ptre/common/communication/grpc/grpc_client_cache.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

#define NUM_TENSORS 3

using std::string;

using ptre::ConsensusManager;
using ptre::RdmaMgr;
using ptre::GrpcClient;
using ptre::GrpcClientCache;
using ptre::BufType;
using ptre::RemoteMR;
using ptre::Flat;

extern std::shared_ptr<ptre::GrpcClientCache> kGrpcClientCache;

using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::DT_FLOAT;

extern int kSize;
extern int kRank;
extern string kHostFile;
extern std::vector<string> kGrpcHosts;

std::vector<string> kTensorNames;
std::vector<Tensor*> kTensors;
ConsensusManager* kCm;
RdmaMgr* kRdmaMgr;

std::thread kAggThread;
std::thread kWriterThread;
std::thread kTrainingThread;

int ProcessAggregation() {
  return kCm->ProcessAggregation();
}

void AggregationThreadLoop() {
  ptre::TensorAggregator* agg = kCm->tensor_aggregator();
  auto last_time = std::chrono::system_clock::now();
  while (true) {
    auto curr_time = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    int ret = ProcessAggregation();
    if (ret > 0) {
      std::cout << "[DEBUG] agg_cnt=" << ret << std::endl;
      //LOG(INFO) << "[DEBUG] agg_cnt=" << ret;
      last_time = curr_time;
    }
    std::chrono::duration<double> since_last = curr_time - last_time;
    if (since_last.count() > 5) {
      LOG(INFO) << "[DEBUG] Agg not performed for a while.";
      agg->PrintDebug(0);
      last_time = curr_time;
    }
  }
}

void WriterThreadLoop() {
  int dst_rank = 0;
  while (false) {
    GrpcClient* grpc_client;
    kGrpcClientCache->GetClient(dst_rank, &grpc_client);
    bool can_push = grpc_client->AttemptPush(1);
    if (can_push) {
      //std::cout << "Can push to " << dst_rank << std::endl;
      kCm->PushTensorsV3(dst_rank);
      grpc_client->NotifyPushDone();
    }
  }
}

void TrainingThreadLoop() {
  bool first_step = 1;
  while (true) {
    for (int i = 0; i < kTensors.size(); i++) {
      Tensor* t = kTensors[i];
      auto flat = t->flat<float>();
      for (int i = 0; i < flat.size(); i++) {
        flat(i) = flat(i) * 0.999;
      }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int i = 0; i < kTensors.size(); i++) {
      int num_incomings = kCm->WaitAndGetNumIncomings();
      Tensor* t = kTensors[i];
      auto flat = t->flat<float>();
      Tensor other = kCm->global_consensus(kTensorNames[i]);
      auto other_flat = other.flat<float>();
      for (int i = 0; i < flat.size(); i++) {
        flat(i) = (flat(i) + other_flat(i)) / num_incomings;
      }
      kCm->CountReduceAndOpenRecv(kTensorNames[i]);
    }
    if (first_step) {
      kCm->InitNumRecvTensors();
      first_step = 0;
    }
    for (int i = 0; i < kSize; i++) {
      //if (i == kRank) continue;
      int dst_rank = i;
      LOG(INFO) << "target to push = " << i;
      GrpcClient* grpc_client;
      kGrpcClientCache->GetClient(dst_rank, &grpc_client);
      bool can_push = grpc_client->AttemptPush(1);
      if (can_push) {
        kCm->PushTensorsV3(dst_rank);
        grpc_client->NotifyPushDone();
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " host_file size rank\n";
    return 1;
  }
  kHostFile = argv[1];
  kSize = atoi(argv[2]);
  kRank = atoi(argv[3]);

  std::cout << "host_file: " << kHostFile << std::endl;
  std::cout << "size: " << kSize << std::endl;
  std::cout << "rank: " << kRank << std::endl;

  // 2. Init Tensors
  for (int i = 0; i < NUM_TENSORS; i++) {
    Tensor* t = new Tensor(DT_FLOAT, TensorShape({2, 2}));
    auto flat = t->flat<float>();
    for (int i = 0; i < flat.size(); i++) {
      flat(i) = 0.1;
    }
    kTensorNames.emplace_back(std::to_string(i));
    kTensors.push_back(t);
    std::cout << "name: " << kTensorNames[i] << ", " << t->DebugString() << std::endl;
  }


  std::cout << "kCm=" << kCm << std::endl;
  ptre::InitTestPtre(kHostFile, kSize, kRank, kTensorNames, kTensors, kCm,
      kRdmaMgr);
  std::cout << "InitTestPtre done.\n";
  std::cout << "kCm=" << kCm << std::endl;
  Tensor other;
  other = kCm->global_consensus("0");
  std::cout << "name: 0, " << other.DebugString() << std::endl;
  kAggThread = std::thread(AggregationThreadLoop);
  std::cout << "Launching Aggregation Thread done.\n";
  kWriterThread = std::thread(WriterThreadLoop);
  std::cout << "Launching Writer Thread done.\n";
  kTrainingThread = std::thread(TrainingThreadLoop);

  kAggThread.join();
  kWriterThread.join();
  kTrainingThread.join();

  return 0;
}
