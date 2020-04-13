//#include "ptre/core/ptre_global.h"

#define EIGEN_USE_THREADS

#include <unistd.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <stdio.h>
#include <typeinfo>
#include <atomic>
#include <chrono>

#include "ptre/kernels/ptre_ops.h"
//#include "ptre/kernels/ptre_op_helpers.h"
//#include "ptre/ptre_global.h"
#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/rdma/rdma_manager.h"
//#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/lib/core/refcount.h"
#include "ptre/communication/grpc/grpc_client_cache.h"

#define LOGSTEP LOG(INFO) << "[DEBUG,step=" << ptre_global.local_step << "]: "

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

using std::string;
using ::ptre::ConsensusManager;
using ::ptre::RdmaManager;
using ::ptre::RdmaServiceImpl;
using ::ptre::GrpcClient;
using ::ptre::GrpcClientCache;
using ::ptre::BufType;
using ::ptre::RemoteMR;
using ::ptre::Flat;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

struct PtreGlobal {
  PtreGlobal() {
    std::cout << (void*) this << std::endl;
    ma_op_cnt = 0;
    ma_op_cnt2 = 0;
    reduce_op_cnt0 = 0;
    reduce_op_cnt1 = 0;
    for (int i = 0; i < num_copy_cnt; i++) {
      copy_cnt[i] = 0;
    }

    /// Init barrier variables
    barrier_counter = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
    barrier_release = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
    memset(barrier_counter, 0, 8);
    memset(barrier_release, 0, 8);
  }
  ConsensusManager cm;
  RdmaManager* rdma_manager = nullptr;
  //std::vector<Tensor*> remote_tensors;
  std::mutex mu;
  //std::queue<PtreRequest> request_queue;
  std::mutex q_mu;
  std::queue<int> q;

  // Grpc Server
  std::unique_ptr<grpc::Server> grpc_server = nullptr;

  std::atomic<bool> is_shutdown;

  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::thread background_thread;
  std::thread aggregation_thread;

  //bool new_incoming;

  int rank;
  int size;

  std::vector<std::string> grpc_hosts;
  std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;


  // Training Infos
  int local_step = 0;
  int virtual_step = 1;

  /// 0: NOT PUSH
  /// 1: PUSH
  /// 2: SKIP
  int push_step_state = 0;
  /// 0: None
  /// 1: New
  /// 2: Used
  int incoming_state = 0;
  int incoming_peer;

  bool barrier_variable = false;
  bool is_broadcast_done = true;

  int num_push = 1;
  bool ever_pushed = false;
  std::atomic<int> ma_op_cnt;
  std::atomic<int> ma_op_cnt2;
  std::atomic<int> reduce_op_cnt0;
  std::atomic<int> reduce_op_cnt1;
  int num_copy_cnt = 2;
  std::atomic<int> copy_cnt[2];

  std::mutex push_mu;

  /// Barrier variables
  uint64_t* barrier_counter;
  uint64_t* barrier_release;

  ~PtreGlobal() {
    if (background_thread.joinable()) {
      //shut_down = true;
      background_thread.join();
    }
    if (grpc_server_thread.joinable()) {
      grpc_server_thread.join();
    }
    if (background_thread.joinable()) {
      aggregation_thread.join();
    }
    if (rdma_manager != nullptr) {
      delete rdma_manager;
    }
  }
};

static PtreGlobal ptre_global;

void RunGrpcServer() {
  RdmaServiceImpl service;
  service.SetRdmaManager(ptre_global.rdma_manager);
  service.SetConsensusManager(&ptre_global.cm);
  service.SetBarrierVariable(&ptre_global.barrier_variable);
  std::string server_address("0.0.0.0:50051");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  ptre_global.grpc_server = builder.BuildAndStart();
  std::cout << "Grpc server listening on " << server_address << std::endl;
  ptre_global.grpc_server->Wait();
}

void ShutdownGrpcServer() {
  if (ptre_global.grpc_server != nullptr) {
    ptre_global.grpc_server->Shutdown();
  }
}

void InitRemoteMR() {
}

void ProcessRequestQueueInternal1() {
  auto& q = ptre_global.q;
  auto& mu = ptre_global.q_mu;
  int dst_rank = -1;
  mu.lock();
  if (!ptre_global.q.empty()) {
    dst_rank = q.front();
    q.pop();
  }
  mu.unlock();
  if (dst_rank >= 0) {
    ptre_global.cm.PushTensors(dst_rank);
    GrpcClient* grpc_client;
    ptre_global.grpc_client_cache->GetClient(dst_rank, &grpc_client);
    grpc_client->AckPushDone();
  }
}

void ProcessRequestQueueInternalAtomicAdd() {
  auto& q = ptre_global.q;
  auto& mu = ptre_global.q_mu;
  int dst_rank = -1;
  mu.lock();
  if (!ptre_global.q.empty()) {
    dst_rank = q.front();
    q.pop();
  }
  mu.unlock();
  if (dst_rank >= 0) {
    // TODO: 1. Attemp
    GrpcClient* grpc_client;
    ptre_global.grpc_client_cache->GetClient(dst_rank, &grpc_client);
    //bool can_push = grpc_client->AttemptPush();
    bool can_push = true;
    // 2. Push
    if (can_push) {
      //std::cout << "\n[RANK=" << ptre_global.rank << "]: PushTensors2()\n";
      ptre_global.cm.PushTensorsV3(dst_rank);
    // TODO: 3. Ack Push done
      //std::cout << "\n[RANK=" << ptre_global.rank << "]: AckPushDone()\n";
      grpc_client->AckPushDone();
    }
  }
}

void ProcessRequestQueueInternalBufferedAggregation() {
  auto& q = ptre_global.q;
  auto& mu = ptre_global.q_mu;
  int dst_rank = -1;
  mu.lock();
  if (!ptre_global.q.empty()) {
    dst_rank = q.front();
    q.pop();
  }
  mu.unlock();
  if (dst_rank >= 0) {
    // TODO: Implement This PushTensorsBufferedAggregation
    ptre_global.cm.PushTensorsV3(dst_rank);
    GrpcClient* grpc_client;
    ptre_global.grpc_client_cache->GetClient(dst_rank, &grpc_client);
    grpc_client->AckPushDone();
  }
}

void ProcessRequestQueue() {
  if (!ptre_global.is_broadcast_done) {
    ProcessRequestQueueInternal1();
    return;
  }
  if (ptre_global.num_push == 1) {
    //ProcessRequestQueueInternal1();
    ProcessRequestQueueInternalBufferedAggregation();
  } else {
#if 0
    ProcessRequestQueueInternalAtomicAdd();
#else
    ProcessRequestQueueInternalBufferedAggregation();
#endif
  }
}

void BackgroundThreadLoop() {
  while (!ptre_global.is_shutdown) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ProcessRequestQueue();
  }
}

int ProcessAggregation() {
  return ptre_global.cm.ProcessAggregation();
}

void AggregationThreadLoop() {
  ptre::TensorAggregator* agg = ptre_global.cm.tensor_aggregator();
  auto last_time = std::chrono::system_clock::now();
  while (!ptre_global.is_shutdown) {
    auto curr_time = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    int ret = ProcessAggregation();
    if (ret > 0) {
      //LOG(INFO) << "[DEBUG] agg_cnt=" << ret;
      last_time = curr_time;
    }
    std::chrono::duration<double> since_last = curr_time - last_time;
    if (since_last.count() > 15) {
      LOG(INFO) << "[DEBUG] Agg not performed for a while.";
      agg->PrintDebug();
      last_time = curr_time;
    }
  }
}

void InitPtreOnce() {
  //while(true) {
  //  if (ptre_global.rdma_manager->IsMRInitialized()) {
  //    break;
  //  }
  //}

}

void load_grpc_hosts(const string& grpc_hosts_file) {
  std::string in_line;
  //std::ifstream in("/home/wkim/experiments/grpc_hosts");
  std::ifstream in(grpc_hosts_file);
  while (std::getline(in, in_line)) {
    if (in_line[0] == '#') continue;
    ptre_global.grpc_hosts.emplace_back(in_line);
  }
  in.close();
  for (int i = 0; i < ptre_global.size; i++) {
    std::cout << ptre_global.grpc_hosts[i] << std::endl;
  }
}

#if 0
void PtreBarrier() {
  if (ptre_global.rank == 0) {
    while (true) {
      usleep(1);
      uint64_t read_val = *ptre_global.barrier_counter;
      if (read_val == ptre_global.size - 1) {
      }
    }
  }
}
#endif

void InitComm(int size, int rank, const string& grpc_hosts_file) {
  /// First Initialization step
  ptre_global.size = size;
  ptre_global.rank = rank;
  ptre_global.cm.set_size(size);
  ptre_global.cm.set_rank(rank);
  // InitPeerSelector

  load_grpc_hosts(grpc_hosts_file);

  ptre_global.grpc_client_cache = std::make_shared<GrpcClientCache>(rank,
      ptre_global.grpc_hosts);

  /// Init RdmaManager
  std::cout << "Initializing RdmaManager" << std::endl;
  ptre_global.rdma_manager = new RdmaManager(size, rank, false);
  ptre_global.cm.SetRdmaManager(ptre_global.rdma_manager);
  ptre_global.is_shutdown = false;
  ptre_global.background_thread = std::thread(BackgroundThreadLoop);
}

void InitGrpcService() {
  std::cout << "Running grpc server" << std::endl;
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);
}

void InitRemoteMRV2() {
  std::vector<BufType> buf_types;
  std::vector<string> names;
  int num_bufs = ptre_global.rdma_manager->
      GetRemoteAccessBufInfos(&buf_types, &names);
#if 0
  for (int i = 0; i < num_bufs; i++) {
    LOG(INFO) << "type=" << buf_types[i] << ", name=" << names[i];
  }
#endif
  for (int i = 0; i < ptre_global.size; i++) {
    if (i != ptre_global.rank) {
      GrpcClient* grpc_client;
      ptre_global.grpc_client_cache->GetClient(i, &grpc_client);
      grpc_client->SetRdmaManager(ptre_global.rdma_manager);
    }
  }
  bool peer_flag[ptre_global.size] = { };
  //for (int j = 0; j < num_bufs; j++) {
  //  struct ibv_mr* mr = ptre_global.rdma_manager->GetMR(buf_types[j], names[j]);
  //  ptre_global.rdma_manager->SetRemoteMRV2(ptre_global.rank, buf_types[j],
  //      names[j], (uint64_t) mr->addr, mr->rkey);
  //}
  peer_flag[ptre_global.rank] = true;
  int done_flag = 0;
  while (!done_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    done_flag = 1;
    for (int i = 0; i < ptre_global.size; i++) {
      if (peer_flag[i]) {
        continue;
      }
      LOG(INFO) << "Get Remote Address of RANK:" << i;
      GrpcClient* grpc_client;
      ptre_global.grpc_client_cache->GetClient(i, &grpc_client);
      if (!ptre_global.rdma_manager->IsDlidSet(i)) {
        int ret = grpc_client->GetRemoteEnv();
        if (ret < 0) {
          done_flag = 0;
          continue;
        }
      }
      int client_status = 0;
      for (int j = 0; j < num_bufs; j++) {
        if (ptre_global.rdma_manager->
            IsRemoteMRSetV2(i, buf_types[j], names[j])) {
          continue;
        }
        RemoteMR rmr;
        int ret = grpc_client->GetRemoteAddressV2(buf_types[j], names[j], &rmr);
        if (ret < 0) {
          client_status = -1;
          break;
        } else {
          ptre_global.rdma_manager->SetRemoteMRV2(i, buf_types[j], names[j],
              rmr.remote_addr, rmr.rkey);
        }
      }
      if (client_status < 0) {
        done_flag = 0;
        continue;
      }
      peer_flag[i] = true;
    }
  }
}

void ConnectQPs() {
  int done_flag = 0;
  while (!done_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    done_flag = 1;
    for (int i = 0; i < ptre_global.size; i++) {
#if 1
      if (i != ptre_global.rank) {
#else
      if (true) {
#endif
        int r = ptre_global.rdma_manager->ConnectQP(i);
        if (r < 0) {
          done_flag = 0;
        }
      }
    }
  }
}

}  // namespace


//namespace functor {
//template <typename T>
//struct ApplyModelAveraging<CPUDevice, T> {
//  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
//                  typename TTypes<T>::ConstFlat remote) {
//    var.device(d) = 0.5 * (var + remote);
//  }
//};
//}  // namespace functor

REGISTER_OP("InitComm")
    .Attr("size: int")
    .Attr("rank: int")
    .Attr("grpc_hosts_file: string = '/home/wkim/experiments/grpc_hosts'");
class InitCommOp : public OpKernel {
 public:
  explicit InitCommOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr<int>("size", &size_);
    context->GetAttr<int>("rank", &rank_);
    context->GetAttr<string>("grpc_hosts_file", &grpc_hosts_file_);
  }
  void Compute(OpKernelContext* context) override {
    InitComm(size_, rank_, grpc_hosts_file_);
  }
 private:
  int size_;
  int rank_;
  string grpc_hosts_file_;
};
REGISTER_KERNEL_BUILDER(Name("InitComm").Device(DEVICE_CPU),
                        InitCommOp);

REGISTER_OP("InitGlobalConsensus")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class InitGlobalConsensusOp : public OpKernel {
 public:
  explicit InitGlobalConsensusOp(OpKernelConstruction* context)
      : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.cm.InitBufTensor(names_[i], context->input(i));
    }
    ptre_global.cm.InitBufParam();
    //std::vector<const Tensor*> inputs;
    //for (int i = 0; i < num_inputs; i++) {
    //  const Tensor &input = context->input(i);
    //  inputs.push_back(&input);
    //}
    //ptre_global.cm.InitGlobalConsensus(inputs);
    //auto recv_tensors = ptre_global.cm.GetGlobalConsensusList();
    //auto send_tensors = ptre_global.cm.GetSendTensorsList();
    //for (int i = 0; i < ptre_global.size; i++) {
    //  if (i == ptre_global.rank) {
    //    continue;
    //  }
    //  for (int j = 0; j < num_inputs; j++) {
    //    ptre_global.rdma_manager->InitTensorMR(i, names_[j], recv_tensors[j],
    //                                           send_tensors[j]);
    //  }
    //}
    //ptre_global.rdma_manager->MarkMRInitialized();
    //usleep((ptre_global.size - ptre_global.rank) * 1000000);
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("InitGlobalConsensus").Device(DEVICE_CPU),
                        InitGlobalConsensusOp);

REGISTER_OP("InitRemoteMr")
    .Attr("names: list(string)");
class InitRemoteMrOp : public OpKernel {
 public:
  explicit InitRemoteMrOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    bool peer_flag[ptre_global.size] = { };
    peer_flag[ptre_global.rank] = true;
    int done_flag = 0;
    while (!done_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      done_flag = 1;
      for (int i = 0; i < ptre_global.size; i++) {
        if (peer_flag[i]) {
          continue;
        }
        //std::unique_ptr<GrpcClient> grpc_client;
        GrpcClient* grpc_client;
        ptre_global.grpc_client_cache->GetClient(i, &grpc_client);
        //GrpcClient grpc_client(ptre_global.rank, i, ptre_global.grpc_hosts[i]);
        grpc_client->SetRdmaManager(ptre_global.rdma_manager);
        if (!ptre_global.rdma_manager->IsDlidSet(i)) {
          int ret = grpc_client->GetRemoteEnv();
          if (ret < 0) {
            done_flag = 0;
            continue;
          }
        }
        int client_status = 0;
        for (int j = 0; j < names_.size(); j++) {
          if (ptre_global.rdma_manager->IsRemoteMRSet(i, names_[j])) {
            continue;
          }
          int ret = grpc_client->GetRemoteAddress(names_[j]);
          if (ret < 0) {
            client_status = -1;
            break;
          }
        }
        if (client_status < 0) {
          done_flag = 0;
          continue;
        }
        if (!ptre_global.rdma_manager->IsRemoteParamMRSet(i)) {
          int ret = grpc_client->GetRemoteParamAddress();
          if (ret < 0) {
            done_flag = 0;
            continue;
          }
        }
        peer_flag[i] = true;
      }
    }
    std::cout << "Init RemoteMR done." << std::endl;
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("InitRemoteMr").Device(DEVICE_CPU),
                        InitRemoteMrOp);

/// MR management V2
///
REGISTER_OP("InitStepOne")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class InitStepOneOp : public OpKernel {
 public:
  explicit InitStepOneOp(OpKernelConstruction* context)
      : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();
    std::vector<const Tensor*> inputs;
    for (int i = 0; i < num_inputs; i++) {
      const Tensor& input = context->input(i);
      inputs.push_back(&input);
    }
    LOG(INFO) << "Initializing global consensus: num_tensors=" << num_inputs;

    /// Register MR
    ptre_global.cm.InitGlobalConsensusV2(names_, inputs);
    ptre_global.aggregation_thread = std::thread(AggregationThreadLoop);
    InitGrpcService();
    InitRemoteMRV2();
    ptre_global.rdma_manager->InitAggWriter();
    ConnectQPs();
  }
 private:
  std::vector<string> names_;
};
REGISTER_KERNEL_BUILDER(Name("InitStepOne").Device(DEVICE_CPU),
                        InitStepOneOp);

REGISTER_OP("InitGlobalConsensusV2")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class InitGlobalConsensusV2Op : public OpKernel {
 public:
  explicit InitGlobalConsensusV2Op(OpKernelConstruction* context)
      : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    std::vector<const Tensor*> inputs;
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      const Tensor& input = context->input(i);
      inputs.push_back(&input);
    }
    ptre_global.cm.InitGlobalConsensusV2(names_, inputs);
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("InitGlobalConsensusV2").Device(DEVICE_CPU),
                        InitGlobalConsensusV2Op);

REGISTER_OP("ConnectQps");
class ConnectQpsOp : public OpKernel {
 public:
  explicit ConnectQpsOp(OpKernelConstruction* context) : OpKernel(context) { }
  void Compute(OpKernelContext* context) override {
    int done_flag = 0;
    while (!done_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      done_flag = 1;
      for (int i = 0; i < ptre_global.size; i++) {
        if (i == ptre_global.rank) {
          continue;
        }
        int r = ptre_global.rdma_manager->ConnectQP(i);
        if (r < 0) {
          done_flag = 0;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ConnectQps").Device(DEVICE_CPU),
                        ConnectQpsOp);

REGISTER_OP("PushModel")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class PushModelOp : public OpKernel {
 public:
  explicit PushModelOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    //std::cout << (void*) context->input(0).tensor_data().begin() << std::endl;
    int num_inputs = context->num_inputs();
    //std::cout << "PushTensor: " << names_[0] << std::endl;
    //std::lock_guard<std::mutex> l(ptre_global.mu);
    //while (!ptre_global.q.empty()) {
    //  ptre_global.q.pop();
    //}
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.cm.CopyTensorSend(names_[i], context->input(i));
    }
    int target_rank = ptre_global.cm.GetRandomTarget();
    //int target_rank = ptre_global.cm.GetIncNeighbor();
    ptre_global.q.push(target_rank);
    //ptre_global.cm.SetPushReady();
    //std::vector<const Tensor*> inputs;
    //for (int i = 0; i < num_inputs; i++) {
    //  const Tensor &input = context->input(i);
    //  inputs.push_back(&input);
    //}
    //ptre_global.cm.EnqueuePushList(inputs);
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("PushModel").Device(DEVICE_CPU), PushModelOp);

REGISTER_OP("BroadcastModel")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class BroadcastModelOp : public OpKernel {
 public:
  explicit BroadcastModelOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    std::cout << "Broadcasting model" << std::endl;
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.cm.CopyTensorSend(names_[i], context->input(i));
    }
    auto& mu = ptre_global.q_mu;
    for (int target_rank = 0; target_rank < ptre_global.size; target_rank++) {
      //std::cout << "Queuing for target" << target_rank << std::endl;
      if (target_rank == ptre_global.rank) {
        continue;
      }
      ptre_global.cm.PushTensors(target_rank);
      //mu.lock();
      //ptre_global.q.push(target_rank);
      //mu.unlock();
    }

    //std::cout << "Wait for send done." << std::endl;
    //bool cond = false;
    //while (!cond) {
    //  mu.lock();
    //  //std::cout << "Queue size=" << ptre_global.q.size() << std::endl;
    //  cond = ptre_global.q.empty();
    //  mu.unlock();
    //}
    //ptre_global.is_broadcast_done = true;
  }

 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("BroadcastModel").Device(DEVICE_CPU), BroadcastModelOp);

REGISTER_OP("GetRemoteVariable")
    .Attr("index: int = -1")
    .Attr("var_name: string = ''")
    .Output("var: float32");
class GetRemoteVariableOp : public OpKernel {
 public:
  explicit GetRemoteVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("index", &index_));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* context) override {
    //Tensor other(ptre_global.cm.global_consensus(index_));
    usleep(1);
    Tensor other;
    if (index_ >= 0) {
      other = ptre_global.cm.global_consensus(index_);
    //}
    } else {
      //std::cout << "get_remote_variable with name: " << var_name_ << std::endl;
      other = ptre_global.cm.global_consensus(var_name_);
    }
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, other.shape(), &output));
    //std::copy(other.tensor_data().begin(), other.tensor_data().end(),
    //          const_cast<char*>(output->tensor_data().begin()));
    auto output_flat = output->flat<float>();
    output_flat = other.flat<float>();
  }

 private:
  int index_;
  std::string var_name_;
};
REGISTER_KERNEL_BUILDER(Name("GetRemoteVariable").Device(DEVICE_CPU),
                        GetRemoteVariableOp);

REGISTER_OP("GetRemoteVariables")
    .Output("out: float32");
class GetRemoteVariablesOp : public OpKernel {
 public:
  explicit GetRemoteVariablesOp(OpKernelConstruction* context)
      : OpKernel(context) { }
  void Compute(OpKernelContext* context) override {
    int num_vars = ptre_global.cm.num_vars();
    for (int i = 0; i < num_vars; i++) {
      Tensor other(ptre_global.cm.global_consensus(i));
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(i, other.shape(), &output));
      auto output_flat = output->flat<float>();
      output_flat = other.flat<float>();
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GetRemoteVariables").Device(DEVICE_CPU),
                        GetRemoteVariablesOp);

REGISTER_OP("GetSendTensor")
    .Attr("index: int")
    .Output("var: float32");
class GetSendTensorOp : public OpKernel {
 public:
  explicit GetSendTensorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("index", &index_));
  }
  void Compute(OpKernelContext* context) override {
    Tensor* other(ptre_global.cm.send_tensor(index_));
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, other->shape(), &output));
    auto output_flat = output->flat<float>();
    output_flat = other->flat<float>();
  }

 private:
  int index_;
};
REGISTER_KERNEL_BUILDER(Name("GetSendTensor").Device(DEVICE_CPU),
                        GetSendTensorOp);

REGISTER_OP("AverageModelWithRemote")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T");
class AverageModelWithRemoteOp : public OpKernel {
 public:
  explicit AverageModelWithRemoteOp(OpKernelConstruction* context)
      : OpKernel(context) {
    //OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
  }
  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      const Tensor &input = context->input(i);
      //Tensor input = context->mutable_input(i, true);
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(i, input.shape(), &output));
      //OP_REQUIRES_OK(context,
      //               context->forward_input_or_allocate_output(
      //                   {i}, i, input.shape(), &output));
      // Load remote tensor from CM
      Tensor other(ptre_global.cm.global_consensus(i));
      //Tensor other(input.dtype(), input.shape());
      //auto other_flat = other.flat<float>();
      //for (int j = 0; j < var_sizes_[i]; j++) {
      //  other_flat(j) = 1.0;
      //}

      // Averaging
      //auto input_flat = input.flat<float>();
      //input_flat = 0.5 * (input_flat + other_flat);
      auto output_flat = output->flat<float>();
      output_flat = 0.5 * (input.flat<float>() + other.flat<float>());
      //std::copy(output->tensor_data().begin(), output->tensor_data().end(),
      //          const_cast<char*>(input.tensor_data().begin()));
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("AverageModelWithRemote").Device(DEVICE_CPU),
                        AverageModelWithRemoteOp);

//REGISTER_OP("PushModel")
//    .Attr("T: {float32}")
//    .Attr("NumTensors: int")
//    .Input("var: NumTensors * T");
//
//class PushModel : public OpKernel {
// public:
//  void Compute(OpKernelContext *context) override {
//    //ptre_global->PushModel
//  }
//};

//REGISTER_OP("Incoming")
//    .

REGISTER_OP("ApplyModelAveraging")
    .Input("var: Ref(T)")
    .Input("remote: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype");
template <typename T>
class ApplyModelAveragingOp : public OpKernel {
 public:
  explicit ApplyModelAveragingOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, false);
    const Tensor& remote = ctx->input(1);
    //const Device& device = ctx->template eigen_device<Device>();
//  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
//                  typename TTypes<T>::ConstFlat remote) {
//    var.device(d) = 0.5 * (var + remote);
//  }
    //functor::ApplyModelAveraging<Device, T>()(
    //    device, var.flat<T>(), remote.flat<T>());
    var.flat<T>() = 0.5 * (var.flat<T>() + remote.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }
};
REGISTER_KERNEL_BUILDER(
    Name("ApplyModelAveraging").Device(DEVICE_CPU).TypeConstraint<float>("float"),
    ApplyModelAveragingOp<float>);

REGISTER_OP("IsNewIncoming")
    .Output("out: bool");
class IsNewIncomingOp : public OpKernel {
 public:
  explicit IsNewIncomingOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }
  void Compute(OpKernelContext* ctx) {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ }), &output));
    usleep(1);
    //volatile bool* ret = ptre_global.cm.is_new_incoming_ptr();
    bool* ret = ptre_global.cm.is_new_incoming_ptr();
    //std::cout << "IsNewIncomingOp: *is_new_incoming_ptr = " << *ret << std::endl;
    std::copy(ret, ret + sizeof(bool), const_cast<char*>(output->tensor_data().begin()));
    //std::cout << "IsNewIncomingOp: *((bool*) output->tensor_data().begin()) = " << *((bool*) output->tensor_data().begin()) << std::endl;
  }
};
REGISTER_KERNEL_BUILDER(
    Name("IsNewIncoming").Device(DEVICE_CPU), IsNewIncomingOp);

REGISTER_OP("MarkNoNew");
class MarkNoNewOp : public OpKernel {
 public:
  explicit MarkNoNewOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }
  void Compute(OpKernelContext* ctx) {
    ptre_global.cm.MarkNoNew();
    //volatile bool* ret = ptre_global.cm.is_new_incoming_ptr();
    bool* ret = ptre_global.cm.is_new_incoming_ptr();
    //std::cout << "MarkNoNewOp: *is_new_incoming_ptr = " << *ret << std::endl;
  }
};
REGISTER_KERNEL_BUILDER(
    Name("MarkNoNew").Device(DEVICE_CPU), MarkNoNewOp);

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace functor {
template <typename T>
struct Modelaverage<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar m,
                  typename TTypes<T>::ConstFlat other) {
    var.device(d) = (var + other) / m();
  }
};

template <typename T>
struct LinearWeightedAverageApprox<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar c1,
                  typename TTypes<T>::ConstFlat other,
                  typename TTypes<T>::ConstScalar c2) {
    var.device(d) = c1() * var + c2() * other;
  }
};

template <typename T>
struct CopyTensorToSendBuf<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Flat src,
                  typename TTypes<T>::Flat dst) {
    auto bytes = sizeof(T) * src.size();
    memcpy(dst.data(), src.data(), bytes);
  }
};
}  // namespace functor

//REGISTER_OP("ResourceGetGlobalConsensus")
//  .Input("var: resource")
//  .Attr("T: numbertype")
//  .Attr("var_name: string")
//  .Output("gcon: resource")
//template <typename Device, typename T>
//class GetRemoteOp : public OpKernel {
// public:
//  explicit GetRemoteOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
//    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
//  }
//  void Compute(OpKernelContext* ctx) {
//    Tensor* output;
//    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, var.shape(), &output));
//    const Tensor other(ptre_global.cm.global_consensus(var_name_));
//  }
// private:
//  string var_name_;
//};

static Status ModelaverageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                  // var
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceModelaverage")
  .Input("var: resource")
  .Attr("T: numbertype")
  .Attr("var_name: string")
  .SetShapeFn(ModelaverageShapeFn);
template <typename Device, typename T>
class ModelaverageOp : public OpKernel {
 public:
  explicit ModelaverageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* ctx) {
    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    Tensor var;
    var = *ref->tensor();

    bool do_reduce = true;
    int num_incomings = 1;
#if 1
    if (1) {
      //if (ptre_global.cm.IsInitNumApplyOps() && ptre_global.ever_pushed) {
      //  LOG(INFO) << "[DEBUG] Waiting for rcv_ing_cnt == num_push";
      //  while (true) {
      //    if (ptre_global.cm.rcv_ing_cnt() == ptre_global.num_push) {
      //      break;
      //    }
      //  }
      //  LOG(INFO) << "[DEBUG] rcv_ing_cnt=" << ptre_global.cm.rcv_ing_cnt();
      //}
      if (ptre_global.ever_pushed) {
        int old;
        while (1) {
          old = ptre_global.ma_op_cnt.fetch_add(1);
          if (old < ptre_global.cm.num_apply_ops()) {
            break;
          }
        }
        if (old == 0) {
          LOG(INFO) << "[DEBUG] Waiting for Agg Step=" << ptre_global.local_step;
        }
      }
      num_incomings = ptre_global.cm.WaitAndGetNumIncomings();
      if (ptre_global.ever_pushed) {
        int old = ptre_global.ma_op_cnt2.fetch_add(1);
        if (old == ptre_global.cm.num_apply_ops() - 1) {
          LOG(INFO) << "[DEBUG] Done Waiting for Aggs" << ptre_global.local_step;
          ptre_global.ma_op_cnt.store(0);
          ptre_global.ma_op_cnt2.store(0);
        }
      }
      if (num_incomings > 0) {
        //LOG(INFO) << "[DEBUG] RECV and AGG all done: num_incomings = " << num_incomings;
      }
      if (num_incomings == 0) {
        do_reduce = false;
      }
    } else {
      bool* ret = ptre_global.cm.is_new_incoming_ptr();
      if (!*ret) {
        do_reduce = false;
      }
    }
#else
    do_reduce = false;
#endif
    if (do_reduce) {
      const Device& d = ctx->template eigen_device<Device>();
      const Tensor other(ptre_global.cm.global_consensus(var_name_));
#if 0
      Tensor m_(DataTypeToEnum<T>::v(), TensorShape({ }));
      m_.flat<T>()(0) = T(num_incomings + 1);
      const Tensor m(m_);
      functor::Modelaverage<Device, T>()(d, var.flat<T>(), m.scalar<T>(),
          other.flat<T>());
#else
      int old = ptre_global.reduce_op_cnt0.fetch_add(1);
      Tensor c1_(DataTypeToEnum<T>::v(), TensorShape({ }));
      Tensor c2_(DataTypeToEnum<T>::v(), TensorShape({ }));
      int k = ptre_global.virtual_step;
      int k_r_sum = ptre_global.cm.rcv_steps_sum();
      int k_sum = k + k_r_sum;
      int n = num_incomings + 1;
      float k_r_avg = (float) k_r_sum / (n - 1);
      float c1_val = (float) k / k_sum;
      float c2_val = (float) k_r_avg / k_sum;
      int www = 0;
      if (k < www) {
        float delta = 0.5 - c1_val;
        c1_val = c1_val + (delta / www) * (www - k);
        c2_val = (1.0 - c1_val) / (n - 1);
      }
      c1_.flat<T>()(0) = T(c1_val);
      c2_.flat<T>()(0) = T(c2_val);
      const Tensor c1(c1_);
      const Tensor c2(c2_);
      if (old == 0) {
        LOG(INFO) << "[DEBUG]\nlocal_step=" << ptre_global.local_step
            << "\nvirtual_step=" << ptre_global.virtual_step
            << "\nc1=" << c1_.flat<T>()(0)
            << ", c2=" << c2_.flat<T>()(0)
            << ", c1 + (n-1)c2 = " << c1_.flat<T>()(0) + (T) (n - 1) * c2_.flat<T>()(0)
            << "\nk=" << k << ", k_r_avg=" << k_r_avg;
      }
      functor::LinearWeightedAverageApprox<Device, T>()(d, var.flat<T>(),
          c1.scalar<T>(), other.flat<T>(), c2.scalar<T>());
      int k_avg = k_sum / n;
      if (k_avg > k) {
        ptre_global.virtual_step = k_avg;
        ptre_global.cm.set_virtual_step(k_avg);
      }
      old = ptre_global.reduce_op_cnt1.fetch_add(1);
      if (old == ptre_global.cm.num_apply_ops() - 1) {
        ptre_global.reduce_op_cnt0 = 0;
        ptre_global.reduce_op_cnt1 = 0;
      }
#endif
      ptre_global.incoming_state = 2;  // Used
    }
    //LOG(INFO) << "CountReduceAndOpenRecv() Entering.";
    ptre_global.cm.CountReduceAndOpenRecv(var_name_);
    //LOG(INFO) << "CountReduceAndOpenRecv() Done.";
  }

 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("ResourceModelaverage") \
                              .Device(DEVICE_##D)      \
                              .HostMemory("var")       \
                              .TypeConstraint<T>("T"), \
                          ModelaverageOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void Modelaverage<GPUDevice, T>::operator()(          \
      const GPUDevice& d, typename TTypes<T>::Flat var, \
      typename TTypes<T>::ConstScalar m,                \
      typename TTypes<T>::ConstFlat other);             \
  extern template struct Modelaverage<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
}  // namespace functor
#endif  // GOOGLE_CUDA

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS


REGISTER_OP("ResourcePushTensor")
  .Input("var: resource")
  .Attr("T: numbertype")
  .Attr("var_name: string");
template <typename Device, typename T>
class PushTensorOp : public OpKernel {
 public:
  explicit PushTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* ctx) {
    if (ptre_global.push_step_state != 1) {
      return;
    }
    Tensor var;
    core::RefCountPtr<Var> ref;
    //TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &ref));
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    var = *ref->tensor();
    const Device& d = ctx->template eigen_device<Device>();
    int old;
    while (1) {
      old = ptre_global.copy_cnt[0].fetch_add(1);
      if (old < ptre_global.cm.num_apply_ops()) {
        break;
      }
    }
    if (old == 0) LOGSTEP << "Starting CopyToSendBuf...";
#if 0
    ptre_global.q_mu.lock();
    int old = ptre_global.copy_cnt[0].fetch_add(1);
    if (old == 0) {
      LOG(INFO) << "[DEBUG] Starting CopyToSendBuf... (step=" << ptre_global.local_step << ")";
    }
    int q_size = ptre_global.q.size();
    LOGSTEP << "Number of remaining queued send requests=" << q_size;
    /// Waiting for the on-going send request.
    if (old == 0) LOGSTEP << "Checking if SendBuf is idle.";
    std::unique_lock<std::mutex> lk(ptre_global.cm.send_mu_);
    ptre_global.cm.send_cv_.wait(lk, [&] {
        return ptre_global.cm.send_status_ == ConsensusManager::SEND_IDLE;
        });

    T* send_buf = (T*) ptre_global.cm.buf_ptr(ptre::BUF_TYPE_SEND_BUF,
        var_name_);
    typename TTypes<T>::Flat send_flat(send_buf, var.flat<T>().size());
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), send_flat);
    lk.unlock();
#elif 0
    Tensor* send_tensor = ptre_global.cm.send_tensor(var_name_);
    // TODO: Make it more correct! (locking)
    std::unique_lock<std::mutex> lk(ptre_global.cm.send_mu_);
    ptre_global.cm.send_cv_.wait(lk, [&] {
        return ptre_global.cm.send_status_ == ConsensusManager::SEND_IDLE;
        });
    lk.unlock();
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(),
        send_tensor->flat<T>());
#elif 1
    T* send_buf = (T*) ptre_global.cm.buf_ptr(ptre::BUF_TYPE_SEND_BUF,
        var_name_);
    // TODO: Make it more correct! (locking)
    int SKIP_IF_SEND_BUF_IS_BUSY = 2;
    if (SKIP_IF_SEND_BUF_IS_BUSY == 1) {
      /// Don't wait for the running send operations at this point.
      auto& mu = ptre_global.cm.send_mu_;
      bool skip = false;
      ptre_global.q_mu.lock();
      if (!ptre_global.q.empty()) {
        ptre_global.push_step_state = 2;
      }
      ptre_global.q_mu.unlock();
      mu.lock();
      if (ptre_global.push_step_state == 1) {  // PUSH
        if (ptre_global.cm.send_status_ != ConsensusManager::SEND_IDLE) {
          /// Skip
          /// NOT ENQUEUEING (NOT ATTEMPT)
          ptre_global.push_step_state = 2;  // SKIP
          skip = true;
        }
      } else if (ptre_global.push_step_state == 2) {  // SKIP set by another
        skip = true;
      } else {
        LOG(ERROR) << "Something's gone wrong.";
        exit(EXIT_FAILURE);
      }
      mu.unlock();
      if (skip) {
        return;
      }
    } else if (SKIP_IF_SEND_BUF_IS_BUSY == 0) {
      /// Wait for the running send operations at this point.
      std::unique_lock<std::mutex> lk(ptre_global.cm.send_mu_);
      ptre_global.cm.send_cv_.wait(lk, [&] {
          return ptre_global.cm.send_status_ == ConsensusManager::SEND_IDLE;
          });
      lk.unlock();
    } else {
      bool q_empty = false;
      while (!q_empty) {
        ptre_global.q_mu.lock();
        if (ptre_global.q.empty()) {
          q_empty = true;
        } else {
          //LOG(INFO) << "[DEBUG] Queue is not empty. size=" << ptre_global.q.size();
        }
        ptre_global.q_mu.unlock();
      }
      std::unique_lock<std::mutex> lk(ptre_global.cm.send_mu_);
      ptre_global.cm.send_cv_.wait(lk, [&] {
          return ptre_global.cm.send_status_ == ConsensusManager::SEND_IDLE;
          });
      lk.unlock();
    }
    typename TTypes<T>::Flat send_flat(send_buf, var.flat<T>().size());
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), send_flat);
#else
    T* send_buf = (T*) ptre_global.cm.buf_ptr(ptre::BUF_TYPE_SEND_BUF,
        var_name_);
    typename TTypes<T>::Flat send_flat(send_buf, var.flat<T>().size());
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), send_flat);
#endif
    int old2 = ptre_global.copy_cnt[1].fetch_add(1);
    if (old2 == ptre_global.cm.num_apply_ops() - 1) {
      LOGSTEP << "Done CopyToSendBuf.";
      ptre_global.copy_cnt[0] = 0;
      ptre_global.copy_cnt[1] = 0;
    }
  }
 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("ResourcePushTensor")   \
                              .Device(DEVICE_##D)      \
                              .HostMemory("var")       \
                              .TypeConstraint<T>("T"), \
                          PushTensorOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void CopyTensorToSendBuf<GPUDevice, T>::operator()(   \
      const GPUDevice& d, typename TTypes<T>::Flat src, \
      typename TTypes<T>::Flat dst);                    \
  extern template struct CopyTensorToSendBuf<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
}  // namespace functor
#endif  // GOOGLE_CUDA

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace tensorflow


extern "C" {

int ptre_init(int size, int rank, char* grpc_hosts_file,
              int selection_strategy, int num_push) {
  tensorflow::ptre_global.num_push = num_push;
  tensorflow::InitComm(size, rank, grpc_hosts_file);
  tensorflow::ptre_global.cm.InitPeerSelector(selection_strategy, num_push);
  LOG(INFO) << "Peer selection strategy = " << selection_strategy;
}

int ptre_init_rdma_grpc_service() {
  tensorflow::InitGrpcService();
}

int ptre_size() {
  return tensorflow::ptre_global.size;
}

int ptre_rank() {
  return tensorflow::ptre_global.rank;
}

bool ptre_is_new_incoming() {
  bool* ret = tensorflow::ptre_global.cm.is_new_incoming_ptr();
  return *ret;
}

void ptre_mark_no_new() {
  bool* ret = tensorflow::ptre_global.cm.is_new_incoming_ptr();
  if (*ret == true && tensorflow::ptre_global.incoming_state == 2) {
    *ret = false;
    tensorflow::ptre_global.incoming_state = 0;
  }
}

void ptre_enqueue_push() {
  using tensorflow::ptre_global;
  ptre_global.ever_pushed = 1;
  std::vector<bool> checker(ptre_global.size, 0);
  //int target_rank = tensorflow::ptre_global.cm.GetRandomTarget();
  //GrpcClient grpc_client(ptre_global.rank, i, ptre_global.grpc_hosts[i]);
  auto& mu = tensorflow::ptre_global.q_mu;
  /// clear the queue
  mu.lock();
  /// TODO: If Push operations are on going at this point,
  ///  Skip enqueuing
#if 0
  while (!ptre_global.q.empty()) {
    ptre_global.q.pop();
  }
#else
  if (!ptre_global.q.empty()) {
    LOG(INFO) << "ptre_enqueue_push(): Queue is not empty, push_state=" << ptre_global.push_step_state
        << ", queue size = " << ptre_global.q.size();
    //exit(EXIT_FAILURE);
    ptre_global.push_step_state = 2;
  }
#endif

#if 0
  int n = ptre_global.num_push;
  int targets[n];
  int ret = ptre_global.cm.get_peers(n, targets);
  for (int i = 0; i < ret; i++) {
    tensorflow::GrpcClient* grpc_client;
    ptre_global.grpc_client_cache->GetClient(targets[i], &grpc_client);
    bool can_push = grpc_client->AttemptPush();
    //LOG(INFO) << "RANK:" << ptre_global.rank << " can_push=" << can_push << " for rank=" << targets[i];
    if (can_push) {
      ptre_global.q.push(targets[i]);
    }
  }
#elif 1
  /// V2
  LOG(INFO) << "[DEBUG] enqueue_push Step=" << ptre_global.local_step;
  int num_targets = 0;
  if (ptre_global.push_step_state == 1) {
    int attempt_cnt = 0;
    checker[ptre_global.rank] = 1;
    while (attempt_cnt < ptre_global.size
        && num_targets < ptre_global.num_push) {
      int target = ptre_global.cm.get_peer();
      attempt_cnt++;
      if (!checker[target]) {
        checker[target] = 1;
        //LOG(INFO) << "[DEBUG] Attempt to push to RANK=" << target;
        //checker[target] = 1;
        tensorflow::GrpcClient* grpc_client;
        ptre_global.grpc_client_cache->GetClient(target, &grpc_client);
        // TODO: Make AttemptPush returns PUSH MODE, e.g. DIRECT_WRITE / BUF_AGG
        bool can_push = grpc_client->AttemptPush(ptre_global.virtual_step);
        if (can_push) {
          ptre_global.q.push(target);
          num_targets++;
        }
      }
    }
  } else if (ptre_global.push_step_state == 2) {
  }
  LOG(INFO) << "[DEBUG] Done enqueue_push: num_targets=" << num_targets << ", Step=" << ptre_global.local_step;
#else
#endif
  mu.unlock();
}

void ptre_set_num_push(int num_push) {
  tensorflow::ptre_global.num_push = num_push;
}

void ptre_set_push() {
  tensorflow::ptre_global.push_step_state = 1;
}

void ptre_unset_push() {
  tensorflow::ptre_global.push_step_state = 0;
}

void ptre_finalize(unsigned int wait_time) {
  sleep(wait_time);
  //tensorflow::PtreBarrier();
  tensorflow::ShutdownGrpcServer();
  tensorflow::ptre_global.is_shutdown = true;
}

void ptre_synchronization_barrier() {
  // One time use only
  using tensorflow::ptre_global;
  LOG(INFO) << "RANK:" << ptre_global.rank << " Entered Barrier";
  ptre_global.barrier_variable = true;
  bool peer_flag[ptre_global.size] = { };
  peer_flag[ptre_global.rank] = true;
  bool global_flag = false;
  while (!global_flag) {
    //std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    global_flag = true;
    for (int i = 0; i < ptre_global.size; i++) {
      if (peer_flag[i]) {
        continue;
      }
      tensorflow::GrpcClient* grpc_client;
      ptre_global.grpc_client_cache->GetClient(i, &grpc_client);
      bool ret = grpc_client->Barrier();
      if (!ret) {
        global_flag = false;
      }
      peer_flag[i] = ret;
    }
  }
  ptre_global.is_broadcast_done = true;
  ptre_global.cm.set_rcv_done_cnt(0);
}

void ptre_init_num_rcv_tensors() {
  LOG(INFO) << "Init num_rcv_tensors";
  tensorflow::ptre_global.cm.InitNumRecvTensors();
}

void ptre_set_broadcast_not_done() {
  tensorflow::ptre_global.is_broadcast_done = false;
}

void ptre_count_step() {
  //tensorflow::ptre_global.local_step++;
  tensorflow::ptre_global.virtual_step++;
  tensorflow::ptre_global.cm.count_virtual_step();
}

void ptre_set_local_step(int local_step) {
  tensorflow::ptre_global.local_step = local_step;
  tensorflow::ptre_global.cm.set_local_step(local_step);
}

}
