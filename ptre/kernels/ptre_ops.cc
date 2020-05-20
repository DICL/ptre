//#include "ptre/core/ptre_global.h"

#define EIGEN_USE_THREADS

#include "ptre/kernels/ptre_ops.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <typeinfo>
#include <unistd.h>
#include <vector>

#include <infiniband/verbs.h>

#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/grpc/grpc_client_cache.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/rdma.h"
#include "ptre/communication/rdma/rdma_manager.h"
//#include "ptre/tensorflow/types.h"
#include "ptre/kernels/job_def.h"
#include "ptre/lib/cache_ctl.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

#define LOGSTEP LOG(INFO) << "[DEBUG,step=" << ptre_global.local_step << "]: "

//using ptre::PushJob;

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
using ::ptre::BufType;
using ::ptre::ConsensusManager;
using ::ptre::Flat;
using ::ptre::GrpcClient;
using ::ptre::GrpcClientCache;
using ::ptre::RdmaManager;
using ::ptre::RdmaServiceImpl;
using ::ptre::RemoteMR;
using ::ptre::PushRequest;
using ::ptre::PushJob;
using ::ptre::PushTask;

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
    //push_op_cnt[0] = 0;

    /// Init barrier variables
    barrier_counter = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
    barrier_release = (uint64_t*) aligned_alloc(8, sizeof(uint64_t));
    memset(barrier_counter, 0, 8);
    memset(barrier_release, 0, 8);
  }
  ConsensusManager* cm = nullptr;
  RdmaManager* rdma_manager = nullptr;
  std::mutex mu;
  std::mutex q_mu;
  std::queue<int> q;
  std::queue<std::shared_ptr<PushRequest>> req_q;
  std::mutex req_q_mu;
  // Grpc Service
  RdmaServiceImpl grpc_service;
  // Grpc Server
  std::unique_ptr<grpc::Server> grpc_server = nullptr;
  std::atomic<bool> is_shutdown;
  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::thread push_thread;
  std::thread aggregation_thread;
  std::thread receive_thread;

  int rank;
  int size;
  std::vector<std::string> grpc_hosts;
  std::shared_ptr<GrpcClientCache> grpc_client_cache = nullptr;

  // Training Infos
  int local_step = 0;
  int virtual_step = 1;
  int num_trainable_variables = -1;
  /// 0: NOT PUSH
  /// 1: PUSH
  /// 2: SKIP
  int push_step_state = 0;
  /// 0: None
  /// 1: New
  /// 2: Used
  int incoming_peer;

  bool barrier_variable = false;
  bool is_broadcast_done = true;

  // PushOp
  std::mutex push_op_mu;
  int push_op_cnt = 0;

  int num_push = 1;
  int peer_selector = 0;
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
    if (push_thread.joinable()) {
      push_thread.join();
    }
    if (grpc_server_thread.joinable()) {
      grpc_server_thread.join();
    }
    if (aggregation_thread.joinable()) {
      aggregation_thread.join();
    }
    if (receive_thread.joinable()) {
      receive_thread.join();
    }
    if (rdma_manager != nullptr) {
      delete rdma_manager;
    }
  }
};

static PtreGlobal ptre_global;

}  // namespace

void RunGrpcServer() {
  auto&& service = ptre_global.grpc_service;
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

// Non-blocking
void PtreSend(int dst_rank, char* buf, size_t len, const string& name) {
  ptre_global.grpc_service.Send(dst_rank, buf, len, name);
}

/*
void PtreSendZeroCopy(int dst_rank, std::shared_ptr<char> buf, size_t len,
    const string& name) {
  ptre_global.grpc_service.SendZeroCopy(dst_rank, buf, len, name);
}
*/

// Blocking
void PtreRecv(int src_rank, char* buf, size_t len, const string& name) {
  GrpcClient* grpc_client;
  ptre_global.grpc_client_cache->GetClient(src_rank, &grpc_client);
  grpc_client->Recv(buf, len, name);
}

void PtreBroadcast(char* buf, size_t len, int root_rank, const string& name) {
  if (ptre_global.rank == root_rank) {
    for (int i = 0; i < ptre_global.size; i++) {
      if (i == root_rank) continue;
      PtreSend(i, buf, len, name);
    }
  } else {
    PtreRecv(root_rank, buf, len, name);
  }
}

#if 0
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
    ptre_global.cm->PushTensors(dst_rank);
    GrpcClient* grpc_client;
    ptre_global.grpc_client_cache->GetClient(dst_rank, &grpc_client);
    grpc_client->NotifyPushDone();
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
      ptre_global.cm->PushTensorsV3(dst_rank);
    // TODO: 3. Notify Push done
      //std::cout << "\n[RANK=" << ptre_global.rank << "]: NotifyPushDone()\n";
      grpc_client->NotifyPushDone();
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
    ptre_global.cm->PushTensorsV3(dst_rank);
    GrpcClient* grpc_client;
    ptre_global.grpc_client_cache->GetClient(dst_rank, &grpc_client);
    grpc_client->NotifyPushDone();
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
    ProcessRequestQueueInternalBufferedAggregation();
  }
}

/*
void PushThreadLoop() {
  while (!ptre_global.is_shutdown) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ProcessRequestQueue();
  }
}
*/
#endif

int SelectPeer(PushRequest* req) {
  int attempt_cnt = 0;
  while (attempt_cnt < ptre_global.size) {
    attempt_cnt++;
    int target = ptre_global.cm->get_peer();
    if (!req->checker(target)) {
      req->check(target);
      GrpcClient* grpc_client;
      ptre_global.grpc_client_cache->GetClient(target, &grpc_client);
      // TODO: Make AttemptPush returns PUSH MODE, e.g. DIRECT_WRITE / BUF_AGG
      bool can_push = grpc_client->AttemptPush(req->step());
      if (can_push) {
        return target;
      }
    }
  }
  return -1;
}

// Return values
//  2: Send buffer is not filled.
//  3: No available peer
int ProcessPushTask(std::shared_ptr<PushTask> task) {
  // TODO: fix not to use this coarse-grained lock
  std::lock_guard<std::mutex> guard(ptre_global.push_op_mu);
  if (ptre_global.rdma_manager->IsPushReady(task->var_name())) {
    int dst = task->dst();
    if (dst < 0) {
      auto job = task->job();
      if (job->dst() < 0) {
        auto req = job->request();
        int ret = SelectPeer(req);
        if (ret < 0) {
          return 3;  // No remote peer available for push
        } else {
          job->set_dst(ret);
        }
      }
      dst = job->dst();
      task->set_dst(dst);
    }
    int push_ret = ptre_global.rdma_manager->PushAndNotify(dst,
        task->var_name());
    //LOG(INFO) << "PUSH " << "step=" << ptre_global.local_step << ", var_name=" << task->var_name() << ", to=" << dst << ", ret=" << push_ret;
    if (push_ret) {
      return 4;
    }
    return 0;  // Success
  } else {
    return 2;  // Send buffer is not ready yet
  }
}

void PushThreadLoop() {
  auto&& req_q = ptre_global.req_q;
  auto&& req_mu = ptre_global.req_q_mu;
  while (!ptre_global.is_shutdown) {
    // Get Recent Job Queue
    req_mu.lock();
    if (req_q.size() == 0) {
      req_mu.unlock();
      continue;
    }
    auto req = req_q.front();
//LOG(INFO) << "req.use_count()=" << req.use_count();
    auto&& job_q = req->q();
//LOG(INFO) << "job_q.size()=" << job_q.size();
    if (job_q.size() == 0) {
      req_q.pop();
//LOG(INFO) << "req.use_count()=" << req.use_count();
      req_mu.unlock();
      continue;
    }
    req_mu.unlock();

    auto job = job_q.front();
//LOG(INFO) << "job.use_count()=" << job.use_count();
    job_q.pop();
//LOG(INFO) << "job.use_count()=" << job.use_count();

    auto&& task_q = job->q();
//LOG(INFO) << "task_q.size()=" << task_q.size();
    auto task = task_q.front();
//LOG(INFO) << "task.use_count()=" << task.use_count();
    task_q.pop();
//LOG(INFO) << "task.use_count()=" << task.use_count();
    int ret = ProcessPushTask(task);
    //int ret = 0;
    if (ret) {
      if (ret == 2) {
        task_q.push(task);
      }
    }
    if (task_q.size() > 0) {
      job_q.push(job);
    }
  }
}

void AggregationThreadLoop() {
  while (!ptre_global.is_shutdown) {
    for (int i = 0; i < ptre_global.num_trainable_variables; i++) {
      auto rvar = ptre_global.cm->remote_variable(i);
      if (rvar) {
        rvar->Aggregate();
      }
    }
  }
}

void ReceiveThreadLoop() {
  while (!ptre_global.is_shutdown) {
    for (int i = 0; i < ptre_global.size; i++) {
      //ptre_global.cm->ReceivePushNotify(i);
      int ret = ptre_global.rdma_manager->ReceivePushNotify(i);
      if (ret >= 0) {
        int idx = ret;
        //LOG(INFO) << "RECV " << "step=" << ptre_global.local_step << ", idx=" << idx << ", from=" << i;
        auto rvar = ptre_global.cm->remote_variable(idx);
        if (rvar) {
          rvar->NewIncoming(i);
          //var->SetAggState(1);
        }
      }
    }
  }
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

void InitComm(int size, int rank, const string& grpc_hosts_file) {
  ptre_global.size = size;
  ptre_global.rank = rank;
  ptre_global.is_shutdown = false;

  // Init Grpc Service
  LOG(INFO) << "Init Grpc Service";
  load_grpc_hosts(grpc_hosts_file);
  ptre_global.grpc_client_cache = std::make_shared<GrpcClientCache>(rank,
      ptre_global.grpc_hosts);
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);

  // Init RdmaManager
  LOG(INFO) << "Init Rdma Manager";
  ptre_global.rdma_manager = new RdmaManager(size, rank);
  ptre_global.grpc_service.SetRdmaManager(ptre_global.rdma_manager);

  // Connect Queue Pairs
  LOG(INFO) << "Connect Queue Pairs";
  for (int i = 0; i < ptre_global.size; i++) {
    GrpcClient* client;
    ptre_global.grpc_client_cache->GetClient(i, &client);
    int ret = -1;
    uint16_t remote_lid;
    while (ret) {
      ret = client->GetLID(&remote_lid);
    }
    ptre_global.rdma_manager->set_remote_lid(i, remote_lid);
    ret = -1;
    uint32_t remote_qpn;
    uint32_t remote_psn;
    while (ret) {
      ret = client->GetQPAttr(&remote_qpn, &remote_psn);
    }
    ptre_global.rdma_manager->ConnectQP(i, remote_qpn);
  }

  LOG(INFO) << "[1/2] Done InitComm";
}

//namespace functor {
//template <typename T>
//struct ApplyModelAveraging<CPUDevice, T> {
//  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
//                  typename TTypes<T>::ConstFlat remote) {
//    var.device(d) = 0.5 * (var + remote);
//  }
//};
//}  // namespace functor

/// MR management V2
///
REGISTER_OP("RegisterVariables")
    .Input("vars: NumTensors * T")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)");
class RegisterVariablesOp : public OpKernel {
 public:
  explicit RegisterVariablesOp(OpKernelConstruction* context)
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
    LOG(INFO) << "Init Consensus Manager: num_trainable_vars=" << num_inputs;
    ptre_global.num_trainable_variables = num_inputs;
    ptre_global.cm = new ConsensusManager(ptre_global.size, ptre_global.rank,
        inputs, names_);
    ptre_global.grpc_service.SetConsensusManager(ptre_global.cm);
    ptre_global.cm->InitPeerSelector(ptre_global.peer_selector,
        ptre_global.num_push);

    // Register MRs
    LOG(INFO) << "Register Memory Regions";
    ptre_global.rdma_manager->SetTrainableVariables(
        ptre_global.cm->remote_variables(), names_);

    // Retrieve Remote Addresses
    LOG(INFO) << "Exchange Remote Addresses for RDMA Communication";
    for (int i = 0; i < ptre_global.size; i++) {
      GrpcClient* client;
      ptre_global.grpc_client_cache->GetClient(i, &client);
      for (int j = 0; j < num_inputs; j++) {
        int ret = -1;
        uint64_t remote_addr;
        uint32_t rkey;
        while (ret) {
          ret = client->GetRemoteAddress(ptre::BUF_TYPE_RECV_BUF, names_[j],
              &remote_addr, &rkey);
        }
        ptre_global.rdma_manager->SetRemoteAddress(i, ptre::BUF_TYPE_RECV_BUF,
            names_[j], remote_addr, rkey);
        ret = -1;
        while (ret) {
          ret = client->GetRemoteAddress(ptre::BUF_TYPE_PUSH_PERMIT, names_[j],
              &remote_addr, &rkey);
        }
        ptre_global.rdma_manager->SetRemoteAddress(i,
            ptre::BUF_TYPE_PUSH_PERMIT, names_[j], remote_addr, rkey);
      }
    }

    // Init Aggregation Thread
    LOG(INFO) << "Starting Aggregation Thread";
    ptre_global.aggregation_thread = std::thread(AggregationThreadLoop);

    // Init Receive Thread
    LOG(INFO) << "Starting Receive Thread";
    ptre_global.receive_thread = std::thread(ReceiveThreadLoop);

    // Init Push Thread
    LOG(INFO) << "Starting Push Thread";
    ptre_global.push_thread = std::thread(PushThreadLoop);

    LOG(INFO) << "[2/2] Done Registering Variables";
  }
 private:
  std::vector<string> names_;
};
REGISTER_KERNEL_BUILDER(Name("RegisterVariables").Device(DEVICE_CPU),
                        RegisterVariablesOp);

REGISTER_OP("Broadcast")
    .Input("var: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("root_rank: int");
class BroadcastOp : public OpKernel {
 public:
  explicit BroadcastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("root_rank", &root_rank_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto node_name = name();
    auto tensor = ctx->input(0);
    Tensor* output = nullptr;
    if (ptre_global.rank == root_rank_) {
      ctx->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor.shape(), &output));
    }
    if (output == nullptr) {
      output = ctx->mutable_output(0);
    }
    PtreBroadcast(const_cast<char*>(output->tensor_data().data()),
        output->tensor_data().size(), root_rank_, node_name);
  }

 private:
  int root_rank_;
};
REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), BroadcastOp);

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
    //Tensor other(ptre_global.cm->global_consensus(index_));
    usleep(1);
    Tensor other;
    if (index_ >= 0) {
      other = ptre_global.cm->global_consensus(index_);
    //}
    } else {
      //std::cout << "get_remote_variable with name: " << var_name_ << std::endl;
      other = ptre_global.cm->global_consensus(var_name_);
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
    int num_vars = ptre_global.cm->num_vars();
    for (int i = 0; i < num_vars; i++) {
      Tensor other(ptre_global.cm->global_consensus(i));
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
    Tensor* other(ptre_global.cm->send_tensor(index_));
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
      Tensor other(ptre_global.cm->global_consensus(i));
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
    //volatile bool* ret = ptre_global.cm->is_new_incoming_ptr();
    bool* ret = ptre_global.cm->is_new_incoming_ptr();
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
    ptre_global.cm->MarkNoNew();
    //volatile bool* ret = ptre_global.cm->is_new_incoming_ptr();
    bool* ret = ptre_global.cm->is_new_incoming_ptr();
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
//    const Tensor other(ptre_global.cm->global_consensus(var_name_));
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
    Tensor* ret;

    int num_incomings = 0;
    auto rvar = ptre_global.cm->remote_variable(var_name_);
    if (rvar) {
      rvar->StopRecv();
      num_incomings = rvar->agg_count();
      ret = rvar->tensor();
    }
    if (num_incomings == 0) {
      do_reduce = false;
    }
    if (do_reduce) {
      LOG(INFO) << "var_name=" << var_name_ << ", num_incomings=" << num_incomings;
      const Device& d = ctx->template eigen_device<Device>();
      const Tensor other(*ret);
#if 1
      // No Step Control
      Tensor m_(DataTypeToEnum<T>::v(), TensorShape({ }));
      m_.flat<T>()(0) = T(num_incomings + 1);
      const Tensor m(m_);
      functor::Modelaverage<Device, T>()(d, var.flat<T>(), m.scalar<T>(),
          other.flat<T>());
#else
      // Step Control
      int old = ptre_global.reduce_op_cnt0.fetch_add(1);
      Tensor c1_(DataTypeToEnum<T>::v(), TensorShape({ }));
      Tensor c2_(DataTypeToEnum<T>::v(), TensorShape({ }));
      int k = ptre_global.virtual_step;
      int k_r_sum = ptre_global.cm->rcv_steps_sum();
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
        ptre_global.cm->set_virtual_step(k_avg);
      }
      old = ptre_global.reduce_op_cnt1.fetch_add(1);
      if (old == ptre_global.cm->num_apply_ops() - 1) {
        ptre_global.reduce_op_cnt0 = 0;
        ptre_global.reduce_op_cnt1 = 0;
      }
#endif
    }
    if (rvar) {
      rvar->StartRecv();
    }
    //ptre_global.cm->OpenReceive(var_name_);
    //ptre_global.cm->CountReduce(var_name_);
    //ptre_global.cm->CountReduceAndOpenRecv(var_name_);
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

void PtreClearQueueAndEnqueueRequest() {
  auto&& req_q = ptre_global.req_q;
  auto&& req_mu = ptre_global.req_q_mu;
  req_mu.lock();
  while (req_q.size() > 0) {
    req_q.pop();
  }
  for (int i = 0; i < ptre_global.num_trainable_variables; i++) {
    ptre_global.rdma_manager->InitPush(i);
  }
  auto req = std::make_shared<PushRequest>(ptre_global.num_push,
      ptre_global.local_step, ptre_global.size,
      ptre_global.cm->variable_names());
  req_q.push(req);
  req_mu.unlock();
}

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
    if (ptre_global.push_step_state != 1) return;

    ptre_global.push_op_mu.lock();
    ptre_global.push_op_cnt++;
    if (ptre_global.push_op_cnt == 1) {
      PtreClearQueueAndEnqueueRequest();
    } else if (ptre_global.push_op_cnt == ptre_global.num_trainable_variables) {
      ptre_global.push_op_cnt = 0;
    }
    ptre_global.push_op_mu.unlock();

    Tensor var;
    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    var = *ref->tensor();
    const Device& d = ctx->template eigen_device<Device>();
    struct ibv_mr* mr = ptre_global.rdma_manager->GetMR(ptre::BUF_TYPE_SEND_BUF,
        var_name_);
    if (!mr) {
      LOG(ERROR) << "buf not found: " << ptre::BUF_TYPE_SEND_BUF << ", " << var_name_;
      exit(1);
    }
    T* send_buf = (T*) mr->addr;
    typename TTypes<T>::Flat send_flat(send_buf, var.flat<T>().size());
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), send_flat);

    ptre_global.rdma_manager->SetPushReady(var_name_);
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
  tensorflow::ptre_global.peer_selector = selection_strategy;
  tensorflow::InitComm(size, rank, grpc_hosts_file);
  //tensorflow::ptre_global.cm->InitPeerSelector(selection_strategy, num_push);
  //LOG(INFO) << "Peer selection strategy = " << selection_strategy;
}

int ptre_init_rdma_grpc_service() {
  //tensorflow::InitGrpcService();
}

int ptre_size() {
  return tensorflow::ptre_global.size;
}

int ptre_rank() {
  return tensorflow::ptre_global.rank;
}

bool ptre_is_new_incoming() {
#if 0
  bool* ret = tensorflow::ptre_global.cm->is_new_incoming_ptr();
  return *ret;
#endif
  LOG(ERROR) << "Deprecated.";
  exit(1);
}

void ptre_set_num_push(int num_push) {
  tensorflow::ptre_global.num_push = num_push;
}

void ptre_set_push() {
  using tensorflow::ptre_global;
  //auto&& q = ptre_global.req_q;
  auto&& mu = ptre_global.req_q_mu;
  mu.lock();
  ptre_global.push_step_state = 1;
  //q.emplace(ptre_global.num_push, ptre_global.local_step);
  mu.unlock();
}

void ptre_unset_push() {
  using tensorflow::ptre_global;
  auto&& mu = ptre_global.req_q_mu;
  mu.lock();
  tensorflow::ptre_global.push_step_state = 0;
  mu.unlock();
}

void ptre_finalize(unsigned int wait_time) {
  sleep(wait_time);
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
  ptre_global.cm->set_rcv_done_cnt(0);
}

#if 0
void ptre_init_num_rcv_tensors() {
  LOG(INFO) << "Init num_rcv_tensors";
  tensorflow::ptre_global.cm->InitNumRecvTensors();
}

void ptre_set_broadcast_not_done() {
  tensorflow::ptre_global.is_broadcast_done = false;
}
#endif

void ptre_count_step() {
  //tensorflow::ptre_global.local_step++;
  tensorflow::ptre_global.virtual_step++;
  tensorflow::ptre_global.cm->count_virtual_step();
}

void ptre_set_local_step(int local_step) {
  tensorflow::ptre_global.local_step = local_step;
  tensorflow::ptre_global.cm->set_local_step(local_step);
}

}
