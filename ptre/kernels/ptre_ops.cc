//#include "ptre/core/ptre_global.h"

#define EIGEN_USE_THREADS

#include "ptre/kernels/ptre_ops.h"

#include <algorithm>
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
#include <functional>
#include <mutex>
#include <sstream>
//#include <shared_mutex>
//#include "ptre/lib/shared_mutex.h"

#include <absl/base/macros.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/grpc/grpc_client_cache.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/rdma.h"
#include "ptre/communication/rdma/rdma_mgr.h"
#include "ptre/communication/rdma/rdma_task.h"
//#include "ptre/tensorflow/types.h"
#include "ptre/kernels/job_def.h"
#include "ptre/lib/cache_ctl.h"
#include "ptre/lib/concurrent_queue.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

#define LOGSTEP LOG(INFO) << "[DEBUG,step=" << ptre_global.local_step << "]: "

#define IBV_CALL(D, X) \
  while (!ptre_global.is_shutdown) { \
    std::lock_guard<std::mutex> guard(*ptre_global.qp_mus[D]); \
    int ibv_call_ret = X; \
    if (ibv_call_ret) { \
      ptre_global.rdma_mgr->RecoverQP(D); \
    } else { \
      break; \
    } \
  }

#define NUM_PUSH_THREADS 4
#define NUM_SEND_POLLING_THREADS 4
#define NUM_RECV_POLLING_THREADS 1
#define NUM_AGG_THREADS 21
#define NUM_AGG_EIGEN_THREADS 8
//#define NUM_RECV_THREADS 1

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
using ::ptre::PushTaskV2;
using ::ptre::ConcurrentQueue;
using ::ptre::ConcurrentUniqueQueue;
using ::ptre::RecvTask;
using ::ptre::RPNTask;
//using ::ptre::SharedMutex;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

struct PtreGlobal {
  PtreGlobal() {
    //std::cout << (void*) this << std::endl;
    ma_op_cnt = 0;
    ma_op_cnt2 = 0;
    reduce_op_cnt0 = 0;
    reduce_op_cnt1 = 0;
    for (int i = 0; i < num_copy_cnt; i++) {
      copy_cnt[i] = 0;
    }
    //push_op_cnt[0] = 0;


    // Counters
    agg_cnt_total = 0;
    rcv_cnt_total = 0;
    send_cnt_total = 0;
  }

  ~PtreGlobal() {
    for (auto& t : push_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    for (auto& t : send_polling_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    for (auto& t : recv_polling_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    if (grpc_server_thread.joinable()) {
      grpc_server_thread.join();
    }
    for (auto& t : aggregation_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    for (auto& t : receive_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    //if (qp_recover_thread.joinable()) {
    //  qp_recover_thread.join();
    //}
    /*
    if (rdma_mgr != nullptr) {
      delete rdma_mgr;
    }
    */
  }

  ConsensusManager* cm = nullptr;
  RdmaManager* rdma_mgr = nullptr;
  std::mutex mu;
  std::mutex q_mu;
  std::queue<int> q;
  std::queue<std::shared_ptr<PushRequest>> req_q;
  std::mutex req_q_mu;

  // Task Oriented
  std::mutex push_q_mu;
  std::vector<int> push_dsts;
  std::queue<std::shared_ptr<PushTaskV2>> push_q;

  // Grpc Service
  RdmaServiceImpl grpc_service;
  // Grpc Server
  std::unique_ptr<grpc::Server> grpc_server = nullptr;
  std::atomic<bool> is_shutdown;
  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::vector<std::thread> push_threads;
  std::vector<std::thread> send_polling_threads;
  std::vector<std::thread> recv_polling_threads;
  std::vector<std::thread> aggregation_threads;
  std::vector<std::thread> receive_threads;
  Eigen::ThreadPool* eigen_pool;

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
  std::vector<string> trainable_var_names;

  // PushOp
  //std::map<string, SharedMutex> push_var_mus;
  std::map<string, std::mutex> push_var_mus;
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

  // Counter
  std::vector<std::vector<int>> rcv_cnts;
  std::atomic<int> agg_cnt_total;
  std::atomic<int> rcv_cnt_total;
  std::atomic<int> send_cnt_total;
  std::vector<std::map<string, int>> agg_cnts;
  int agg_cnts_last = 1;

  std::map<string, int> push_success_cnt;

  std::vector<std::mutex*> qp_mus;

  std::mutex rpn_checker_mu;
  std::map<uint64_t, string> rpn_checker;
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
  //builder.SetMaxMessageSize(1 * 1024 * 1024 * 1024);
  ptre_global.grpc_server = builder.BuildAndStart();
  //std::cout << "Grpc server listening on " << server_address << std::endl;
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
  int ret = -1;
  while (ret) {
    ret = grpc_client->Recv(buf, len, name);
  }
}

void PtreBroadcast(char* buf, size_t len, int root_rank, const string& name) {
  if (ptre_global.rank == root_rank) {
    //LOG(INFO) << "BCASTSEND " << name << ": var[0]=" << ((float*) buf)[0];
    for (int i = 0; i < ptre_global.size; i++) {
      if (i == root_rank) continue;
      PtreSend(i, buf, len, name);
    }
  } else {
    PtreRecv(root_rank, buf, len, name);
    //LOG(INFO) << "BCASTRECV " << name << ": var[0]=" << ((float*) buf)[0];
  }
}

void PtreBarrier() {
  int size = ptre_global.size;
  if (size == 1) return;
  int my_rank = ptre_global.rank;
  int mask = 0x1;
  while (mask < size) {
    int dst = (my_rank + mask) % size;
    PtreSend(dst, NULL, 0, "PtreBarrier");
    int src = (my_rank - mask + size) % size;
    PtreRecv(src, NULL, 0, "PtreBarrier");
    mask <<= 1;
  }
}

// TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
// TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
// TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
// TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
// Read indicator key
// Read tensor
// Read and check indicator key
//
// Aggregate

void EnqueuePullTask(PullTask* task) {
  auto q = GetQueue(task);
  q.lock();
  q.push(task);
}

// Delete a task object.
void DeletePullTask(PullTask* task) {
  const string var_name = task->var_name();
  auto table = GetVariablePullTaskTable(var_name);
  PTRE_VAR_LOCK(var_name);
  auto it = table.begin();
  while (it != table.end()) {
    if (*it == task) {
      it = table.erase(it);
    } else {
      it++;
    }
  }
  delete task;
  PTRE_VAR_UNLOCK(var_name);
}

void ProcessPullTaskCQ(PullTask* task) {
  switch (task->GetState()) {
    case PullTask::STATE_TENSOR_READ: {
      task->PostReadValidation();
    }
    case PullTask::STATE_KEY_READ: {
      task->PostReadTensor();
      break;
    }
    case PullTask::STATE_VALIDATION_READ: {
      if (task->IsTensorValid()) {
        EnqueueAggregation(task);
      } else {
        task->PostReadKey();
      }
      break;
    }
    default: {
      break;
    }
  }

  if (task->state() == PullTask::STATE_ABORTED) {
    auto job = reinterpret_cast<PullJob*>(task->job_handle());
    job->DeleteTask(task);
  }
}

int ProcessCQ(int dst, struct ibv_wc* wcs) {
  struct ibv_cq* cq = ptre_global.rdma_mgr->send_cq(dst);
  int ne = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
  if (ne <= 0) return 1;
  int ret;
  for (int i = 0; i < ne; i++) {
    PullTask* task = reinterpret_cast<PullTask*>(wcs[i].wr_id);
    if (wcs[i].status == IBV_WC_SUCCESS) {
      ProcessPullTaskCQ(task);
    } else {
      LOG(ERROR) << "wc bad status = " << wcs[i].status;
      LOG(ERROR) << "PullTask Must be freed: " << (void*) task;
    }
  }
  return 0;
}


// Share task resource with Modelaverage OpKernel -> ClearPullTasks()
// NEVER SET STATE TO ABORTED
void ConcurrentAggThreadLoop() {
  auto&& q = ptre_global.agg_q;
  while (!ptre_global.is_shutdown) {
    PullTask* task;
    q.wait_and_pop(task);
    auto job = reinterpret_cast<PullJob*>(task->job_handle());
    auto rvar = ptre_global.cm->remote_variable(task->var_name());
    if (rvar) {
      if (task->state() == PullTask::STATE_VALID) {
        rvar->Aggregate(*task->tensor());
        job->DeleteTask(task);
      } else {
        int ret = task->PostReadKey();
        if (ret) {
          job->DeleteTask(task);
        }
      }
    } else {
      job->DeleteTask(task);
    }
  }
}

void load_grpc_hosts(const string& grpc_hosts_file) {
  std::string in_line;
  std::ifstream in(grpc_hosts_file);
  while (std::getline(in, in_line)) {
    if (in_line[0] == '#') continue;
    ptre_global.grpc_hosts.emplace_back(in_line);
  }
  in.close();
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
  ptre_global.rdma_mgr = new RdmaManager(size, rank);
  ptre_global.grpc_service.SetRdmaManager(ptre_global.rdma_mgr);
  for (int i = 0; i < ptre_global.size; i++) {
    ptre_global.qp_mus.push_back(new std::mutex());
  }

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
    ptre_global.rdma_mgr->set_remote_lid(i, remote_lid);
    ret = -1;
    uint32_t remote_qpn;
    uint32_t remote_psn;
    while (ret) {
      ret = client->GetQPAttr(&remote_qpn, &remote_psn);
    }
    ptre_global.rdma_mgr->ConnectQP(i, remote_qpn);
  }
  PtreBarrier();

  // Connectivity Check
  int ret;
  do {
    ret = ptre_global.rdma_mgr->ConnectivityCheck();
    PtreBarrier();
  } while (ret);

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

void RdmaSetRemoteAddress(int dst, BufType buf_type, const string& var_name) {
  GrpcClient* client;
  ptre_global.grpc_client_cache->GetClient(dst, &client);
  uint64_t remote_addr;
  uint32_t rkey;
  int ret;
  do {
    ret = client->GetRemoteAddress(buf_type, var_name, &remote_addr, &rkey);
  } while (ret && !ptre_global.is_shutdown);
  ptre_global.rdma_mgr->SetRemoteAddress(dst, buf_type, var_name,
      remote_addr, rkey);
}


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
    ptre_global.trainable_var_names = names_;
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.push_success_cnt[names_[i]] = 0;
      //ptre_global.push_var_mus.emplace(names_[i], std::mutex());
      ptre_global.push_var_mus[names_[i]];
    }

    //ptre_global.rcv_cnts.resize(ptre_global.local_step + 1);
    //ptre_global.rcv_cnts.back().resize(num_inputs);
    ptre_global.cm = new ConsensusManager(ptre_global.size, ptre_global.rank,
        inputs, names_);
    ptre_global.grpc_service.SetConsensusManager(ptre_global.cm);
    ptre_global.cm->InitPeerSelector(ptre_global.peer_selector,
        ptre_global.num_push);

    // Register MRs
    LOG(INFO) << "Register Memory Regions";
    ptre_global.rdma_mgr->InitMRs(ptre_global.cm->remote_variables());

    // Retrieve Remote Addresses
    LOG(INFO) << "Exchange Remote Addresses for RDMA Communication";
    for (int i = 0; i < ptre_global.size; i++) {
      for (int j = 0; j < num_inputs; j++) {
        RdmaSetRemoteAddress(i, BUF_TYPE_PULL_KEY, names_[j]);
        RdmaSetRemoteAddress(i, BUF_TYPE_PULL_TENSOR_A, names_[j]);
        RdmaSetRemoteAddress(i, BUF_TYPE_PULL_TENSOR_B, names_[j]);
      }
    }

    // Init Push Thread
    LOG(INFO) << "Starting Push Threads: num_threads=" << NUM_PUSH_THREADS;
    for (int i = 0; i < NUM_PUSH_THREADS; i++) {
      ptre_global.push_threads.emplace_back(
          std::thread(ConcurrentPushThreadLoop));
    }

    // Polling Thread
    LOG(INFO) << "Starting Send Polling Threads: num_threads=" << NUM_SEND_POLLING_THREADS;
    for (int i = 0; i < NUM_SEND_POLLING_THREADS; i++) {
      ptre_global.send_polling_threads.emplace_back(
          std::thread(SendCQProcessThreadLoop, i));
    }
    LOG(INFO) << "Starting Recv Polling Threads: num_threads=" << NUM_RECV_POLLING_THREADS;
    for (int i = 0; i < NUM_RECV_POLLING_THREADS; i++) {
      ptre_global.recv_polling_threads.emplace_back(
          std::thread(RecvCQProcessThreadLoop, i));
    }

    // Init Aggregation Thread
    LOG(INFO) << "Starting Aggregation Threads: num_threads=" << NUM_AGG_THREADS;
    //ptre_global.eigen_pool = new Eigen::ThreadPool(NUM_AGG_EIGEN_THREADS);
    ptre_global.eigen_pool = new Eigen::ThreadPool(NUM_AGG_THREADS);
    //Eigen::ThreadPoolDevice d(&pool, NUM_AGG_EIGEN_THREADS);
    for (int i = 0; i < NUM_AGG_THREADS; i++) {
      ptre_global.aggregation_threads.emplace_back(
          std::thread(AggregationThreadLoop, i));
    }

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

template <typename T>
struct CopyRemoteToVar<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstFlat remote) {
    auto bytes = sizeof(T) * var.size();
    memcpy(var.data(), remote.data(), bytes);
  }
};

}  // namespace functor

void ClearPullJobs() {
  std::lock_guard<std::mutex> guard(ptre_global.job_table_mu);
  auto&& table = ptre_global.pull_jobs;
  auto it = table.begin();
  while (it != table.end()) {
    if (it->second->NumLiveTasks() == 0) {
      it = table.erase(it);
    } else {
      it++;
    }
  }
}

void CreatePullJob(int step, int num_pull) {
  std::map<int, std::vector<string>> task_init_attr;
  std::map<bool> checker;
  for (int i = 0; i < num_pull; i++) {
    int dst;
    do {
      dst = ptre_global.cm->get_peer();
    } while (checker.find(dst) != checker.end());
    checker[dst] = 1;
    std::vector<string> vars_to_pull;
    task_init_attr[dst] = ptre_global.trainable_var_names;
  }
  PullJob* new_job = new PullJob(step, task_init_attr);
  ptre_global.job_table_mu.lock();
  ptre_global.pull_jobs[step] = new_job;
  ptre_global.job_table_mu.unlock();
}

void EnqueuePullTasks(const string& var_name, int num_pull) {
  // TODO: add tasks to the job.
  // TODO: post task
  int step = ptre_global.local_step;
  auto job = ptre_global.pull_jobs[step];
  std::vector<int> dsts;
  job->GetDstPeers(var_name, &dsts);
  for (auto dst : dsts) {
    PullTask* task = new PullTask(ptre_global.rdma_mgr, dst,
        ptre_global.cm->remote_variable(var_name), (void*) job);
    job->SetTask(dst, var_name, task);
    // TODO: check with CQ process thread
    int ret = task->PostReadKey();
    if (ret) {
      job->DeleteTask(dst, var_name);
    }
  }
}

void StopPullTasks(const string& var_name) {
  std::lock_guard<std::mutex> guard(ptre_global.job_table_mu);
  auto&& job_table = ptre_global.pull_jobs;
  for (auto&& it : job_table) {
    it.second->StopTasks(var_name);
  }
}

static Status ModelaverageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                  // var
  if (c->num_outputs() > 0) {
  :wa
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
    auto rvar = ptre_global.cm->remote_variable(var_name_);
    if (!rvar) return;

    StopPullTasks(var_name_);
    rvar->StopAggregation();
    int num_incomings = rvar->agg_count() - 1;
    if (num_incomings > 0) {
      rvar->Reduce();
      core::RefCountPtr<Var> ref;
      LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
      Tensor var = *ref->tensor();
      const Tensor remote = *rvar->tensor();
      const Device& d = ctx->template eigen_device<Device>();
      functor::CopyRemoteToVar<Device, T>()(d, var.flat<T>(), remote.flat<T>());
    }
    rvar->StartAggregation();
    EnqueuePullTasks(var_name_, ptre_global.num_push);

    ptre_global.agg_cnts[ptre_global.local_step][var_name_] = num_incomings;
  }

 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("ResourceModelaverage") \
                              .Device(DEVICE_##D)      \
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
    ptre_global.rdma_mgr->InitPush(i);
  }
  auto req = std::make_shared<PushRequest>(ptre_global.num_push,
      ptre_global.local_step, ptre_global.size,
      ptre_global.cm->variable_names());
  req_q.push(req);
  /*
  LOG(INFO) << "SEND COUNT TOTAL = " << ptre_global.send_cnt_total;
  ptre_global.send_cnt_total = 0;
  */
  req_mu.unlock();
}

void ClearTasks(const string& var_name) {
  //LOG(INFO) << "Clear Tasks: " << var_name;
  auto&& q = ptre_global.push_q;
  auto&& mu = ptre_global.push_q_mu;
  mu.lock();
  for (int i = 0; i < q.size(); i++) {
    auto task = q.front();
    q.pop();
    if (task->var_name().compare(var_name)) {  // they differ
      q.push(task);
    } else {
      CancelPushVar(task->dst(), task->var_name());
    }
  }
  mu.unlock();
  ptre_global.rpn_checker_mu.lock();
  auto it = ptre_global.rpn_checker.begin();
  while (it != ptre_global.rpn_checker.end()) {
    if (!it->second.compare(var_name)) {
      it = ptre_global.rpn_checker.erase(it);
    } else {
      it++;
    }
  }
  ptre_global.rpn_checker_mu.unlock();
}

void EnqueueTasks(const string& var_name, int num_push) {
  auto&& q = ptre_global.push_q;
  auto&& mu = ptre_global.push_q_mu;

  mu.lock();
  auto&& dsts = ptre_global.push_dsts;
  int curr_num_dsts = dsts.size();
  if (curr_num_dsts < num_push) {
    for (int i = 0; i < num_push - curr_num_dsts; i++) {
      for (int j = 0; j < ptre_global.size; j++) {
        int dst = ptre_global.cm->get_peer();
        auto search = std::find(dsts.begin(),
            dsts.end(), dst);
        if (search == dsts.end()) {
          dsts.push_back(dst);
          break;
        }
      }
    }
  }

  int num_actual_push = std::min((int) dsts.size(), num_push);
  for (int i = 0; i < num_actual_push; i++) {
    auto task = std::make_shared<PushTaskV2>(dsts[i], var_name);
    //LOG(INFO) << "Enqueued " << var_name << ", rank=" << dsts[i];
    q.push(task);
  }
  mu.unlock();
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
    std::lock_guard<std::mutex> guard(ptre_global.push_var_mus[var_name_]);

    auto pvar = ptre_global.rdma_mgr->push_variable(var_name_);
    if (!pvar) return;

    pvar->StopPush();
    ClearTasks(var_name_);

    /*
    ptre_global.push_op_mu.lock();
    ptre_global.push_op_cnt++;
    if (ptre_global.push_op_cnt == 1) {
      PtreClearQueueAndEnqueueRequest();
    } else if (ptre_global.push_op_cnt == ptre_global.num_trainable_variables) {
      ptre_global.push_op_cnt = 0;
    }
    ptre_global.push_op_mu.unlock();
    */

    Tensor var;
    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    var = *ref->tensor();
    const Device& d = ctx->template eigen_device<Device>();
    /*
    struct ibv_mr* mr = ptre_global.rdma_mgr->GetMR(ptre::BUF_TYPE_SEND_BUF,
        var_name_);
    if (!mr) {
      LOG(ERROR) << "buf not found: " << ptre::BUF_TYPE_SEND_BUF << ", " << var_name_;
      exit(1);
    }
    T* send_buf = (T*) mr->addr;
    */
    T* send_buf = (T*) pvar->data();
    typename TTypes<T>::Flat send_flat(send_buf, var.flat<T>().size());
    /*
    LOG(INFO) << "\n" << "COPYTOSENDBUF " << var_name_;
    */
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), send_flat);

    pvar->StartPush();
    //ptre_global.rdma_mgr->SetPushReady(var_name_);
    EnqueueTasks(var_name_, ptre_global.num_push);
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

REGISTER_OP("ResourceUpdatePullVariable")
  .Input("var: resource")
  .Attr("T: numbertype")
  .Attr("var_name: string");
template <typename Device, typename T>
class UpdatePullVariableOp : public OpKernel {
 public:
  explicit UpdatePullVariableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* ctx) {
    auto pvar = ptre_global.rdma_mgr->pull_variable(var_name_);
    if (!pvar) return;

    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    Tensor var = *ref->tensor();

    T* next_buf = (T*) pvar->next_data();
    typename TTypes<T>::Flat next_flat(next_buf, var.flat<T>().size());
    const Device& d = ctx->template eigen_device<Device>();
    pvar->SetNextKey(ptre_global.local_step);
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), next_flat);

    pvar->Switch();

    auto rvar = ptre_global.cm->remote_variable(var_name_);
    if (rvar) {
      rvar->Aggregate(pvar->curr_data());
    }
  }
 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("ResourceUpdatePullVariable")   \
                              .Device(DEVICE_##D)      \
                              .HostMemory("var")       \
                              .TypeConstraint<T>("T"), \
                          UpdatePullVariableOp<D##Device, T>);
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

using tensorflow::ptre_global;

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
  auto&& mu = ptre_global.push_q_mu;
  auto&& q = ptre_global.push_q;
  mu.lock();
  ptre_global.push_step_state = 1;
  ptre_global.push_dsts.clear();
  mu.unlock();
}

void ptre_unset_push() {
  auto&& mu = ptre_global.push_q_mu;
  mu.lock();
  ptre_global.push_step_state = 0;
  mu.unlock();
}

void ptre_finalize(unsigned int wait_time) {
  sleep(wait_time);
  /*
  for (auto it : ptre_global.push_success_cnt) {
    LOG(INFO) << it.first << ": push_success_cnt=" << it.second;
  }
  */
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

int rcv_cnt_last = 0;

void ptre_print_recv_count() {
  LOG(INFO) << "rcv_count(step=" << ptre_global.local_step << ") = "
      << ptre_global.rcv_cnt_total - rcv_cnt_last;
  rcv_cnt_last = ptre_global.rcv_cnt_total;
  LOG(INFO) << "agg_count(step=" << ptre_global.local_step << ") = "
      << ptre_global.agg_cnt_total;
}

void ptre_count_step() {
  //tensorflow::ptre_global.local_step++;
  tensorflow::ptre_global.virtual_step++;
  tensorflow::ptre_global.cm->count_virtual_step();
}

void ptre_set_local_step(int local_step) {
  using tensorflow::ptre_global;
  ptre_global.agg_cnt_total = 0;
  /*
  using std::string;
  string a;
  int sum = 0;
  for (int i = 0; i < tensorflow::ptre_global.num_trainable_variables; i++) {
    a.append(" " + std::to_string(tensorflow::ptre_global.rcv_cnts[tensorflow::ptre_global.local_step][i]));
    sum += tensorflow::ptre_global.rcv_cnts[tensorflow::ptre_global.local_step][i];
  }
  LOG(INFO) << "rcv_cnts =" << a;
  LOG(INFO) << "total = " << sum;
  tensorflow::ptre_global.rcv_cnts.resize(local_step + 1);
  tensorflow::ptre_global.rcv_cnts.back().resize(tensorflow::ptre_global.num_trainable_variables);
  */

  tensorflow::ptre_global.local_step = local_step;
  tensorflow::ptre_global.cm->set_local_step(local_step);
  ptre_global.agg_cnts.resize(local_step + 1);
}

void ptre_barrier() {
  tensorflow::PtreBarrier();
}

void ptre_print_counter_summary_epoch() {
  int begin = ptre_global.agg_cnts_last;
  int end = ptre_global.agg_cnts.size();
  int n = end - begin;
  float avg_bytes = 0;
  float avg_count = 0;
  std::stringstream ss;
  for (auto&& name : ptre_global.trainable_var_names) {
    int sum = 0;
    std::vector<int> l;
    for (int i = begin; i < end; i++) {
      sum += ptre_global.agg_cnts[i][name];
      l.push_back(ptre_global.agg_cnts[i][name]);
    }
    std::sort(l.begin(), l.end(), std::greater<int>());
    float avg = (float) sum / (n - 1);
    avg_count += avg;
    ss << "(" << avg << ", " << l[(n - 1) / 2] << ") ";
    avg_bytes += avg * ptre_global.cm->remote_variable(name)->rcv_length();
  }
  ptre_global.agg_cnts_last = end;
  LOG(INFO) << ss.str() << "\nAVG COUNT=" << avg_count << ", AVG BYTES=" << (int) avg_bytes << ", n=" << n;
}

void ptre_print_counter_summary() {
  int n = ptre_global.agg_cnts.size();
  float avg_bytes = 0;
  float avg_count = 0;
  for (auto&& name : ptre_global.trainable_var_names) {
    int sum = 0;
    std::vector<int> l;
    for (int i = 1; i < n; i++) {
      sum += ptre_global.agg_cnts[i][name];
      l.push_back(ptre_global.agg_cnts[i][name]);
    }
    std::sort(l.begin(), l.end(), std::greater<int>());
    float avg = (float) sum / (n - 1);
    avg_count += avg;
    LOG(INFO) << name << ": avg=" << avg
        << ", mid=" << l[(n - 1) / 2];
    avg_bytes += avg * ptre_global.cm->remote_variable(name)->rcv_length();
  }
  LOG(INFO) << "AVG COUNT=" << avg_count;
  LOG(INFO) << "AVG BYTES=" << (int) avg_bytes;
}

}
