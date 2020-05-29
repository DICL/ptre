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
//#include <shared_mutex>
//#include "ptre/lib/shared_mutex.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/grpc/grpc_client_cache.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/rdma.h"
#include "ptre/communication/rdma/rdma_manager.h"
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
    int ibv_call_ret = X; \
    if (ibv_call_ret) { \
      ptre_global.rdma_manager->RecoverQP(D); \
    } else { \
      break; \
    } \
  }

#define NUM_PUSH_THREADS 8
#define NUM_POLLING_THREADS 8
#define NUM_AGG_THREADS 8
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
    if (push_thread.joinable()) {
      push_thread.join();
    }
    for (auto& t : polling_threads) {
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
    if (aggregation_thread.joinable()) {
      aggregation_thread.join();
    }
    for (auto& t : receive_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    if (receive_thread.joinable()) {
      receive_thread.join();
    }

    //if (qp_recover_thread.joinable()) {
    //  qp_recover_thread.join();
    //}
    /*
    if (rdma_manager != nullptr) {
      delete rdma_manager;
    }
    */
  }

  ConsensusManager* cm = nullptr;
  RdmaManager* rdma_manager = nullptr;
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
  std::thread push_thread;
  std::vector<std::thread> push_threads;
  std::vector<std::thread> polling_threads;
  std::thread aggregation_thread;
  std::vector<std::thread> aggregation_threads;
  std::thread receive_thread;
  std::vector<std::thread> receive_threads;

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

  std::map<string, int> push_success_cnt;
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

/*
void PushThreadLoop(int tid, int num_t) {
  int begin = ptre_global.size * tid / num_t;
  int end = ptre_global.size * (tid + 1) / num_t;

  for (int i = begin; i < end; i++) {
  }
}
*/

/*
int PollSendCQ(int dst) {
}

void AThreadLoop() {
  while (!ptre_global.is_shutdown) {
  }
}
*/
int ProcessRecvCQ(int dst, struct ibv_wc* wcs) {
  struct ibv_cq* cq = ptre_global.rdma_manager->recv_cq(dst);
  //struct ibv_wc wcs[ptre::MAX_CQE_DEFAULT];
  int ne = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
  if (ne <= 0) return 1;
  for (int i = 0; i < ne; i++) {
    RecvTask* task = reinterpret_cast<RecvTask*>(wcs[i].wr_id);
    if (wcs[i].status == IBV_WC_SUCCESS) {
      //LOG(INFO) << "Push from rank=" << dst;
      uint32_t idx = ntohl(wcs[i].imm_data);
      if (idx >= 0) {
        LOG(INFO) << "Push from rank=" << dst << ", var_name=" << idx;
        auto rvar = ptre_global.cm->remote_variable(idx);
        if (rvar) {
          rvar->NewIncoming(i);
        }
      }
    } else {
      ptre_global.rdma_manager->RecoverQP(dst);
    }
    IBV_CALL(dst, task->PostRecv());
  }
  return 0;
}

int ProcessSendCQ(int dst, struct ibv_wc* wcs) {
  struct ibv_cq* cq = ptre_global.rdma_manager->send_cq(dst);
  int ne = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
  if (ne <= 0) return 1;
  int ret;
  for (int i = 0; i < ne; i++) {
    RPNTask* task = reinterpret_cast<RPNTask*>(wcs[i].wr_id);
    if (wcs[i].status == IBV_WC_SUCCESS) {
      if (task->state() == RPNTask::STATE_WRITE) {
        delete task;
        continue;
      } else if (task->state() == RPNTask::STATE_READ) {
        if (task->permit() == ptre_global.rank) {
          do {
            ret = task->PostWrite();
            if (ret) {
              ptre_global.rdma_manager->RecoverQP(dst);
            }
          } while (ret && !ptre_global.is_shutdown);
        } else {
          do {
            ret = task->PostRead();
            if (ret) {
              ptre_global.rdma_manager->RecoverQP(dst);
            }
          } while (ret && !ptre_global.is_shutdown);
        }
      } else {
        LOG(ERROR) << "Unknown task state: " << task->state();
      }
    } else {
      //ptre_global.rdma_manager->RecoverQP(dst);
      if (task->state() == RPNTask::STATE_READ) {
        do {
          ret = task->PostRead();
          if (ret) {
            ptre_global.rdma_manager->RecoverQP(dst);
          }
        } while (ret && !ptre_global.is_shutdown);
      } else if (task->state() == RPNTask::STATE_WRITE) {
        do {
          ret = task->PostWrite();
          if (ret) {
            ptre_global.rdma_manager->RecoverQP(dst);
          }
        } while (ret && !ptre_global.is_shutdown);
      } else {
        LOG(ERROR) << "Unknown task state: " << task->state();
      }
    }
  }
  return 0;
}

void CQProcessThreadLoop(int tid, int num_t) {
  int begin = ptre_global.size * tid / num_t;
  int end = ptre_global.size * (tid + 1) / num_t;
  //const size_t num_wcs = ptre::MAX_CQE_DEFAULT;
  struct ibv_wc wcs[4096];
  int ret;
  while (!ptre_global.is_shutdown) {
    for (int i = begin; i < end; i++) {
      ProcessSendCQ(i, wcs);
      ProcessRecvCQ(i, wcs);
    }
  }
}

void ConcurrentPushThreadLoop() {
  auto&& q = ptre_global.push_q;
  auto&& mu = ptre_global.push_q_mu;
  while (!ptre_global.is_shutdown) {
    mu.lock();
    if (q.size() == 0) {
      mu.unlock();
      continue;
    }
    auto task = q.front();
    q.pop();
    mu.unlock();

    //LOG(INFO) << "Processing task " << task->var_name() << ", dst=" << task->dst();
    auto&& var_mu = ptre_global.push_var_mus[task->var_name()];
    //var_mu.lock_shared();
    var_mu.lock();
    auto pvar = ptre_global.rdma_manager->push_variable(task->var_name());
    if (!pvar) {
      //var_mu.unlock_shared();
      var_mu.unlock();
      continue;
    }

    if(!pvar->GetState()) {
      mu.lock();
      q.push(task);
      mu.unlock();
      //var_mu.unlock_shared();
      var_mu.unlock();
      continue;
    }

    if (!task->IsAttemptDone()) {
      GrpcClient* client;
      ptre_global.grpc_client_cache->GetClient(task->dst(), &client);
      int ret = client->AttemptPushVar(task->var_name());
      if (ret) {
        //var_mu.unlock_shared();
        var_mu.unlock();
        continue;
      }
      //LOG(INFO) << "Done AttemptPushVar: " << task->var_name() << ", dst=" << task->dst();
      task->SetAttemptDone();
    }

    //LOG(INFO) << "PushAndNotify " << task->dst() << ", " << task->var_name();
    //LOG(INFO) << "PushAndNotify";
    /*
    int ret = ptre_global.rdma_manager->PushAndNotify(task->dst(),
        task->var_name());
    */

    RPNTask* rdma_task = new RPNTask(ptre_global.rdma_manager, task->dst(),
        task->var_name());
    IBV_CALL(task->dst(), rdma_task->PostRead());
    //LOG(INFO) << "Done PostRead dst=" << task->dst() << ", var_name=" << task->var_name();

    //LOG(INFO) << "Done PushAndNotify";
#if 0
    if (ret) {
      if (ret < 0) {
      }
      mu.lock();
      q.push(task);
      mu.unlock();
    }
#endif
    //var_mu.unlock_shared();
    var_mu.unlock();
  }
}

void PostReceiveWRs() {
  int ret;
  for (int i = 0; i < ptre_global.size; i++) {
    RecvTask* task = new RecvTask(ptre_global.rdma_manager, i);
    do {
      ret = task->PostRecv();
    } while (ret && !ptre_global.is_shutdown);
  }
}

#if 0
void ReceiveThreadLoop(int tid, int num_t) {
  std::map<int, bool> checker;
  int begin = ptre_global.size * tid / num_t;
  int end = ptre_global.size * (tid + 1) / num_t;
  //LOG(INFO) << "RECV_THREAD[TID:" << tid << "] begin=" << begin << ", end=" << end;
  for (int i = begin; i < end; i++) {
    //LOG(INFO) << "ReceivePushAndNotify";
    int ret = ptre_global.rdma_manager->ReceivePushNotify(i);
    //LOG(INFO) << "Done ReceivePushAndNotify";
    if (ret) {
      checker[i] = false;
    } else {
      checker[i] = true;
    }
  }
  while (!ptre_global.is_shutdown) {
    for (int i = begin; i < end; i++) {
      if (checker[i]) {
        int ret = ptre_global.rdma_manager->PollPushNotify(i);
        if (ret >= 0) {
          int idx = ret;
          auto rvar = ptre_global.cm->remote_variable(idx);
          if (rvar) {
            rvar->NewIncoming(i);
          }
          int ret_inner = ptre_global.rdma_manager->ReceivePushNotify(i);
          if (ret_inner) {
            checker[i] = false;
          } else {
            checker[i] = true;
          }
        } else if (ret == -1) {
            checker[i] = false;
        }
      } else {
        //LOG(INFO) << "ReceivePushAndNotify";
        int ret = ptre_global.rdma_manager->ReceivePushNotify(i);
        //LOG(INFO) << "Done ReceivePushAndNotify";
        if (ret) {
          checker[i] = false;
        } else {
          checker[i] = true;
        }
      }
    }
  }
}
#endif

void AggregationThreadLoop(int tid, int num_t) {
  int begin = ptre_global.num_trainable_variables * tid / num_t;
  int end = ptre_global.num_trainable_variables * (tid + 1) / num_t;
  //LOG(INFO) << "AGG_THREAD[TID:" << tid << "] begin=" << begin << ", end=" << end;
  while (!ptre_global.is_shutdown) {
    for (int i = begin; i < end; i++) {
      auto rvar = ptre_global.cm->remote_variable(i);
      if (rvar) {
        rvar->Aggregate();
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
  /*
  for (int i = 0; i < ptre_global.size; i++) {
    std::cout << ptre_global.grpc_hosts[i] << std::endl;
  }
  */
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
  PtreBarrier();

  // Connectivity Check
  int ret;
  do {
    ret = ptre_global.rdma_manager->ConnectivityCheck();
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

    //ptre_global.qp_recover_thread = std::thread(QPRecoverThreadLoop);

    // Init Aggregation Thread
    LOG(INFO) << "Starting Aggregation Threads: num_threads=" << NUM_AGG_THREADS;
    //ptre_global.aggregation_thread = std::thread(AggregationThreadLoop);
    for (int i = 0; i < NUM_AGG_THREADS; i++) {
      ptre_global.aggregation_threads.emplace_back(
          std::thread(AggregationThreadLoop, i, NUM_AGG_THREADS));
    }

    // Post Recv
    PostReceiveWRs();
#if 0
    // Init Receive Thread
    LOG(INFO) << "Starting Receive Threads: num_threads=" << NUM_RECV_THREADS;
    //ptre_global.receive_thread = std::thread(ReceiveThreadLoop);
    for (int i = 0; i < NUM_RECV_THREADS; i++) {
      ptre_global.receive_threads.emplace_back(
          std::thread(ReceiveThreadLoop, i, NUM_RECV_THREADS));
    }
#endif

    // Init Push Thread
    LOG(INFO) << "Starting Push Threads: num_threads=" << NUM_PUSH_THREADS;
    //ptre_global.push_thread = std::thread(PushThreadLoop);
    for (int i = 0; i < NUM_PUSH_THREADS; i++) {
      ptre_global.push_threads.emplace_back(
          std::thread(ConcurrentPushThreadLoop));
    }

    // Polling Thread
    LOG(INFO) << "Starting Polling Threads: num_threads=" << NUM_POLLING_THREADS;
    for (int i = 0; i < NUM_POLLING_THREADS; i++) {
      ptre_global.polling_threads.emplace_back(
          std::thread(CQProcessThreadLoop, i, NUM_POLLING_THREADS));
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
      //std::this_thread::sleep_for(std::chrono::seconds(1));
      //LOG(INFO) << "StopRecv";
      rvar->StopRecv();
      //LOG(INFO) << "Done StopRecv";
      /*
      int permit = rvar->permit();
      ptre_global.rdma_manager->RdmaWrite(ptre_global.rank,
          BUF_TYPE_PUSH_PERMIT, var_name_, (void*) &permit, sizeof(int));
      */
      num_incomings = rvar->agg_count();
      ret = rvar->tensor();
    }
    //ptre_global.rcv_cnts[ptre_global.local_step][ptre_global.cm->var_name_to_index(var_name_)]++;
    //ptre_global.agg_cnt_total.fetch_add(num_incomings);
    ptre_global.agg_cnts[ptre_global.local_step][var_name_] = num_incomings;
    if (num_incomings == 0) {
      do_reduce = false;
    }
    if (do_reduce) {
      const Device& d = ctx->template eigen_device<Device>();
      const Tensor other(*ret);
#if 1
      // No Step Control
      Tensor m_(DataTypeToEnum<T>::v(), TensorShape({ }));
      m_.flat<T>()(0) = T(num_incomings + 1);
      const Tensor m(m_);
      /*
      LOG(INFO) << "\n" << "MAVG " << var_name_;
      */
      //usleep(100);
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
      //LOG(INFO) << "StartRecv";
      rvar->StartRecv();
      //LOG(INFO) << "Done StartRecv";
      /*
      int permit = rvar->permit();
      ptre_global.rdma_manager->RdmaWrite(ptre_global.rank,
          BUF_TYPE_PUSH_PERMIT, var_name_, (void*) &permit, sizeof(int));
      */
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
    }
  }
  mu.unlock();
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

    auto pvar = ptre_global.rdma_manager->push_variable(var_name_);
    if (!pvar) return;

    ptre_global.push_var_mus[var_name_].lock();
    pvar->StopPush();
    ptre_global.push_var_mus[var_name_].unlock();
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
    struct ibv_mr* mr = ptre_global.rdma_manager->GetMR(ptre::BUF_TYPE_SEND_BUF,
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

    EnqueueTasks(var_name_, ptre_global.num_push);
    ptre_global.push_var_mus[var_name_].lock();
    pvar->StartPush();
    //ptre_global.rdma_manager->SetPushReady(var_name_);
    ptre_global.push_var_mus[var_name_].unlock();
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

void ptre_print_counter_summary() {
  int n = ptre_global.agg_cnts.size();
  float avg_bytes = 0;
  for (auto&& name : ptre_global.trainable_var_names) {
    int sum = 0;
    std::vector<int> l;
    for (int i = 1; i < n; i++) {
      sum += ptre_global.agg_cnts[i][name];
      l.push_back(ptre_global.agg_cnts[i][name]);
    }
    std::sort(l.begin(), l.end(), std::greater<int>());
    float avg = (float) sum / (n - 1);
    LOG(INFO) << name << ": avg=" << avg
        << ", mid=" << l[(n - 1) / 2];
    avg_bytes += avg * ptre_global.cm->remote_variable(name)->rcv_length();
  }
  LOG(INFO) << "AVG BYTES=" << (int) avg_bytes;
}

}
