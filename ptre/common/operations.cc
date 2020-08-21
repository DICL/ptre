#include "ptre/common/operations.h"

#include <chrono>
#include <deque>
#include <fstream>
//#include <future>

#include "ptre/common/buffer_table.h"
#include "ptre/common/logging.h"
#include "ptre/common/communication/tcp/tcp_grpc_client.h"
#include "ptre/common/rdma/rdma_controller.h"
#include "ptre/common/utils/host_file_parser.h"

#include "tensorflow/core/common_runtime/device.h"

#include <arpa/inet.h>

// up to 346ms/step 8 nodes
//#define THREAD_SLEEP_DURATION std::chrono::nanoseconds(100)
// up to 233ms/step 8 nodes
//#define THREAD_SLEEP_DURATION std::chrono::microseconds(1)
// up to 254ms/step 8 nodes
//#define THREAD_SLEEP_DURATION std::chrono::microseconds(10)
// 230ms/step
#define THREAD_SLEEP_DURATION std::chrono::microseconds(50)
//#define THREAD_SLEEP_DURATION std::chrono::microseconds(100)
// 305ms/step
//#define THREAD_SLEEP_DURATION std::chrono::milliseconds(1)

namespace ptre {
namespace common {

namespace {

PtreGlobal ptre_global;

}  // namespace

#if 0
template<typename T>
inline bool IsFutureReady(const std::future<T>& f) {
  return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}
#endif

// Parse host information
//  A host file is structured as:
//
//   hostname_0:port_0_0,port_0_1, ...
//   hostname_1:port_1_0, ...
//   ...
//   hostname_n:port_n_0, ...
void load_grpc_hosts(const string& grpc_hosts_file) {
  HostFileParser p(grpc_hosts_file);
  p.Parse(ptre_global.size);
  ptre_global.nodes = p.nodes();
  ptre_global.workers = p.workers();
  for (auto& worker : ptre_global.workers) {
    ptre_global.grpc_hosts.push_back(worker.grpc_host);
  }
  ptre_global.this_worker = ptre_global.workers[ptre_global.rank];
}

void PrintDebugMessageTable() {
  std::stringstream ss;
  ss << __FUNCTION__;
  auto& table = ptre_global.message_table;
  for (auto& iter1 : table) {
    ss << std::endl;
    auto& tensor_name = iter1.first;
    auto& row = iter1.second;
    ss << tensor_name << ":";
    for (auto& iter2 : row) {
      ss << " " << iter2.first;
    }
  }
  LOG(INFO) << ss.str();
}

void PrintDebugResponseList(ResponseList& response_list) {
  std::stringstream ss;
  ss << __FUNCTION__;
  for (auto& res : response_list.responses()) {
    ss << std::endl;
    ss << "type: " << res.tensor_type();
    for (auto& name: res.tensor_names()) {
      ss << "\n" << name;
    }
  }
  DVLOG(0) << ss.str();
}

void RunGrpcServer() {
  try {

  // Rdma Service
  auto&& service = ptre_global.grpc_service;
  service.SetBarrierVariable(&ptre_global.barrier_variable);
  string server_address = "0.0.0.0:"
      + std::to_string(ptre_global.this_worker.port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Tcp Service
  builder.RegisterService(&ptre_global.tcp_grpc_service);

  builder.SetMaxMessageSize(1 * 1024 * 1024 * 1024);
  ptre_global.grpc_server = builder.BuildAndStart();
  LOG(INFO) << "Grpc server listening on " << server_address;
  ptre_global.grpc_server->Wait();

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

void BackgroundThreadLoop(PtreGlobal& state);

void BackgroundThreadLoopModelaverage();

//void BackgroundThreadLoopPull();

void BackgroundMemcpyThread();

void PushThread();

void PollingThread();

void PollingThreadPerQP(int dst);

void PollingRecvThread();

void PostRecvTensorIdNumber(const int rank);

void EnqueueAvgThread();

void AvgThread();

void InitComm(int size, int rank, const string& grpc_hosts_file) {
  ptre_global.size = size;
  ptre_global.rank = rank;
  ptre_global.shutdown = false;

  // Init BufferTable
  try {
  ptre_global.buf_table = std::make_shared<BufferTable>();
  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
  ptre_global.grpc_service.SetBufferTable(ptre_global.buf_table);

  // Init Grpc Service
  load_grpc_hosts(grpc_hosts_file);

  if (size > ptre_global.grpc_hosts.size()) {
    LOG(ERROR) << "NOT ENOUGH HOSTS in the hostfile";
    exit(1);
  }
  ptre_global.grpc_client_cache =
    std::make_shared<GrpcClientCache<GrpcClient>>(rank, ptre_global.grpc_hosts);
  ptre_global.tcp_grpc_client_cache =
    std::make_shared<GrpcClientCache<TcpGrpcClient>>(rank,
                                                     ptre_global.grpc_hosts);
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);

  // Init RdmaMgr
  DVLOG(0) << "Init Rdma Manager";
  ptre_global.rdma_mgr = new RdmaMgr(size, rank);
  ptre_global.grpc_service.SetRdmaMgr(ptre_global.rdma_mgr);
  //for (int i = 0; i < ptre_global.size; i++) {
  //  ptre_global.qp_mus.push_back(new std::mutex());
  //}

  // Connect Queue Pairs
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
  LOG(INFO) << "Connected Queue Pairs";

  // Connectivity Check
  int ret;
  do {
    ret = ptre_global.rdma_mgr->ConnectivityCheck();
    PtreBarrier();
  } while (ret);

  // Init channels
  // TODO: Do it in a more sophisticated way.
  for (int i = 0; i < ptre_size(); i++) {
    auto channel = ptre_global.rdma_mgr->GetChannel(i);
  }

  //for (int i = 0; i < ptre_size(); i++) {
  //  ptre_global.polling_threads.emplace_back(
  //      std::thread(PollingThreadPerQP, i));
  //}
  ptre_global.polling_thread = std::thread(PollingThread);
  //ptre_global.polling_recv_thread = std::thread(PollingRecvThread);
  //ptre_global.polling_threads.emplace_back(std::thread(PollingThreadLoop));
  LOG(INFO) << "Launched Polling Thread";

  ptre_global.rdma_ctx = new RdmaContext(ptre_global.rdma_mgr);

  // Background thread for Allreduce requests
  //ptre_global.background_thread = std::thread(
  //    BackgroundThreadLoop, std::ref(ptre_global));

  // Background thread for Modelaverage requests
  //ptre_global.background_thread_modelaverage = std::thread(
  //    BackgroundThreadLoopModelaverage);

  // Memcpy Thread
  ptre_global.memcpy_thread = std::thread(BackgroundMemcpyThread);

  // RDMA push thread
  ptre_global.push_thread = std::thread(PushThread);

  // Enqueue Avg Thread
  //ptre_global.enq_avg_thread = std::thread(EnqueueAvgThread);

  // Avg Thread
  //for (int i = 0; i < 1; i++) {
  //  ptre_global.avg_threads.emplace_back(std::thread(AvgThread));
  //}
  ptre_global.avg_thread = std::thread(AvgThread);

  // Background thread for Pull requests
  //ptre_global.background_thread_pull = std::thread(
  //    BackgroundThreadLoopPull);
  //LOG(INFO) << "Launched Request Processing Threads";

  DVLOG(0) << "[1/2] Done InitComm";
}

void PostRecvTensorIdNumber(const int rank) {
  RdmaRecvEntry* entry = new RdmaRecvEntry();
  entry->rank = rank;
  entry->channel = ptre_global.rdma_mgr->GetChannel(entry->rank);
  auto post_result = PostRecvWithImm(entry);
}


void ShutdownGrpcServer() {
  if (ptre_global.grpc_server != nullptr) {
    ptre_global.grpc_server->Shutdown();
  }
}

PtreGlobal& PtreGlobalState() {
  return ptre_global;
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
    if (ret) exit(1);
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

void CreatePullJob(int step, int num_pull) {
  //auto last_time = std::chrono::system_clock::now();
  std::map<int, std::vector<string>> task_init_attr;
  std::map<int, bool> checker;
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
  //auto curr_time = std::chrono::system_clock::now();
  //std::chrono::duration<double> since_last = curr_time - last_time;
  //LOG(INFO) << __FUNCTION__ << ": " << since_last.count() / 1000 << "msec";
}

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

void EnqueuePullTasks(const string& var_name, int num_pull) {
  // TODO: add tasks to the job.
  // TODO: post task
//auto last_time = std::chrono::system_clock::now();
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
LOG(ERROR) << "Failed to PostReadKey()";  // DEBUG
      //job->DeleteTask(dst, var_name);
      job->DeleteTask(task);
    }
  }
//auto curr_time = std::chrono::system_clock::now();
//std::chrono::duration<double> since_last = curr_time - last_time;
//LOG(INFO) << __FUNCTION__ << ": " << since_last.count() / 1000 << "msec";
}

void StopPullTasks(const string& var_name) {
  //auto last_time = std::chrono::system_clock::now();
  std::lock_guard<std::mutex> guard(ptre_global.job_table_mu);
  auto&& job_table = ptre_global.pull_jobs;
  for (auto&& it : job_table) {
    it.second->StopTasks(var_name);
  }
  //auto curr_time = std::chrono::system_clock::now();
  //std::chrono::duration<double> since_last = curr_time - last_time;
  //LOG(INFO) << __FUNCTION__ << ": " << since_last.count() / 1000 << "msec";
}

void EnqueueAggregation(PullTask* task) {
  std::lock_guard<std::mutex> guard(ptre_global.agg_q_mu);
  ptre_global.agg_q.push(task);
}

void ProcessPullTaskCQ(PullTask* task) {
  int ret;
  auto job = reinterpret_cast<PullJob*>(task->job_handle());
  switch (task->GetState()) {
    case PullTask::STATE_TENSOR_READ: {
      ret = task->PostReadValidation();
      if (ret) {
        job->DeleteTask(task);
      }
      break;
    }
    case PullTask::STATE_KEY_READ: {
      ret = task->PostReadTensor();
      if (ret) {
        job->DeleteTask(task);
      }
      break;
    }
    case PullTask::STATE_VALIDATION_READ: {
      bool is_valid = false;
      ret = task->IsTensorValid(&is_valid);
      if (ret || !is_valid) {
        job->DeleteTask(task);
      } else {
        EnqueueAggregation(task);
      }
      break;
    }
    case PullTask::STATE_STOPPED: {
      job->DeleteTask(task);
      break;
    }
    default: {
      job->DeleteTask(task);
      break;
    }
  }

#if 0  // DEBUG
  if (task->state() == PullTask::STATE_ABORTED) {
    auto job = reinterpret_cast<PullJob*>(task->job_handle());
    job->DeleteTask(task);
  }
#endif  // DEBUG
}

int ProcessCQ(int dst, struct ibv_wc* wcs) {
  struct ibv_cq* cq = ptre_global.rdma_mgr->send_cq(dst);
  int ne = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
  if (ne <= 0) return 1;
  int ret;
  std::vector<PullTask*> bad_tasks;
  for (int i = 0; i < ne; i++) {
    PullTask* task = reinterpret_cast<PullTask*>(wcs[i].wr_id);
    if (wcs[i].status == IBV_WC_SUCCESS) {
      ProcessPullTaskCQ(task);
    } else {
      bad_tasks.push_back(task);
    }
  }

  if (bad_tasks.size() > 0) {
    auto channel = ptre_global.rdma_mgr->GetChannel(dst);
    LOG(ERROR) << "Recovering RdmaChannel for rank=" << dst;
    if (channel->Recover()) {
      LOG(ERROR) << "Failed to Recover RdmaChannel for rank=" << dst << ", Terminating.";
      exit(1);
    }
    for (auto&& task : bad_tasks) {
      /*
      int ret = task
      LOG(ERROR) << "wc bad status = " << wcs[i].status;
      LOG(ERROR) << task->state() << ", " << task->var_name();
      */
      auto job = reinterpret_cast<PullJob*>(task->job_handle());
      job->DeleteTask(task);
      //LOG(ERROR) << "PullTask Must be freed: " << (void*) task;
    }
  }
  return 0;
}

#if 0
void PollingThreadLoop(int tid) {
  int n = NUM_POLLING_THREADS;
  int begin = tid * ptre_global.size / n;
  int end = (tid + 1) * ptre_global.size / n;
  struct ibv_wc wcs[MAX_CQE_DEFAULT];
  while (!ptre_global.shutdown) {
    for (int i = begin; i < end; i++) {
      ProcessCQ(i, wcs);
    }
  }
}
#endif

// Share task resource with Modelaverage OpKernel -> ClearPullTasks()
// NEVER SET STATE TO ABORTED
void ConcurrentAggregationThreadLoop() {
  auto&& q = ptre_global.agg_q;
  auto&& mu = ptre_global.agg_q_mu;
  while (!ptre_global.shutdown) {
    PullTask* task = NULL;
    mu.lock();
    if (q.size() > 0) {
      task = q.front();
      q.pop();
    }
    mu.unlock();
    if (!task) continue;

    auto job = reinterpret_cast<PullJob*>(task->job_handle());
    auto rvar = ptre_global.cm->remote_variable(task->var_name());
    if (rvar) {
      if (task->state() == PullTask::STATE_VALID) {
        if (ptre_global.last_key[task->dst()][task->var_name()]
            < task->curr_key()
            && task->curr_key() >= ptre_global.local_step) {
          Eigen::ThreadPoolDevice d(ptre_global.agg_eigen_pool,
              NUM_AGG_EIGEN_THREADS);
          rvar->Aggregate(*task->tensor(), d);
          ptre_global.last_key[task->dst()][task->var_name()] =
              task->curr_key();
          ptre_global.peer_agg_cnt[task->dst()][task->var_name()]++;
          job->DeleteTask(task);
        } else {
#if 0
          PullTask* new_task = new PullTask(ptre_global.rdma_mgr,
              task->dst(), ptre_global.cm->remote_variable(task->var_name()),
              (void*) job);
          job->SetTask(task->dst(), task->var_name(), new_task);
          int ret = new_task->PostReadKey();
          if (ret) {
            job->DeleteTask(new_task);
          }
#else
          job->DeleteTask(task);
#endif
        }
      } else if (task->state() == PullTask::STATE_INVALID) {
#if 0
        PullTask* new_task = new PullTask(ptre_global.rdma_mgr,
            task->dst(), ptre_global.cm->remote_variable(task->var_name()),
            (void*) job);
        job->SetTask(task->dst(), task->var_name(), new_task);
        int ret = new_task->PostReadKey();
        if (ret) {
          job->DeleteTask(new_task);
        }
#else
        job->DeleteTask(task);
#endif
      } else {
#if 0
        int ret = task->PostReadKey();
        if (ret) {
          job->DeleteTask(task);
        }
#else
        job->DeleteTask(task);
#endif
      }
    } else {
      LOG(ERROR) << "Unknown var_name=" << task->var_name();
      job->DeleteTask(task);
    }
  }
}

void RdmaSetRemoteAddress(int dst, BufType buf_type, const string& var_name) {
  GrpcClient* client;
  ptre_global.grpc_client_cache->GetClient(dst, &client);
  uint64_t remote_addr;
  uint32_t rkey;
  int ret;
  do {
    ret = client->GetRemoteAddress(buf_type, var_name, &remote_addr, &rkey);
  } while (ret && !ptre_global.shutdown);
  ptre_global.rdma_mgr->SetRemoteAddress(dst, buf_type, var_name,
      remote_addr, rkey);
}

void RegisterTrainableVariables(OpContext* context,
                                const std::vector<string>& names_) {
  int num_inputs = context->num_inputs();
  std::vector<const Tensor*> inputs;
  for (int i = 0; i < num_inputs; i++) {
    const Tensor& input = context->input(i);
    inputs.push_back(&input);
  }
  LOG(INFO) << "Init Consensus Manager: num_trainable_vars=" << num_inputs
      << ", peer_selector=" << ptre_global.peer_selector;
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
  ptre_global.tcp_grpc_service.SetConsensusManager(ptre_global.cm);
  ptre_global.tcp_grpc_service.SetCommBufTables(&ptre_global.sendbuf_table,
      &ptre_global.recvbuf_table, &ptre_global.commbuf_table_mu);
  ptre_global.cm->InitPeerSelector(ptre_global.peer_selector,
      ptre_global.num_push);

  // Register MRs
  LOG(INFO) << "Register Memory Regions";
  ptre_global.rdma_mgr->InitMRs(ptre_global.cm->remote_variables());

  // Retrieve Remote Addresses
  LOG(INFO) << "Exchange Remote Addresses for RDMA Communication";
  for (int i = 0; i < ptre_global.size; i++) {
//if (i == ptre_global.rank) continue;  // DEBUG
    for (int j = 0; j < num_inputs; j++) {
      RdmaSetRemoteAddress(i, ptre::BUF_TYPE_PULL_KEY, names_[j]);
      RdmaSetRemoteAddress(i, ptre::BUF_TYPE_PULL_TENSOR_A, names_[j]);
      RdmaSetRemoteAddress(i, ptre::BUF_TYPE_PULL_TENSOR_B, names_[j]);
    }
  }

  for (int i = 0; i < ptre_global.size; i++) {
    for (int j = 0; j < num_inputs; j++) {
      ptre_global.last_key[i][names_[j]] = 0;
      ptre_global.peer_agg_cnt[i][names_[j]] = 0;
    }
  }

  // Init Polling Threads
#if 0
  LOG(INFO) << "Starting Polling Threads: num_threads=" << NUM_POLLING_THREADS;
  for (int i = 0; i < NUM_POLLING_THREADS; i++) {
    ptre_global.send_polling_threads.emplace_back(
        std::thread(PollingThreadLoop, i));
  }
#else
#endif
#if 0
  // Init Aggregation Thread
  LOG(INFO) << "Starting Aggregation Threads: num_threads="
      << NUM_AGG_THREADS;
  //ptre_global.eigen_pool = new Eigen::ThreadPool(NUM_AGG_THREADS);
  //Eigen::ThreadPoolDevice d(&pool, NUM_AGG_EIGEN_THREADS);
  for (int i = 0; i < NUM_AGG_THREADS; i++) {
    ptre_global.aggregation_threads.emplace_back(
        std::thread(ConcurrentAggregationThreadLoop));
  }
  int agg_pool_size = AGG_EIGEN_POOLSIZE;
  LOG(INFO) << "AGG_EIGEN_THREADS=" << NUM_AGG_EIGEN_THREADS
      << ", POOL_SIZE=" << agg_pool_size;
  ptre_global.agg_eigen_pool = new Eigen::ThreadPool(agg_pool_size);
#ifdef PTRE_CPU_REDUCE
  LOG(INFO) << "REDUCE_EIGEN_THREADS=" << NUM_REDUCE_EIGEN_THREADS;
  ptre_global.reduce_eigen_pool =
      new Eigen::ThreadPool(NUM_REDUCE_EIGEN_THREADS);
#endif
#endif

  LOG(INFO) << "[2/2] Done Registering Variables";
}


extern "C" {

int ptre_init(int size, int rank, const char* grpc_hosts_file,
              int selection_strategy, int num_push) {
  ptre_global.num_push = num_push;
  ptre_global.peer_selector = selection_strategy;
  InitComm(size, rank, grpc_hosts_file);
  //ptre_global.cm->InitPeerSelector(selection_strategy, num_push);
  //LOG(INFO) << "Peer selection strategy = " << selection_strategy;
}

void ptre_finalize(unsigned int wait_time) {
  sleep(wait_time);
  //ShutdownGrpcServer();
  ptre_global.shutdown = true;
}

int ptre_size() {
  return ptre_global.size;
}

int ptre_rank() {
  return ptre_global.rank;
}

void ptre_set_local_step(int local_step) {
  ptre_global.agg_cnt_total = 0;
  /*
  using std::string;
  string a;
  int sum = 0;
  for (int i = 0; i < ptre_global.num_trainable_variables; i++) {
    a.append(" " + std::to_string(ptre_global.rcv_cnts[ptre_global.local_step][i]));
    sum += ptre_global.rcv_cnts[ptre_global.local_step][i];
  }
  LOG(INFO) << "rcv_cnts =" << a;
  LOG(INFO) << "total = " << sum;
  ptre_global.rcv_cnts.resize(local_step + 1);
  ptre_global.rcv_cnts.back().resize(ptre_global.num_trainable_variables);
  */

  ptre_global.local_step = local_step;
  ptre_global.cm->set_local_step(local_step);
  ptre_global.agg_cnts.resize(local_step + 1);
}

void ptre_create_pull_job() {
  ClearPullJobs();
  CreatePullJob(ptre_global.local_step, ptre_global.num_push);
}

void ptre_barrier() {
  PtreBarrier();
}

void ptre_print_counter_summary_epoch() {
  int begin = ptre_global.agg_cnts_last;
  int end = ptre_global.agg_cnts.size();
  int n = end - begin;
  float avg_bytes = 0;
  float avg_count = 0;
  std::stringstream ss;
  ss << "\n===AGGREGATION COUNTER SUMMARY per epoch===\n";
  ss << "n=" << n << std::endl;
  if (n > 0) {
    for (auto&& name : ptre_global.trainable_var_names) {
      int sum = 0;
      std::vector<int> l;
      for (int i = begin; i < end; i++) {
        sum += ptre_global.agg_cnts[i][name];
        l.push_back(ptre_global.agg_cnts[i][name]);
      }
      std::sort(l.begin(), l.end(), std::greater<int>());
      float avg = (float) sum / n;
      avg_count += avg;
      ss << "(" << avg << ", " << l[n / 2] << ") ";
      avg_bytes += avg * ptre_global.cm->remote_variable(name)->tensor()
          ->AllocatedBytes();
    }
  }
  ptre_global.agg_cnts_last = end;
  ss << "\nAVG COUNT=" << avg_count << std::endl;
  ss << "AVG BYTES=" << (int) avg_bytes << std::endl;
  LOG(INFO) << ss.str();
}

void ptre_print_counter_summary() {
  int begin = 1;
  int end = ptre_global.agg_cnts.size();
  int n = end - begin;
  float avg_bytes = 0;
  float avg_count = 0;
  std::stringstream ss;
  ss << "\n===AGGREGATION COUNTER SUMMARY===\n";
  ss << "n=" << n << std::endl;
  if (n > 0) {
    for (auto&& name : ptre_global.trainable_var_names) {
      int sum = 0;
      std::vector<int> l;
      for (int i = begin; i < end; i++) {
        sum += ptre_global.agg_cnts[i][name];
        l.push_back(ptre_global.agg_cnts[i][name]);
      }
      std::sort(l.begin(), l.end(), std::greater<int>());
      float avg = (float) sum / n;
      avg_count += avg;
      ss << name << ": avg=" << avg
          << ", mid=" << l[n / 2] << std::endl;
      avg_bytes += avg * ptre_global.cm->remote_variable(name)->tensor()
          ->AllocatedBytes();
    }
  }
  ss << "AVG COUNT=" << avg_count << std::endl;
  ss << "AVG BYTES=" << (int) avg_bytes << std::endl;
  LOG(INFO) << ss.str();


  ss.str("\n");
  std::vector<int> cnts(ptre_size(), 0);
  int max = 0;
  for (auto&& it : ptre_global.peer_agg_cnt) {
    int dst = it.first;
    for (auto&& name_cnt : it.second) {
      cnts[dst] += name_cnt.second;
      if (cnts[dst] > max) max = cnts[dst];
    }
  }
  for (int i = 0; i < ptre_size(); i++) {
    cnts[i] = cnts[i] * 10 / max;
  }
  for (int row = 10; row > 0; row--) {
    for (int dst = 0; dst < ptre_size(); dst++) {
      if (cnts[dst] >= row) {
        ss << "O";
      } else {
        ss << " ";
      }
    }
    ss << "\n";
  }
  LOG(INFO) << ss.str();
}

}

// --------------------------------------------------------------------------

ReadyTensor* GetReadyTensor(const string& name) {
  return ptre_global.cm->ready_tensor(name);
}

// --------------------------------------------------------------------------

namespace {

#if 0
bool IncrementTensorCount(std::unique_ptr<MessageTable>& message_table,
                          Request msg, int comm_size) {
  auto name = msg.tensor_name();
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    message_table->emplace(name, std::vector<Request>({msg}));
    table_iter = message_table->find(name);
  } else {
    table_iter->second.push_back(msg);
  }

  int count = table_iter->second.size();
  return count == comm_size;
}
#endif

}

namespace {
  std::deque<RvarRequest> rvar_queue;
  std::mutex rvar_mu;
}

// TODO: float -> T
#if 0
void PerformGetRemoteVariable(RvarRequest req) {
  auto rvar = ptre_global.cm->remote_variable(req.var_name);
  rvar->StopAggregation();

  auto output_flat = req.output->flat<float>();
  output_flat = rvar->tensor()->flat<float>();
  auto num_agg_flat = req.num_agg->flat<int>();
  num_agg_flat(0) = rvar->agg_count();

  (req.callback)(Status::OK());
  rvar->StartAggregation();
  EnqueuePullTasks(req.var_name, ptre_global.num_push);
}

bool RunLoopOnceRvar(PtreGlobal& state) {
  std::deque<RvarRequest> reqs;
  rvar_queue_mu.lock();
  while (rvar_queue.empty()) {
    RvarRequest req = rvar_queue.front();
    rvar_queue.pop();
    reqs.push_back(std::move(req));
  }
  rvar_queue_mu.unlock();

  for (auto& req : reqs) {
    PerformGetRemoteVariable(req);
  }

  return !state.shutdown;
}

void BackgroundThreadLoopRvar(PtreGlobal& state) {
  while (RunLoopOnceRvar(state)) continue;
}
#endif

void MemcpyDeviceToHost(OpContext* context,
                        std::shared_ptr<Tensor> d,
                        std::shared_ptr<Tensor> h,
                        StatusCallback callback) {
  auto device_context = context->op_device_context();
  auto device = static_cast<::tensorflow::Device*>(context->device());
  device_context->CopyDeviceTensorToCPU(d.get(), "", device, h.get(), callback);
}

void MemcpyHostToDevice(OpContext* context,
                        std::shared_ptr<Tensor> h,
                        std::shared_ptr<Tensor> d,
                        StatusCallback callback) {
  auto device_context = context->op_device_context();
  auto device = static_cast<::tensorflow::Device*>(context->device());
  device_context->CopyCPUTensorToDevice(h.get(), device, d.get(), callback);
}

void BackgroundMemcpyThread() {
  try {

  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(THREAD_SLEEP_DURATION);

    std::deque<MemcpyRequest> tmp_queue;
    ptre_global.memcpy_mu.lock();
    ptre_global.memcpy_queue.swap(tmp_queue);
    ptre_global.memcpy_mu.unlock();

    for (auto& req : tmp_queue) {
      if (req.type == MEMCPY_DEVICE_TO_HOST) {
        ptre_global.commbuf_table_mu.lock();
        auto search = ptre_global.sendbuf_table.find(req.key);
        if (search == ptre_global.sendbuf_table.end()) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << "<DtoH> SKIP " << req.key;
          // Allocate new sendbuf and recvbuf and their states
          auto new_sendbuf = std::make_shared<Tensor>(
              req.tensor->dtype(), req.tensor->shape());
          auto new_sendbuf_state = std::make_shared<StateMutex>();
          ptre_global.sendbuf_table.emplace(
              req.key, TensorState(new_sendbuf, new_sendbuf_state));
          auto new_recvbuf = std::make_shared<Tensor>(
              req.tensor->dtype(), req.tensor->shape());
          auto new_recvbuf_state = std::make_shared<StateMutex>();
          ptre_global.recvbuf_table.emplace(
              req.key, TensorState(new_recvbuf, new_recvbuf_state));
          ptre_global.commbuf_table_mu.unlock();
          ptre_global.id_mu.lock();
          uint32_t new_id = ptre_global.id_table.size();
          ptre_global.id_to_name.emplace(new_id, req.key);
          ptre_global.id_table.emplace(req.key, new_id);
          ptre_global.id_mu.unlock();
#ifdef PTRE_RDMA
          // Register on BufferTable
          ptre_global.buf_table->Set(BUF_TYPE_SENDBUF, req.key,
              (void*) const_cast<char*>(new_sendbuf->tensor_data().data()),
              new_sendbuf->tensor_data().size());
          ptre_global.buf_table->Set(BUF_TYPE_SENDBUF_STATE, req.key,
              (void*) &new_sendbuf_state->state, sizeof(int));
          ptre_global.buf_table->Set(BUF_TYPE_RECVBUF, req.key,
              (void*) const_cast<char*>(new_recvbuf->tensor_data().data()),
              new_recvbuf->tensor_data().size());
          ptre_global.buf_table->Set(BUF_TYPE_RECVBUF_STATE, req.key,
              (void*) &new_recvbuf_state->state, sizeof(int));
          for (int i = 0; i < ptre_size(); i++) {
            PostRecvTensorIdNumber(i);
          }
#endif
          req.callback(Status(::tensorflow::error::Code::UNKNOWN, "skip"));
          continue;
        }

        auto rsm = ptre_global.recvbuf_table[req.key].second;
        auto sendbuf = ptre_global.sendbuf_table[req.key].first;
        auto sm = ptre_global.sendbuf_table[req.key].second;
        ptre_global.commbuf_table_mu.unlock();
        rsm->mu.lock();
        rsm->state = RECVBUF_STATE_READY;
        rsm->mu.unlock();
        sm->mu.lock();
        if (sm->state != SENDBUF_STATE_INIT) {
          // Not pulled yet. Retry.
          sm->mu.unlock();
          ptre_global.memcpy_mu.lock();
          ptre_global.memcpy_queue.push_back(std::move(req));
          ptre_global.memcpy_mu.unlock();
          continue;
        }
//if (req.key == "predictions_kernel_0") {
//  DVLOGR(0, ptre_rank()) << __FUNCTION__ << " MEMCPY_DEVICE_TO_HOST " << req.key;
//}
        sm->mu.unlock();
        auto callback = req.callback;
#if 1
        MemcpyDeviceToHost(req.context, req.tensor, sendbuf,
            [sm, callback](const Status& s) {
              if (s.ok()) {
                sm->mu.lock();
                sm->state = SENDBUF_STATE_READY;
                sm->mu.unlock();
                callback(s);
              } else {
                callback(s);
              }
            });
#else
        // For Testing
        ::tensorflow::Notification note;
        MemcpyDeviceToHost(req.context, req.tensor, sendbuf,
            [&note](const Status& s) {
              note.Notify();
            });
        note.WaitForNotification();
        sm->state = SENDBUF_STATE_READY;
        sm->mu.unlock();
        callback(Status::OK());
#endif
      } else {
        ptre_global.commbuf_table_mu.lock();
        auto search = ptre_global.recvbuf_table.find(req.key);
        ptre_global.commbuf_table_mu.unlock();
        if (search == ptre_global.recvbuf_table.end()) {
          req.callback(Status::OK());
          continue;
        }
        ptre_global.commbuf_table_mu.lock();
        auto recvbuf = ptre_global.recvbuf_table[req.key].first;
        auto sm = ptre_global.recvbuf_table[req.key].second;
        ptre_global.commbuf_table_mu.unlock();
        sm->mu.lock();
        if (sm->state == RECVBUF_STATE_INIT) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << "<HtoD> SKIP " << req.key;
          // No communication request posted for this tensor. Skip.
          sm->mu.unlock();
          req.callback(Status::OK());
        } else if (sm->state == RECVBUF_STATE_MEMCPY_READY) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << "<HtoD> FETCH " << req.key;
//if (req.key == "predictions_kernel_0") {
//  DVLOGR(0, ptre_rank()) << __FUNCTION__ << " MEMCPY_HOST_TO_DEVICE " << req.key;
//}
          sm->mu.unlock();
          auto callback = req.callback;
#if 1
          MemcpyHostToDevice(req.context, recvbuf, req.tensor,
              [sm, callback](const Status& s) {
                if (s.ok()) {
                  sm->mu.lock();
                  sm->state = RECVBUF_STATE_INIT;
                  sm->mu.unlock();
                  callback(s);
                } else {
                  callback(s);
                }
              });
#else
          // For Testing
          ::tensorflow::Notification note;
          MemcpyHostToDevice(req.context, recvbuf, req.tensor,
              [&note](const Status& s) {
              note.Notify();
            });
          note.WaitForNotification();
          sm->state = RECVBUF_STATE_INIT;
          sm->mu.unlock();
          callback(Status::OK());
#endif
        } else {
//DVLOGR(0, ptre_rank()) << __FUNCTION__ << "<HtoD> NOT_READY " << req.key;
          // Not ready yet. Retry.
          sm->mu.unlock();
          ptre_global.memcpy_mu.lock();
          ptre_global.memcpy_queue.push_back(std::move(req));
          ptre_global.memcpy_mu.unlock();
          continue;
        }
      }
    }
  }

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

Status EnqueueTensorPush(const string& name);

Status EnqueueTensorAsyncComm(OpContext* context,
                              const string var_name,
                              std::shared_ptr<Tensor> tensor,
                              StatusCallback callback,
                              CommOp comm_op) {
  try {

DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << var_name;
  MemcpyRequest req;
  req.context = context;
  req.key = var_name;
  req.tensor = tensor;
  req.type = MEMCPY_DEVICE_TO_HOST;
  req.callback = [callback, var_name](const Status& s) {
        Status enqueue_result;
        if (s.ok()) {
#ifdef PTRE_RDMA_PULL
          auto enqueue_result = EnqueueTensorPull(var_name);
#else
          auto enqueue_result = EnqueueTensorPush(var_name);
#endif
          callback(enqueue_result);
        } else {
          if (s.error_message() == "skip") {
            callback(Status::OK());
          } else {
            callback(s);
          }
        }
      };

  std::lock_guard<std::mutex> guard(ptre_global.memcpy_mu);
  ptre_global.memcpy_queue.push_back(std::move(req));

  return Status::OK();

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

Status EnqueueTensorAwaitComm(OpContext* context,
                              const string var_name,
                              std::shared_ptr<Tensor> tensor,
                              StatusCallback callback) {
  try {

  MemcpyRequest req;
  req.context = context;
  req.key = var_name;
  req.tensor = tensor;
  req.type = MEMCPY_HOST_TO_DEVICE;
  req.callback = callback;

  std::lock_guard<std::mutex> guard(ptre_global.memcpy_mu);
  ptre_global.memcpy_queue.push_back(std::move(req));

  return Status::OK();

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

Status EnqueueTensorModelaverage(OpContext* ctx, Tensor& tensor, Tensor& output,
                                 const string& var_name,
                                 StatusCallback callback,
                                 ModelaverageOp modelaverage_op) {
  Request message;
  message.set_tensor_name(var_name);
  message.set_tensor_type(tensor.dtype());

  TensorTableEntry entry;
  entry.tensor_name = var_name;
  entry.context = ctx;
  entry.tensor = std::make_shared<Tensor>(tensor);
  entry.output = std::make_shared<Tensor>(output);
  entry.callback = callback;
  // TODO: Init TensorTableEntry correctly.
  std::lock_guard<std::mutex> guard(ptre_global.mu_modelaverage);
  ptre_global.tensor_table_modelaverage.emplace(var_name, std::move(entry));
  ptre_global.message_queue_modelaverage.push(std::move(message));

  return Status::OK();
}

Status EnqueueTensorAverage(const string var_name,
                            std::shared_ptr<Tensor> sendbuf,
                            std::shared_ptr<Tensor> tensor,
                            int n);

Status EnqueueTensorPull(const string name) {
//if (name == "predictions_kernel_0") {
//  DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
//}
  ptre_global.commbuf_table_mu.lock();
  auto send_tensor = ptre_global.sendbuf_table[name].first;
  auto recv_tensor = ptre_global.recvbuf_table[name].first;
  auto sm = ptre_global.recvbuf_table[name].second;
  ptre_global.commbuf_table_mu.unlock();
#if 1
  auto enqueue_func = [name, send_tensor, recv_tensor]() {
        EnqueueTensorAverage(name, send_tensor, recv_tensor, 2);
      };
  TensorTableEntry entry;
  entry.tensor_name = name;
  entry.tensor = send_tensor;
  entry.output = recv_tensor;
  entry.callback = [sm, enqueue_func](const Status& s) {
        sm->mu.lock();
        sm->state = RECVBUF_STATE_RECV_DONE;
        sm->mu.unlock();
        enqueue_func();
      };

  std::lock_guard<std::mutex> guard(ptre_global.pull_mu);
  ptre_global.pull_table.emplace(name, entry);
  ptre_global.pull_queue.push_back(name);

  return Status::OK();
#else
  // For Debugging.
  EnqueueTensorAverage(name, send_tensor, recv_tensor, 2);
  ptre_global.commbuf_table_mu.lock();
  auto ssm = ptre_global.sendbuf_table[name].second;
  ptre_global.commbuf_table_mu.unlock();
  ssm->mu.lock();
  ssm->state = SENDBUF_STATE_INIT;
  ssm->mu.unlock();
  return Status::OK();
#endif
}

Status EnqueueTensorPush(const string& name) {
  try {

DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
  ptre_global.commbuf_table_mu.lock();
  auto send_tensor = ptre_global.sendbuf_table[name].first;
  auto send_tensor_state = ptre_global.sendbuf_table[name].second;
  ptre_global.commbuf_table_mu.unlock();

  RdmaEntry* entry;
  entry = new RdmaEntry();
  entry->tensor_name = name;
  entry->tensor = send_tensor;
  entry->tensor_state = send_tensor_state;
  // Next peer strategy
  // TODO: Use dynamic peer selection
  entry->rank = (ptre_rank() + 1) % ptre_size();
  entry->state = RDMA_OP_STATE_WRITE_TENSOR;

  ptre_global.push_mu.lock();
  ptre_global.push_queue.push_back(entry);
  ptre_global.push_mu.unlock();

  return Status::OK();

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

Status EnqueueTensorAllreduce(OpContext* ctx, Tensor& tensor, Tensor& output,
                              const string node_name, StatusCallback callback,
                              ReduceOp reduce_op) {
  //Status status;
  Request message;
  message.set_request_rank(ptre_rank());
  message.set_tensor_name(node_name);
  message.set_tensor_type(tensor.dtype());

  TensorTableEntry entry;
  entry.tensor_name = node_name;
//LOG(INFO) << __FUNCTION__ << ": ctx=" << (uint64_t) ctx;
  entry.context = ctx;
  entry.tensor = std::make_shared<Tensor>(tensor);
  entry.output = std::make_shared<Tensor>(output);
  entry.callback = callback;
  //entry.device = device;

  std::lock_guard<std::mutex> guard(ptre_global.mu);
  ptre_global.tensor_table.emplace(entry.tensor_name, entry);
  ptre_global.message_queue.push(message);
//DVLOG(0) << __FUNCTION__ << "\n***tensor=" << (uint64_t) entry.tensor->tensor_data().data() << ", output=" << (uint64_t) entry.output->tensor_data().data() << ", name=" << node_name;

  return Status::OK();
}

void PerformAverage(std::shared_ptr<Tensor> sendbuf,
                    std::shared_ptr<Tensor> tensor, int n,
                    std::shared_ptr<StateMutex> sm) {
  float* data = (float*) const_cast<char*>(tensor->tensor_data().data());
  if (sendbuf == nullptr) {
    for (int i = 0; i < tensor->NumElements(); i++) {
      data[i] /= n;
    }
  } else {
    float* sdata = (float*) const_cast<char*>(sendbuf->tensor_data().data());
    for (int i = 0; i < tensor->NumElements(); i++) {
      data[i] = (data[i] + sdata[i]) / n;
    }
  }
  sm->mu.lock();
  sm->state = RECVBUF_STATE_MEMCPY_READY;
  sm->mu.unlock();
}

Status EnqueueTensorAverage(const string var_name,
                            std::shared_ptr<Tensor> sendbuf,
                            std::shared_ptr<Tensor> tensor,
                            int n) {
//if (var_name == "predictions_kernel_0") {
//  DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << var_name;
//}
#if 0
  ptre_global.commbuf_table_mu.lock();
  auto sm = ptre_global.recvbuf_table[var_name].second;
  ptre_global.commbuf_table_mu.unlock();
  auto handle = std::async(std::launch::async, PerformAverage, sendbuf, tensor,
      n, sm);
  return Status::OK();
#elif 0
  // For Debugging.
  ptre_global.commbuf_table_mu.lock();
  auto ssm = ptre_global.sendbuf_table[var_name].second;
  auto sm = ptre_global.recvbuf_table[var_name].second;
  ptre_global.commbuf_table_mu.unlock();
  std::copy(sendbuf->tensor_data().begin(), sendbuf->tensor_data().end(),
      const_cast<char*>(tensor->tensor_data().data()));
  sm->mu.lock();
  sm->state = RECVBUF_STATE_MEMCPY_READY;
  sm->mu.unlock();
  ssm->mu.lock();
  ssm->state = SENDBUF_STATE_INIT;
  ssm->mu.unlock();
  return Status::OK();
#else
  // For Debugging.
  ptre_global.commbuf_table_mu.lock();
  auto ssm = ptre_global.sendbuf_table[var_name].second;
  auto sm = ptre_global.recvbuf_table[var_name].second;
  ptre_global.commbuf_table_mu.unlock();
  ptre_global.avg_mu.lock();
  ptre_global.avg_queue.push(var_name);
  ptre_global.avg_mu.unlock();
  ptre_global.avg_cv.notify_one();
  return Status::OK();
#endif
}

void AvgThread() {
  try {

  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(THREAD_SLEEP_DURATION);

    //std::vector<string> tensor_names;
    std::unique_lock<std::mutex> lk(ptre_global.avg_mu);
    ptre_global.avg_cv.wait(lk, [&] { return !ptre_global.avg_queue.empty(); });
    //while (!ptre_global.avg_queue.empty()) {
    //  tensor_names.push_back(std::move(ptre_global.avg_queue.front()));
    //  ptre_global.avg_queue.pop();
    //}
    auto name = std::move(ptre_global.avg_queue.front());
    ptre_global.avg_queue.pop();
    lk.unlock();

    //for (auto& name : tensor_names) {
      ptre_global.commbuf_table_mu.lock();
      auto sb = ptre_global.sendbuf_table[name].first;
      auto ssm = ptre_global.sendbuf_table[name].second;
      auto rb = ptre_global.recvbuf_table[name].first;
      auto rsm = ptre_global.recvbuf_table[name].second;
      ptre_global.commbuf_table_mu.unlock();

      PerformAverage(sb, rb, 2, rsm);
    //}
  }

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

// --------------------------------------------------------------------------

#ifdef PTRE_RDMA_PUSH
inline int ReadStateVolatile(volatile int* state) {
  return *state;
}

void EnqueueAvgThread() {
  try {

  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    std::vector<string> tensor_names;
    std::vector<std::shared_ptr<Tensor>> sendbufs;
    std::vector<std::shared_ptr<Tensor>> recvbufs;
    ptre_global.commbuf_table_mu.lock();
    for (auto& it : ptre_global.recvbuf_table) {
      // TODO: Should we do RDMA_READ?
      if (it.second.second->state == RECVBUF_STATE_RECV_DONE) {
        tensor_names.push_back(it.first);
        sendbufs.push_back(ptre_global.sendbuf_table[it.first].first);
        recvbufs.push_back(it.second.first);
      }
    }
    ptre_global.commbuf_table_mu.unlock();

    for (int i = 0; i < tensor_names.size(); i++) {
      EnqueueTensorAverage(tensor_names[i], sendbufs[i], recvbufs[i], 2);
    }
  }

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}
#endif

// --------------------------------------------------------------------------

void PrepareRdmaPush(RdmaEntry* entry);

void PushThread() {
  try {

  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(THREAD_SLEEP_DURATION);

    // RdmaEntry queue for this cycle
    std::deque<RdmaEntry*> entries;
    ptre_global.push_mu.lock();
    ptre_global.push_queue.swap(entries);
    ptre_global.push_mu.unlock();

    for (auto entry : entries) {
      // Assume not retry for this entry in this thread
      PrepareRdmaPush(entry);

      RdmaWrite(entry);
    }
  }

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
}

// --------------------------------------------------------------------------

void ProcessWC(RdmaEntry* entry) {
#ifdef PTRE_RDMA_PULL
  exit(1);
#else
  if (entry->state == RDMA_OP_STATE_WRITE_TENSOR) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " WRITE_TENSOR DONE " << entry->tensor_name;
#if 1
    entry->tensor_state->mu.lock();
    entry->tensor_state->state = SENDBUF_STATE_INIT;
    entry->tensor_state->mu.unlock();
    delete entry;
#else
    // WRITE_TENSOR done. Enqueue WRITE STATE
    entry->state = RDMA_OP_STATE_WRITE_STATE;
    *((int*) entry->state_mr->addr) = RECVBUF_STATE_RECV_DONE;
    RdmaWrite(entry);
#endif
  } else if (entry->state == RDMA_OP_STATE_WRITE_STATE) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " WRITE_STATE DONE " << entry->tensor_name;
    entry->tensor_state->mu.lock();
    entry->tensor_state->state = SENDBUF_STATE_INIT;
    entry->tensor_state->mu.unlock();
    delete entry;
  } else {
    exit(1);
  }
#endif
}

void ProcessSendWCs(struct ibv_wc* wcs, const int num_wcs) {
  std::vector<RdmaEntry*> entries;
  for (int i = 0; i < num_wcs; i++) {
    assert(wcs[i].status == IBV_WC_SUCCESS);
    RdmaEntry* entry = reinterpret_cast<RdmaEntry*>(wcs[i].wr_id);
    entries.push_back(entry);
  }

  for (auto entry : entries) {
    ProcessWC(entry);
  }
}

void EnqueueAvgWithTensorIdNumber(const uint32_t id) {
  ptre_global.id_mu.lock();
  string& name = ptre_global.id_to_name[id];
  ptre_global.id_mu.unlock();
  ptre_global.commbuf_table_mu.lock();
  auto sendbuf = ptre_global.sendbuf_table[name].first;
  auto recvbuf = ptre_global.recvbuf_table[name].first;
  auto rsm = ptre_global.recvbuf_table[name].second;
  ptre_global.commbuf_table_mu.unlock();
  rsm->mu.lock();
  rsm->state = RECVBUF_STATE_RECV_DONE;
  rsm->mu.unlock();
  EnqueueTensorAverage(name, sendbuf, recvbuf, 2);
}

void ProcessRecvWCs(struct ibv_wc* wcs, const int num_wcs) {
  std::vector<uint32_t> ids;
  int rank;
  for (int i = 0; i < num_wcs; i++) {
    assert(wcs[i].status == IBV_WC_SUCCESS);
    RdmaRecvEntry* entry = reinterpret_cast<RdmaRecvEntry*>(wcs[i].wr_id);
    if (i == 0) rank = entry->rank;
    ids.push_back(ntohl(wcs[i].imm_data));
    delete entry;
  }

  for (auto id : ids) {
    EnqueueAvgWithTensorIdNumber(id);
  }

  for (int i = 0; i < ids.size(); i++) {
    PostRecvTensorIdNumber(rank);
  }
}

void PollingThreadPerQP(int dst) {
  assert(!"Not implemented yet.");
#if 0
  struct ibv_wc wcs[MAX_CQE_DEFAULT];
  struct ibv_cq* cq = ptre_global.rdma_mgr->send_cq(dst);
  struct ibv_cq* rcq = ptre_global.rdma_mgr->recv_cq(dst);
  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(THREAD_SLEEP_DURATION);
    ProcessSendCQ(cq, wcs);
    ProcessRecvCQ(rcq, wcs);
  }
#endif
}

void PollingThread() {
  std::vector<RdmaChannel*> channels;
  channels.reserve(ptre_size());
  for (int dst = 0; dst < ptre_size(); dst++) {
    channels[dst] = ptre_global.rdma_mgr->GetChannel(dst);
  }
  int num_wcs = 0;
  struct ibv_wc wcs[MAX_CQE_DEFAULT];
  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(THREAD_SLEEP_DURATION);
    for (int dst = 0; dst < ptre_size(); dst++) {
      channels[dst]->PollSendCQ(wcs, &num_wcs);
      ProcessSendWCs(wcs, num_wcs);
      channels[dst]->PollRecvCQ(wcs, &num_wcs);
      ProcessRecvWCs(wcs, num_wcs);
    }
  }
}

void PollingRecvThread() {
  assert(!"Not implemented yet.");
#if 0
  try {

  struct ibv_wc wcs[MAX_CQE_DEFAULT];
  while (!ptre_global.shutdown) {
    std::this_thread::sleep_for(THREAD_SLEEP_DURATION);
    for (int dst = 0; dst < ptre_size(); dst++) {
      struct ibv_cq* rcq = ptre_global.rdma_mgr->recv_cq(dst);
      ProcessRecvCQ(rcq, wcs);
    }
  }

  } catch (const std::bad_alloc& e) {
    LOG(ERROR) << "Allocation failed: " << e.what();
    exit(1);
  }
#endif
}

// --------------------------------------------------------------------------

bool RunLoopOnceModelaverage() {
  std::vector<Request> requests;
  {
    std::lock_guard<std::mutex> guard(ptre_global.mu_modelaverage);
    while (!ptre_global.message_queue_modelaverage.empty()) {
      auto& req = ptre_global.message_queue_modelaverage.front();
      requests.push_back(std::move(req));
      ptre_global.message_queue_modelaverage.pop();
    }
  }

  std::vector<TensorTableEntry> entries;
  {
    std::lock_guard<std::mutex> guard(ptre_global.mu_modelaverage);
    for (auto& req : requests) {
      const string& name = req.tensor_name();
      auto search = ptre_global.tensor_table_modelaverage.find(name);
      if (search == ptre_global.tensor_table_modelaverage.end()) {
        LOG(ERROR) << "KEY NOT FOUND: " << name;
        exit(EXIT_FAILURE);
      }
      auto& entry = search->second;
      entries.push_back(std::move(entry));
      ptre_global.tensor_table_modelaverage.erase(search);
    }
  }

  for (auto& entry : entries) {
    const string& name = entry.tensor_name;
//LOG(INFO) << __FUNCTION__ << ": entry.tensor_name()=" << name;
    // TODO: Check whether this remote variable is up-to-date
    RemoteVariable* rvar = ptre_global.cm->remote_variable(name);
    if (rvar->agg_count() > 0) {
      memcpy(const_cast<char*>(entry.output->tensor_data().data()),
          rvar->tensor()->tensor_data().data(),
          rvar->tensor()->AllocatedBytes());
      /*
      std::copy(
          const_cast<char*>(entry.output->tensor_data().begin()), entry.output->tensor_data().end(),
          rvar->tensor()->tensor_data().begin());
      */
    }
    entry.callback(Status::OK());
  }

  return !ptre_global.shutdown;
}

void BackgroundThreadLoopModelaverage() {
  while (RunLoopOnceModelaverage()) continue;
}

// --------------------------------------------------------------------------

RemoteAddr GetOrRetrieveRemoteAddress(const int rank, const BufType& type,
                                        const string& name) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
  int ret;
  RemoteAddr addr;
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
  ret = ptre_global.rdma_mgr->GetRemoteAddress(
      rank, type, name, &addr.remote_addr, &addr.rkey);
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
  if (ret) {
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
    GrpcClient* client;
    ptre_global.grpc_client_cache->GetClient(rank, &client);
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
    ret = client->GetRemoteAddress(
        type, name, &addr.remote_addr, &addr.rkey);
    assert(ret == 0);
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
    ptre_global.rdma_mgr->SetRemoteAddress(
        rank, type, name, addr.remote_addr, addr.rkey);
  }
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
  return addr;
}

void PrepareRdmaPush(RdmaEntry* entry) {
  // Sendbuf for RDMA_WRITE tensor
  struct ibv_mr* tensor_mr = ptre_global.rdma_mgr->GetMR(BUF_TYPE_SENDBUF,
      entry->tensor_name);
  if (tensor_mr == NULL) {
    ptre_global.commbuf_table_mu.unlock();
    auto iter = ptre_global.sendbuf_table.find(entry->tensor_name);
    // sendbuf must have been initialized at memcpy stage.
    assert(iter != ptre_global.sendbuf_table.end());
    auto sendbuf = iter->second.first;
    ptre_global.commbuf_table_mu.unlock();
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << entry->tensor_name;
    tensor_mr = ptre_global.rdma_mgr->RegisterMR(BUF_TYPE_SENDBUF,
        entry->tensor_name,
        (void*) const_cast<char*>(sendbuf->tensor_data().data()),
        sendbuf->tensor_data().size(), IBV_ACCESS_LOCAL_WRITE);
  }
  entry->tensor_mr = tensor_mr;

  // Sendbuf for RDMA_WRITE state
  struct ibv_mr* state_mr = ptre_global.rdma_mgr->GetMR(
      BUF_TYPE_RECVBUF_STATE_WRITE, entry->tensor_name);
  if (state_mr == NULL) {
    void* writebuf = ptre_global.buf_table->GetOrAllocate(
        BUF_TYPE_RECVBUF_STATE_WRITE, entry->tensor_name);
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << entry->tensor_name;
    state_mr = ptre_global.rdma_mgr->RegisterMR(BUF_TYPE_RECVBUF_STATE_WRITE,
        entry->tensor_name, writebuf, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
  }
  entry->state_mr = state_mr;

  // Remote address of RECVBUF
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << entry->tensor_name;
  entry->tensor_addr = GetOrRetrieveRemoteAddress(entry->rank,
      BUF_TYPE_RECVBUF, entry->tensor_name);

  // Remote address of RECVBUF_STATE
DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << entry->tensor_name;
  auto state_addr = GetOrRetrieveRemoteAddress(
      entry->rank, BUF_TYPE_RECVBUF_STATE, entry->tensor_name);
  entry->state_addr = state_addr;

  // Remote tensor ID number
  // TODO: Do all workers always use the same id??
  entry->tensor_id = ptre_global.id_table[entry->tensor_name];

DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << entry->tensor_name;
  entry->channel = ptre_global.rdma_mgr->GetChannel(entry->rank);
}

#if 0
void PrepareRdmaPull(RdmaEntry* entry) {
  // Readbuf for reading remote's SENDBUF_STATE
  auto state_mr = ptre_global.rdma_mgr->GetMR(BUF_TYPE_SENDBUF_STATE_READ,
      entry->tensor_name);
  if (state_mr == NULL) {
    void* readbuf = ptre_global.buf_table->GetOrAllocate(
        BUF_TYPE_SENDBUF_STATE_READ, entry->tensor_name);
    state_mr = ptre_global.rdma_mgr->RegisterMR(BUF_TYPE_SENDBUF_STATE_READ,
        entry->tensor_name, readbuf, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
  }
  entry->state_mr = state_mr;

  // Remote address of SENDBUF_STATE
  auto state_addr = GetOrRetrieveRemoteAddress(
      entry->rank, BUF_TYPE_SENDBUF_STATE, entry->tensor_name);
  entry->state_addr = state_addr;

}
#endif

#if 0
bool RunLoopOncePull() {
  std::this_thread::sleep_for(std::chrono::microseconds(100));

  // Prepare entries for this cycle
  std::deque<RdmaEntry*> entries;
  {
    std::lock_guard<std::mutex> guard(ptre_global.pull_mu);
    ptre_global.pull_queue.swap(tmp_queue);
  }

  // Prepare RDMA attributes
  for (auto entry : entries) {
    PrepareRdma(entry);
  }

  // TODO: select dst dynamically.
  int dst = (ptre_global.rank + 1) % ptre_global.size;
  for (auto entry : entries) {
    RdmaRead
  }
  return !ptre_global.shutdown;
}
#elif 0
// TODO: Update the re-queueing mechanism
bool RunLoopOncePull() {
  std::this_thread::sleep_for(std::chrono::microseconds(100));

  std::deque<string> tmp_queue;
  std::vector<RdmaTensorEntry> entries;
  ptre_global.pull_mu.lock();
  ptre_global.pull_queue.swap(tmp_queue);
  for (auto& name : tmp_queue) {
    auto it = ptre_global.pull_table.find[name];
    assert(it != ptre_global.pull_table.end());
    entries.push_back(std::move(it->second));
    ptre_global.pull_table.erase(it);
  }
  ptre_global.pull_mu.unlock();

  // TODO: select dst dynamically.
  int dst = (ptre_global.rank + 1) % ptre_global.size;
  for (auto& entry : entries) {
    auto& name = entry.name;
    TcpGrpcClient* client;
    ptre_global.tcp_grpc_client_cache->GetClient(dst, &client);
    int ret = client->PullTensor(name, 0, *entry.output);
    if (ret == 0) {
      entry.callback(Status::OK());
      ptre_global.pull_mu.lock();
      ptre_global.pull_table.erase(name);
      ptre_global.pull_mu.unlock();
    } else {
      // Retry
      ptre_global.pull_mu.lock();
      ptre_global.pull_queue.push_back(name);
      ptre_global.pull_mu.unlock();
    }
  }
  return !ptre_global.shutdown;
}

void BackgroundThreadLoopPull() {
  while (RunLoopOncePull()) continue;
}
#endif

// --------------------------------------------------------------------------

bool RunLoopOnce(PtreGlobal& state);

void BackgroundThreadLoop(PtreGlobal& state) {
  return;
  while (RunLoopOnce(state)) continue;
DVLOG(0) << __FUNCTION__ << "END END END";
}

void ComputeResponseList(ResponseList& response_list);

void PerformOperation(Response response, PtreGlobal& state);

bool RunLoopOnce(PtreGlobal& state) {

LOG(INFO) << "1111";
  ResponseList response_list;
  ComputeResponseList(response_list);
LOG(INFO) << "2222";

  for (auto& response : response_list.responses()) {
    PerformOperation(response, state);
  }
LOG(INFO) << "3333";
  //bool ret = !ptre_global.shutdown;
//LOG(INFO) << "ret=" << ret;

  return !ptre_global.shutdown;
  //return ret;
}

void ComputeResponseList(ResponseList& response_list) {
std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  std::vector<Request> requests;
  {
    std::lock_guard<std::mutex> guard(ptre_global.mu);
    while (!ptre_global.message_queue.empty()) {
      auto&& req = ptre_global.message_queue.front();
//DVLOG(0) << "FROM MESSAGE QUEUE: " << req.tensor_name();
      requests.push_back(std::move(req));
      ptre_global.message_queue.pop();
    }
  }
LOG(INFO) << "COLLECTED REQUESTS FROM REQUEST QUEUE: " << requests.size();
bool has_requests = requests.size() > 0;

  // TODO: Use RDMA Write/Read for this coordination.
  // TODO: Use local rank to perform sub-group Allreduce
  bool is_coordinator = ptre_rank() == 0;
  int my_rank = ptre_rank();
  int comm_size = ptre_size();

  // TODO: BUG FIX!
  //  There are cases where there are no additional requests from message queue
  //  but we still need to receive remote requests.
  auto& req_table = ptre_global.message_table;
  for (auto& req : requests) {
    auto& name = req.tensor_name();
    if (req_table.find(name) == req_table.end()) {
      req_table.emplace(name, std::move(std::unordered_map<int, Request>()));
    }
    //req_table[name].push_back(std::move(req));
    req_table[name].emplace(my_rank, req);
  }

// 1 === 1 === 1 === 1 === 1 === 1 === 1 === 1 === 1 === 1 === 1 === 1 === 1 ===

  RequestList my_req_list;
  for (auto& req : requests) {
    Request* added = my_req_list.add_requests();
    *added = std::move(req);
  }
  if (is_coordinator) {

// <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2 <<= 2
    // Gather Requests
    int serialized_lengths[ptre_size()];
    std::map<int, RdmaRequest> rdma_reqs;
    for (int i = 0; i < ptre_size(); i++) {
      if (i == 0) continue;
      bool do_recv = false;
      for (auto& iter : req_table) {
        auto& table = iter.second;
        if (table.find(i) == table.end()) {
          do_recv = true;
          break;
        }
      }
      if (do_recv || true) {
        //PrintDebugMessageTable();
        rdma_reqs[i];
        //rdma_reqs.emplace(i, RdmaRequest());
        RdmaIrecv((void*) &serialized_lengths[i], 1, DataType::DT_INT32, i, 0,
            ptre_global.rdma_ctx, &rdma_reqs[i]);
      }
    }
    std::map<int, string> recvbufs;
    //std::vector<string> recvbufs;
    //recvbufs.emplace_back();
    //for (int i = 0; i < ptre_size(); i++) {
    for (auto& iter : rdma_reqs) {
      auto& i = iter.first;
//DVLOG(0) << __FUNCTION__ << ": 1000 Receive Length of messages";
LOG(INFO) << "WAIT IRECV message length from RANK " << i;
      RdmaWait(&iter.second, NULL);
//DVLOG(0) << __FUNCTION__ << ": 1001 serialized_lengths[" << i << "]=" << serialized_lengths[i];
      recvbufs.emplace(iter.first, std::move(string(serialized_lengths[i], 0)));
      //recvbufs.emplace_back(serialized_lengths[i], 0);
      iter.second.Clear();
      RdmaIrecv((void*) recvbufs[iter.first].c_str(), serialized_lengths[i],
          DataType::DT_BOOL, i, 0, ptre_global.rdma_ctx, &iter.second);
    }
    //}
    //std::vector<RequestList> req_lists;
    //req_lists.push_back(std::move(my_req_list));
    std::map<int, RequestList> req_lists;
    req_lists.emplace(my_rank, std::move(my_req_list));
    //for (int i = 0; i < ptre_size(); i++) {
    for (auto& iter : rdma_reqs) {
      auto& i = iter.first;
LOG(INFO) << "WAIT IRECV RequestList from RANK " << i;
      RdmaWait(&rdma_reqs[iter.first], NULL);
      RequestList req_list;
//DVLOG(0) << __FUNCTION__ << ": 2001 recvbufs[" << i << "]=" << recvbufs[i];
      req_list.ParseFromString(recvbufs[i]);
      req_lists.emplace(iter.first, std::move(req_list));
      //req_lists.emplace_back();
      //req_lists.back().ParseFromString(recvbufs[i]);
    }
LOG(INFO) << "GATHERED REQUESTS FROM OTHER RANKS";
    // }

    // Compute Responses
    // 1 Response 1 Allreduce
    auto&& table = ptre_global.message_table;
    for (auto& iter : req_lists) {
      auto& rcv_rank = iter.first;
      auto& req_list = iter.second;
      for (auto& req : req_list.requests()) {
        auto& rcv_name = req.tensor_name();
        if (table.find(rcv_name) == table.end()) {
          //table.emplace(rcv_name, std::move(std::vector<Request>()));
          table.emplace(rcv_name,
              std::move(std::unordered_map<int, Request>()));
        }
        //table[rcv_name].push_back(std::move(req));
        table[rcv_name].emplace(rcv_rank, std::move(req));
      }
    }
    if (rdma_reqs.size() > 0) {
      //PrintDebugMessageTable();
    }

    std::vector<string> ready_tensor_names;
    auto table_it = table.begin();
    while (table_it != table.end()) {
//LOG(INFO) << __FUNCTION__ << ": tensor = " << table_it->first;
      auto& row = table_it->second;
      if (row.size() == comm_size) {
//LOG(INFO) << __FUNCTION__ << ": ready_tensor = " << table_it->first;
        //ready_tensor_names.push_back(std::move(table_it->first));
        ready_tensor_names.push_back(table_it->first);
        //table_it = table.erase(table_it);
        table_it++;
      } else {
        table_it++;
      }
    }
LOG(INFO) << "COMPUTED READY TENSORS";
    // TODO: Implement Tensor Fusion
    for (auto&& name : ready_tensor_names) {
      auto search = table.find(name);
      Response res;
      res.set_response_type(RESPONSE_TYPE_ALLREDUCE);
      //res.set_tensor_type(DataType(table[name][0].tensor_type()));
      res.set_tensor_type(DataType(search->second[0].tensor_type()));
      res.add_tensor_name(std::move(name));
      response_list.add_response(std::move(res));
      table.erase(search);
    }
LOG(INFO) << "ADDED READY TENSORS TO RESPONSE_LIST";
//if (response_list.responses().size() > 0) PrintDebugResponseList(response_list);
// 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>> 3 =>>
    if (response_list.responses().size() > 0 || true && comm_size > 1) {
//LOG(INFO) << ": response_list.responses().size()=" << response_list.responses().size();
      // Send Final Tensors
      ResponseListProto proto;
      response_list.AsProto(&proto);
      string sendbuf;
      proto.SerializeToString(&sendbuf);
//LOG(INFO) << __FUNCTION__ << std::endl << sendbuf;
      int serialized_length = sendbuf.length();
//LOG(INFO) << __FUNCTION__ << ": 3000";
      RdmaBcast((void*) &serialized_length, 1, DataType::DT_INT32, 0,
          ptre_global.rdma_ctx);
LOG(INFO) << "Successfully Bcasted LENGTH(message)=" << serialized_length;
      RdmaBcast((void*) sendbuf.c_str(), serialized_length, DataType::DT_BOOL, 0,
          ptre_global.rdma_ctx);
LOG(INFO) << "Successfully Bcasted message";
    }
LOG(INFO) << "BROADCASTED RESPONSE_LIST TO OTHER RANKS";
  } else {
    // *** Other ranks
    if (req_table.size() > 0 || true) {
      if (my_req_list.requests_size() > 0 || true) {
//LOG(INFO) << __FUNCTION__ << ": my_req_list.requests_size()=" << my_req_list.requests_size();
    // 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>> 2 =>>
        string send_str(std::move(my_req_list.SerializeAsString()));
        int serialized_length = send_str.length();
//DVLOG(0) << __FUNCTION__ << ": 5000 Send Length of message = " << serialized_length;
LOG(INFO) << "SEND msg_size TO COORDINATOR RANK";
        RdmaSend((void*) &serialized_length, 1, DataType::DT_INT32, 0, 0,
            ptre_global.rdma_ctx);
//DVLOG(0) << __FUNCTION__ << ": 6000 Send Message = " << send_str;
LOG(INFO) << "SEND REQUEST_LIST TO COORDINATOR RANK";
        RdmaSend((void*) send_str.c_str(), serialized_length, DataType::DT_BOOL, 0,
            0, ptre_global.rdma_ctx);
      }

  // <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3 <<= 3
      // Receive Response
//LOG(INFO) << __FUNCTION__ << ": 7000";
      int serialized_length;
LOG(INFO) << "RECV msg_size FROM COORDINATOR RANK";
      RdmaRecv((void*) &serialized_length, 1, DataType::DT_INT32, 0, 0,
          ptre_global.rdma_ctx, NULL);
      string recvbuf(serialized_length, 0);
//LOG(INFO) << __FUNCTION__ << ": 8000";
LOG(INFO) << "RECV RESPONSE_LIST FROM COORDINATOR RANK";
      RdmaRecv((void*) recvbuf.c_str(), serialized_length, DataType::DT_BOOL, 0,
          0, ptre_global.rdma_ctx, NULL);
      ResponseListProto proto;
      proto.ParseFromString(recvbuf);
      response_list.FromProto(std::move(proto));

      for (auto& res : response_list.responses()) {
        for (auto& name : res.tensor_names()) {
          req_table.erase(name);
        }
      }
    }
  }

//if (has_requests) LOG(INFO) << __FUNCTION__ << "ENDENDENDEND";
}

void PrintDebugTensorTable() {
  std::stringstream ss;
  auto& table = ptre_global.tensor_table;
  ss << __FUNCTION__;
  for (auto& iter1 : table) {
    ss << std::endl;
    auto& tensor_name = iter1.first;
    ss << tensor_name << ", tensor->NumElements()=" << iter1.second.tensor->NumElements();
  }
  //DVLOG(0) << ss.str();
}

void PerformOperation(Response response, PtreGlobal& state) {
std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//LOG(INFO) << __FUNCTION__ << "1000";
  //if (response.tensor_names().size() == 0) return;

  std::vector<TensorTableEntry> entries;
  entries.reserve(response.tensor_names().size());
  {
    std::lock_guard<std::mutex> guard(ptre_global.mu);
//if (response.tensor_names().size() > 0) PrintDebugTensorTable();
    for (auto&& name : response.tensor_names()) {
      auto search = ptre_global.tensor_table.find(name);
      assert(search != ptre_global.tensor_table.end());
//LOG(INFO) << __FUNCTION__ << ": tensor_name=" << search->second.tensor_name << ", tensor->NumElements()=" << search->second.tensor->NumElements();
      entries.push_back(std::move(search->second));
      ptre_global.tensor_table.erase(search);
    }
//LOG(INFO) << "ENTRIES:";
//for (auto& entry : entries) LOG(INFO) << entry.tensor_name << ", ctx->input(0).NumElements()=" << entry.context->input(0).NumElements();
//for (auto& entry : entries) DVLOG(0) << entry.tensor_name << ", ctx->input(0).NumElements()=" << entry.context->input(0).NumElements();
  }
//LOG(INFO) << __FUNCTION__ << "2000";

  void* recvbuf;
  int num_elements = 0;
  if (entries.size() > 1) {
//LOG(INFO) << __FUNCTION__ << "3000";
    size_t total_size = 0;
    for (auto&& entry : entries) {
      auto tensor = entry.context->input(0);
      total_size += tensor.AllocatedBytes();
      num_elements += tensor.NumElements();
    }
//LOG(INFO) << __FUNCTION__ << "4000";
    recvbuf = malloc(total_size);
    size_t offset = 0;
    for (auto&& entry : entries) {
      auto tensor = entry.context->input(0);
      memcpy((void*) ((char*) recvbuf + offset), tensor.tensor_data().data(),
          tensor.AllocatedBytes());
      offset += tensor.AllocatedBytes();
    }
  } else {
//LOG(INFO) << __FUNCTION__ << "5000: response has one tensor";
    recvbuf = (void*) entries[0].output->tensor_data().data();
    num_elements = entries[0].context->input(0).NumElements();
//LOG(INFO) << __FUNCTION__ << "5100 recvbuf=" << (uint64_t) recvbuf << ", num_elements=" << num_elements;
  }
  const void* sendbuf = (entries.size() > 1
      || entries[0].context->input(0).tensor_data().data()
      == entries[0].output->tensor_data().data())
      ? COMM_IN_PLACE : entries[0].context->input(0).tensor_data().data();
//LOG(INFO) << __FUNCTION__ << "6000 sendbuf=" << (uint64_t) sendbuf;

  // TODO: Use OperationManager after generalizing operations.
#if 0
  Status status;
  try {
    status = op_mgr->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    status = errors::Unknown("PerformOperation");
  }
#else
  RdmaContext ctx(ptre_global.rdma_mgr);
//LOG(INFO) << __FUNCTION__ << "\n***sendbuf=" << (uint64_t) sendbuf << ", recvbuf=" << (uint64_t) recvbuf << ", dtype()=" << entries[0].context->input(0).dtype();
LOG(INFO) << "RDMA ALLREDUCE";
  int ret = RdmaAllreduce(sendbuf, recvbuf, num_elements,
      entries[0].context->input(0).dtype(), ReduceOp::REDUCE_SUM, &ctx);
  assert(ret == 0);
//LOG(INFO) << __FUNCTION__ << "7777";
#endif

  if (entries.size() > 1) free(recvbuf);

  for (auto&& entry : entries) {
//LOG(INFO) << __FUNCTION__ << "\n===***DONE:" << entry.tensor_name;
    entry.callback(Status::OK());
  }

}

Status PtreAllreduce(const void* sendbuf, void* recvbuf, int count) {
  int ret;
  RdmaContext ctx(ptre_global.rdma_mgr);
  ret = RdmaAllreduce(sendbuf, recvbuf, count, DataType::DT_FLOAT,
      ReduceOp::REDUCE_SUM, &ctx);
}

int ProcessCQRdmaRequest(int dst, struct ibv_cq* cq, struct ibv_wc* wcs);

void PollingThreadLoop() {
  struct ibv_wc wcs[MAX_CQE_DEFAULT];
  do {
    for (int dst = 0; dst < ptre_size(); dst++) {
      struct ibv_cq* cq = ptre_global.rdma_mgr->send_cq(dst);
      ProcessCQRdmaRequest(dst, cq, wcs);
    }
    for (int dst = 0; dst < ptre_size(); dst++) {
      struct ibv_cq* cq = ptre_global.rdma_mgr->recv_cq(dst);
      ProcessCQRdmaRequest(dst, cq, wcs);
    }
  } while (!ptre_global.shutdown);
}

// RDMA Operations
int ProcessCQRdmaRequest(int dst, struct ibv_cq* cq, struct ibv_wc* wcs) {
std::this_thread::sleep_for(std::chrono::milliseconds(1));
  int ne = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
//DVLOG(0) << __FUNCTION__ << ": ne=" << ne;
//LOG(INFO) << __FUNCTION__ << ": ne=" << ne;
  assert(ne >= 0);
  std::vector<RdmaRequest*> bad_reqs;
  for (int i = 0; i < ne; i++) {
    RdmaRequest* req = reinterpret_cast<RdmaRequest*>(wcs[i].wr_id);
    if (wcs[i].status == IBV_WC_SUCCESS) {
      if (wcs[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        req->set_imm_data(ntohl(wcs[i].imm_data));
      }
      req->Done();
    } else {
      DVLOG(0) << "wc.status=" << wcs[i].status << ", opcode="
          << wcs[i].opcode << ", mr=" << (uint64_t) req->mr()->addr;
      bad_reqs.push_back(req);
    }
  }
  if (bad_reqs.size() > 0) {
    auto channel = ptre_global.rdma_mgr->GetChannel(dst);
    LOG(ERROR) << "Recovering RdmaChannel for rank=" << dst;
    assert(channel->Recover() == 0);
    for (auto&& req : bad_reqs) {
      req->DoneFailure();
    }
  }
}

}  // namespace common
}  // namespace ptre
