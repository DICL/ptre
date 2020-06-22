#include "ptre/common/operations.h"

#include <fstream>

#include "ptre/common/logging.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/rdma/rdma_mpi.h"

namespace ptre {
namespace common {

namespace {

PtreGlobal ptre_global;

}  // namespace

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
  ptre_global.shutdown = false;

  // Init Grpc Service
  DVLOG(0) << "Init Grpc Service";
  load_grpc_hosts(grpc_hosts_file);
  ptre_global.grpc_client_cache = std::make_shared<GrpcClientCache>(rank,
      ptre_global.grpc_hosts);
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);

  // Init RdmaMgr
  DVLOG(0) << "Init Rdma Manager";
  ptre_global.rdma_mgr = new RdmaMgr(size, rank);
  ptre_global.grpc_service.SetRdmaMgr(ptre_global.rdma_mgr);
  for (int i = 0; i < ptre_global.size; i++) {
    ptre_global.qp_mus.push_back(new std::mutex());
  }

  // Connect Queue Pairs
  DVLOG(0) << "Connect Queue Pairs";
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

  DVLOG(0) << "Starting Polling Thread Loop";
  ptre_global.polling_threads.emplace_back(std::thread(PollingThreadLoop));

  DVLOG(0) << "[1/2] Done InitComm";
}

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
  //LOG(INFO) << __PRETTY_FUNCTION__ << ": " << since_last.count() / 1000 << "msec";
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
//LOG(INFO) << __PRETTY_FUNCTION__ << ": " << since_last.count() / 1000 << "msec";
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
  //LOG(INFO) << __PRETTY_FUNCTION__ << ": " << since_last.count() / 1000 << "msec";
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
  ShutdownGrpcServer();
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

namespace {

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

}

struct RvarRequest {
  OpContext* context;
  string var_name;
  Tensor* output;
  Tensor* num_agg;
  StatusCallback callback;
};

namespace {
std::queue<RvarRequest> rvar_queue;
std::mutex rvar_queue_mu;
}

// TODO: float -> T
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

Status EnqueueGetRemoteVariable(OpContext* ctx, const string& var_name,
                                Tensor* output, Tensor* num_agg,
                                StatusCallback callback) {
  Status status;

  RvarRequest e;
  e.context = ctx;
  e.var_name = var_name;
  e.output = output;
  e.num_agg = num_agg;
  e.callback = callback;

  rvar_queue_mu.lock();
  rvar_queue.push(e);
  rvar_queue_mu.unlock();

  return Status::OK();
}

bool RunLoopOnce(PtreGlobal& state);

void BackgroundThreadLoop(PtreGlobal& state) {
#if 0
  int rank = ptre_rank();
  int size = ptre_size();

  bool is_coordinator = rank == 0;
  if (is_coordinator) {
    ptre_global.message_table =
        std::unique_ptr<MessageTable>(new MessageTable());
  }

  bool should_shut_down = false;
  do {
    std::queue<Request> message_queue;
    {
      std::lock_guard<std::mutex> guard(ptre_global.mu);
      while (!ptre_global.message_queue.empty()) {
        Request message = ptre_global.message_queue.front();
        ptre_global.message_queue.pop();
        message_queue.push(message);
      }
    }

    std::vector<string> ready_to_reduce;
    while (!message_queue.empty()) {
      Request message = message_queue.front();
      message_queue.pop();

      if (is_coordinator) {
        bool reduce =
            IncrementTensorCount(ptre_global.message_table, message, size);
        if (reduce) {
          ready_to_reduce.push_back(message.tensor_name());
        }
      } else {
        string encoded_message;
        message.SerializeToString(&encoded_message);
        MPI_Send(encoded_message.c_str(), encoded_message.length() + 1,
                 MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);
      }
    }

    if (is_coordinator) {
      int completed_ranks = 1;

    }

  } while (should_shut_down);
#else
  while (RunLoopOnce(state)) continue;
#endif
}

void PerformOperation(Response response, PtreGlobal& state);

bool RunLoopOnce(PtreGlobal& state) {

  std::vector<Response> response_list;

  for (auto& response : response_list) {
    PerformOperation(response, state);
  }
}

void PerformOperation(Response response, PtreGlobal& state) {
  std::vector<TensorTableEntry> entries;
  {
    std::lock_guard<std::mutex> guard(ptre_global.mu);
    const string& name = response.tensor_names()[0];
    auto search = ptre_global.tensor_table.find(name);
    assert(search != ptre_global.tensor_table.end());
    entries.push_back(std::move(search->second));
    ptre_global.tensor_table.erase(search);
  }

  // TODO: Use OperationManager after generalizing operations.
  Status status;
#if 0
  try {
    status = op_mgr->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    status = errors::Unknown("PerformOperation");
  }
#else
  status = PtreAllreduce(
      static_cast<const void*>(entries[0].tensor->tensor_data().data()),
      (void*) entries[0].output->tensor_data().data(),
      (int) entries[0].tensor->NumElements());
#endif
}

Status PtreAllreduce(const void* sendbuf, void* recvbuf, int count) {
  int ret;
  RdmaContext ctx(ptre_global.rdma_mgr);
  ret = RdmaAllreduce(sendbuf, recvbuf, count, DataType::DT_FLOAT,
      ReduceOp::REDUCE_SUM, &ctx);
}

Status EnqueueTensorAllreduce(OpContext* ctx, Tensor* tensor, Tensor* output,
                              const string node_name, StatusCallback callback,
                              ReduceOp reduce_op) {
  Status status;
  Request message;
  message.set_request_rank(ptre_rank());
  message.set_tensor_name(node_name);
  message.set_tensor_type(tensor->dtype());

  TensorTableEntry entry;
  entry.tensor_name = node_name;
  entry.context = ctx;
  entry.tensor = tensor;
  entry.output = output;
  entry.callback = callback;

  std::lock_guard<std::mutex> guard(ptre_global.mu);
  ptre_global.tensor_table.emplace(entry.tensor_name, entry);
  ptre_global.message_queue.push(message);
}

// RDMA Operations
int ProcessCQRdmaRequest(int dst, struct ibv_cq* cq, struct ibv_wc* wcs) {
  int ne = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
  assert(ne >= 0);
  std::vector<RdmaRequest*> bad_reqs;
  for (int i = 0; i < ne; i++) {
    RdmaRequest* req = reinterpret_cast<RdmaRequest*>(wcs[i].wr_id);
    if (wcs[i].status == IBV_WC_SUCCESS) {
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

}  // namespace common
}  // namespace ptre