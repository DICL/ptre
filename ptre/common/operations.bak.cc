
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

void EnqueueAvgThread() {
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

