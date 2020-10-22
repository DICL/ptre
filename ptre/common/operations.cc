#include "ptre/common/operations.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <queue>
#include <random>

#include "ptre/common/buffer_table.h"
#include "ptre/common/logging.h"
#include "ptre/common/communication/tcp/tcp_grpc_client.h"
#include "ptre/common/rdma/rdma_controller.h"
#include "ptre/common/utils/host_file_parser.h"
#include "ptre/lib/distributions.h"
#include "third_party/minitrace/minitrace.h"

#include "tensorflow/core/common_runtime/device.h"

#include <arpa/inet.h>

#define LOGR(x) LOG(x) << __FUNCTION__ << "]" << "[" << ptre_rank() << "]"

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

std::unordered_map<string, const char*> persistent_strs;
std::mutex ps_mu;
auto& op_tracers = ptre_global.op_tracers;
std::mutex ot_mu;

std::mutex push_cnt_mu;
std::unordered_map<string, int> push_cnts;

std::mutex peer_sel_mu;
int peer_selected;
int peer_sel_cnt;
std::random_device rd;  // Non-deterministic
std::mt19937 gen(rd());
std::unique_ptr<MyDistribution> inverse_count_distribution;

std::mutex await_mu;
int await_cnt;

std::unordered_map<string, int*> simple_htod_cnt;
std::unordered_map<string, int> recent_avg_enqueue_rank;
string filePath;

}  // namespace

inline const char* ToPersistentCStr(const string& str) {
#ifdef MTR_ENABLED
  auto it = persistent_strs.find(str);
  if (it == persistent_strs.end()) {
    ps_mu.lock();
    it = persistent_strs.find(str);
    if (it == persistent_strs.end()) {
      char* new_cstr = (char*) malloc(sizeof(char) * (str.length() + 1));
      std::copy(str.begin(), str.end(), new_cstr);
      new_cstr[str.length()] = NULL;
      persistent_strs.emplace(str, new_cstr);
    }
    ps_mu.unlock();
  }
  const char* ret = persistent_strs[str];
  return ret;
#else
  return NULL;
#endif
}

void OpTrace(const string& cat, const string& str, const string& step) {
#ifdef MTR_ENABLED
  std::lock_guard<std::mutex> guard(ot_mu);
  auto category = ToPersistentCStr(cat);
  auto var_name = ToPersistentCStr(str);
  auto step_name = ToPersistentCStr(step);
  if (op_tracers.find(cat) == op_tracers.end()) {
    op_tracers[cat];
  }
  auto& t = op_tracers[cat];
  if (t.find(str) == t.end()) {
    t.emplace(str, int());
    MTR_START(category, var_name, &t[str]);
  }
  //LOG(INFO) << category << " " << var_name << " " << step_name;
  MTR_STEP(category, var_name, &t[str], step_name);
#endif
}

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
  // Rdma Service
  auto&& service = ptre_global.grpc_service;
  service.SetBarrierVariable(&ptre_global.barrier_variable);
  string server_address = "0.0.0.0:"
      + std::to_string(ptre_global.this_worker.port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Tcp Service
  //builder.RegisterService(&ptre_global.tcp_grpc_service);

  builder.SetMaxMessageSize(1 * 1024 * 1024 * 1024);
  ptre_global.grpc_server = builder.BuildAndStart();
  LOG(INFO) << "Grpc server listening on " << server_address;
  ptre_global.grpc_server->Wait();
}

void BackgroundMemcpyThread();

void PushThread();

void PollingThread();

void PollingThreadPerQP(int dst);

void PostRecvTensorIdNumber(const int rank);

void AverageThread();

void BackgroundThread();

void InitComm(int size, int rank, const string& grpc_hosts_file) {
  ptre_global.size = size;
  ptre_global.rank = rank;
  ptre_global.shutdown = false;

  // Init BufferTable
  ptre_global.buf_table = std::make_shared<BufferTable>();
  ptre_global.grpc_service.SetBufferTable(ptre_global.buf_table);
  ptre_global.grpc_service.SetCommbufState(ptre_global.commbuf_state);

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

  for (int i = 0; i < ptre_size(); i++) {
     for (int j = 0; j < 256; j++) {
       PostRecvTensorIdNumber(i);
     }
  }

  //for (int i = 0; i < ptre_size(); i++) {
  //  ptre_global.polling_threads.emplace_back(
  //      std::thread(PollingThreadPerQP, i));
  //}
  //ptre_global.polling_thread = std::thread(PollingThread);
  ////ptre_global.polling_threads.emplace_back(std::thread(PollingThreadLoop));
  //LOG(INFO) << "Launched Polling Thread";

  ptre_global.rdma_ctx = new RdmaContext(ptre_global.rdma_mgr);

  // Memcpy Thread
  //ptre_global.memcpy_thread = std::thread(BackgroundMemcpyThread);

  // RDMA push thread
  //ptre_global.push_thread = std::thread(PushThread);

  // Avg Thread
  int num_avg_threads = 4;
  if (const char* env_p = std::getenv("PTRE_NUM_AVERAGE_THREADS")) {
    num_avg_threads = atoi(env_p);
  }
  for (int i = 0; i < num_avg_threads; i++) {
    ptre_global.avg_threads.emplace_back(std::thread(AverageThread));
  }
  //ptre_global.avg_thread = std::thread(AverageThread);

  ptre_global.background_thread = std::thread(BackgroundThread);
  DVLOG(0) << "Done InitComm";
}

void PostRecvTensorIdNumber(const int rank) {
  RdmaRecvEntry* entry = new RdmaRecvEntry();
  entry->rank = rank;
  entry->channel = ptre_global.rdma_mgr->GetChannel(entry->rank);
  auto post_result = PostRecvWithImm(entry, false);
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

  std::lock_guard<std::mutex> guard(ptre_global.bcast_mu);
  // PtreBroadcast_training_SGD_fc2_kernel_momentum_0
  // PtreBroadcast_fc2_kernel_0
  string var_name = name.substr(14);
  ptre_global.bcast_done[var_name] = true;
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

#if 1
void PtreFlushSimpleHtod() {
#ifdef SIMPLE_HTOD_CNT
  std::ofstream writeFile;
  writeFile.open(filePath, std::fstream::out | std::fstream::app);
  if (writeFile.is_open()) {
    writeFile << "---------\n";
    for (int i = 0; i < ptre_global.num_tvars; i++) {
      auto& name = ptre_global.id_to_name[i];
      writeFile << name << ":";
      for (int j = 0; j < ptre_size(); j++) {
        writeFile << " " << simple_htod_cnt[name][j];
        simple_htod_cnt[name][j] = 0;  // Initialize
      }
      writeFile << std::endl;
    }
    writeFile.close();
  }
#endif
}
#endif

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

extern "C" {

int ptre_init(int size, int rank, const char* grpc_hosts_file,
              int selection_strategy, int num_push) {
  ptre_global.num_push = num_push;
  ptre_global.peer_selector = selection_strategy;
  InitComm(size, rank, grpc_hosts_file);
  //ptre_global.cm->InitPeerSelector(selection_strategy, num_push);
  //LOG(INFO) << "Peer selection strategy = " << selection_strategy;
  peer_sel_cnt = 0;
  await_cnt = 0;
  inverse_count_distribution =
      std::unique_ptr<MyDistribution>(new MyDistribution(size, rank));

#ifdef SIMPLE_HTOD_CNT
  string nameBase = "simple_htod_cnt";
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[32];
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 32, "%Y%m%d%H%M", timeinfo);

  filePath = "/tmp/simple_htod_cnt_";
  filePath = filePath + buffer + ".txt";
  LOG(INFO) << "filePath: " << filePath;
#endif
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
  LOG(ERROR) << "Deprecated.";
  exit(1);
  //ClearPullJobs();
  //CreatePullJob(ptre_global.local_step, ptre_global.num_push);
}

void ptre_barrier() {
  PtreBarrier();
}

void ptre_print_counter_summary_epoch() {
#if 0
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
#endif
}

void ptre_print_counter_summary() {
#if 0
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
#endif
}

void ptre_call_generic(const char* func_name) {
#if 1
  if (!strcmp(func_name, "FlushSimpleHtod")) {
    PtreFlushSimpleHtod();
  }
#endif
}

}  // extern "C"

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
#ifdef ATOMIC_MODEL
          {
            std::lock_guard<std::mutex> guard(ptre_global.htod_mu);
            ptre_global.num_htod++;
          }
#endif
          continue;
        }
        ptre_global.commbuf_table_mu.lock();
        auto recvbuf = ptre_global.recvbuf_table[req.key].first;
        auto sm = ptre_global.recvbuf_table[req.key].second;
        ptre_global.commbuf_table_mu.unlock();
        sm->mu.lock();
        if (sm->state == RECVBUF_STATE_INIT) {
          // No communication request posted for this tensor. Skip.
          DVLOGR(0, ptre_rank()) << __FUNCTION__ << "<HtoD> SKIP " << req.key;
          sm->mu.unlock();
          req.callback(Status::OK());
        } else if (sm->state == RECVBUF_STATE_MEMCPY_READY) {
#ifdef ATOMIC_MODEL
          {
            std::lock_guard<std::mutex> guard(ptre_global.htod_mu);
            bool skip = false;
            if (ptre_global.htod_ever_skipped) {
              skip = true;
            } else {
              ptre_global.htod_ever_performed = true;
            }
            ptre_global.htod_cnt++;
            if (ptre_global.htod_cnt == ptre_global.num_htod) {
              ptre_global.htod_ever_performed = false;
              ptre_global.htod_ever_skipped = false;
              ptre_global.htod_cnt = 0;
            }
            if (skip) {
              sm->state = RECVBUF_STATE_READY;
              sm->mu.unlock();
              req.callback(Status::OK());
              continue;
            }
          }
#endif
          DVLOGR(0, ptre_rank()) << __FUNCTION__ << "<HtoD> FETCH " << req.key;
          sm->mu.unlock();
          auto callback = req.callback;
#if 1
          MemcpyHostToDevice(req.context, recvbuf, req.tensor,
              [sm, callback](const Status& s) {
                if (s.ok()) {
                  sm->mu.lock();
                  sm->state = RECVBUF_STATE_READY;
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
          // Not ready yet. Retry.
#ifdef SKIP_HTOD_IF_NOT_READY
#ifdef ATOMIC_MODEL
          {
            std::lock_guard<std::mutex> guard(ptre_global.htod_mu);
            if (ptre_global.htod_ever_performed) {
              // Wait
              sm->mu.unlock();
              ptre_global.memcpy_mu.lock();
              ptre_global.memcpy_queue.push_back(std::move(req));
              ptre_global.memcpy_mu.unlock();
              continue;
            }
            ptre_global.htod_cnt++;
            ptre_global.htod_ever_skipped = true;
            if (ptre_global.htod_cnt == ptre_global.num_htod) {
              ptre_global.htod_ever_performed = false;
              ptre_global.htod_ever_skipped = false;
              ptre_global.htod_cnt = 0;
            }
          }
#endif
          // TODO: Check if there won't be a probelm if we set the state to
          //  RECVBUF_STATE_READY
          sm->state = RECVBUF_STATE_READY;
          sm->mu.unlock();
          req.callback(Status::OK());
#else
          sm->mu.unlock();
          ptre_global.memcpy_mu.lock();
          ptre_global.memcpy_queue.push_back(std::move(req));
          ptre_global.memcpy_mu.unlock();
#endif
        }
      }
    }
  }
}

Status EnqueueTensorPush(const string& name);

Status EnqueueTensorAsyncComm(OpContext* context,
                              const string var_name,
                              std::shared_ptr<Tensor> tensor,
                              StatusCallback callback,
                              CommOp comm_op) {
#if 0
  if (var_name == "fc1_kernel_0") {
    callback(Status::OK());
    return Status::OK();
  }
#endif
//DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << var_name;
  MemcpyRequest req;
  req.context = context;
  req.key = var_name;
  req.tensor = tensor;
  req.type = MEMCPY_DEVICE_TO_HOST;
  req.callback = [callback, var_name](const Status& s) {
        Status enqueue_result;
        if (s.ok()) {
          // TODO: DECOUPLE THIS CHAINED CALLBACK
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

  std::lock_guard<std::mutex> guard(ptre_global.dtoh_mu);
  ptre_global.dtoh_queue.push_back(std::move(req));

  return Status::OK();
}

Status EnqueueTensorAwaitComm(OpContext* context,
                              const string var_name,
                              std::shared_ptr<Tensor> tensor,
                              StatusCallback callback) {
#if 0
  if (var_name == "fc1_kernel_0") {
    callback(Status::OK());
    return Status::OK();
  }
#endif
//DVLOGR(0, ptre_rank()) << __FUNCTION__ << "] " << var_name;
  MemcpyRequest req;
  req.context = context;
  req.key = var_name;
  req.tensor = tensor;
  req.type = MEMCPY_HOST_TO_DEVICE;
  req.callback = [callback, var_name](const Status& s) {
        if (!s.ok() && s.error_message() == "skip") {
          callback(Status::OK());
        } else {
          callback(s);
        }
      };

  std::lock_guard<std::mutex> guard(ptre_global.htod_mu);
  ptre_global.htod_queue.push_back(std::move(req));

  return Status::OK();
}

Status EnqueueTensorAverage(const string var_name,
                            std::shared_ptr<Tensor> sendbuf,
                            std::shared_ptr<Tensor> tensor,
                            int n,
                            int from_rank);

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
        EnqueueTensorAverage(name, send_tensor, recv_tensor, 2, -1);
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
//DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << name;
  //ptre_global.commbuf_table_mu.lock();
  auto send_tensor = ptre_global.sendbuf_table[name].first;
  auto send_tensor_state = ptre_global.sendbuf_table[name].second;
  //ptre_global.commbuf_table_mu.unlock();

  RdmaEntry* entry;
  entry = new RdmaEntry();
  entry->tensor_name = name;
  entry->tensor = send_tensor;
  entry->tensor_state = send_tensor_state;
#if 0
  // Next peer strategy
  // TODO: Use dynamic peer selection
  entry->rank = (ptre_rank() + 1) % ptre_size();
  {
    std::lock_guard<std::mutex> guard(peer_sel_mu);
    if (peer_sel_cnt == 0) {
      int peer = (ptre_rank() + 1) % ptre_size();
      GrpcClient* client;
      ptre_global.grpc_client_cache->GetClient(peer, &client);
      if (client->AttemptPush()) {
        peer_selected = peer;
      } else {
        peer_selected = -1;  // skip
      }
    }
    entry->rank = peer_selected;
    peer_sel_cnt = (peer_sel_cnt + 1) % ptre_global.num_tvars;
  }
#elif 0
  // Random peer
  {
    std::uniform_int_distribution<int> distribution(0, ptre_size() - 1);
    std::lock_guard<std::mutex> guard(peer_sel_mu);
    if (peer_sel_cnt == 0) {
      peer_selected = -1;  // skip
      for (int i = 0; ; i++) {
        int peer = distribution(gen);
        if (peer == ptre_rank()) {
          i--;
          continue;
        }
        GrpcClient* client;
        ptre_global.grpc_client_cache->GetClient(peer, &client);
        if (client->AttemptPush()) {
          peer_selected = peer;
          break;
        }
      }
    }
    entry->rank = peer_selected;
    peer_sel_cnt = (peer_sel_cnt + 1) % ptre_global.num_tvars;
  }
#else
  // Least Selected First
  {
    std::lock_guard<std::mutex> guard(peer_sel_mu);
    if (peer_sel_cnt == 0) {
      peer_selected = -1;  // skip
      for (int i = 0; ; i++) {
        //break;
        int peer = (*inverse_count_distribution)(gen);
        if (peer == ptre_rank()) {
          i--;
          continue;
        }
        GrpcClient* client;
        ptre_global.grpc_client_cache->GetClient(peer, &client);
        if (client->AttemptPush()) {
          peer_selected = peer;
          break;
        }
      }
      if (peer_selected >= 0) {
        inverse_count_distribution->count(peer_selected);
      }
    }
    entry->rank = peer_selected;
    peer_sel_cnt = (peer_sel_cnt + 1) % ptre_global.num_tvars;
  }
#endif
  entry->state = RDMA_OP_STATE_WRITE_TENSOR;

  ptre_global.push_mu.lock();
  ptre_global.push_queue.push_back(entry);
  ptre_global.push_mu.unlock();

  return Status::OK();
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
                    std::shared_ptr<StateMutex> sm,
                    const string& name) {
#if 0
  {
    std::lock_guard<std::mutex> guard(sm->mu);
    if (sm->state != RECVBUF_STATE_RECV_DONE) {
DBGR(name, ptre_rank()) << "state=" << sm->state;
      sm->state = RECVBUF_STATE_READY;
      return;
    }
  }
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
  {
    std::lock_guard<std::mutex> guard(sm->mu);
    if (sm->state == RECVBUF_STATE_RECV_DONE) {
DBGR(name, ptre_rank()) << "state=" << sm->state;
      sm->state = RECVBUF_STATE_MEMCPY_READY;
    } else {
DBGR(name, ptre_rank()) << "state=" << sm->state;
      sm->state = RECVBUF_STATE_READY;
    }
  }
#else
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
  std::lock_guard<std::mutex> guard(sm->mu);
  sm->state = RECVBUF_STATE_MEMCPY_READY;
#endif
}

Status EnqueueTensorAverage(const string var_name,
                            std::shared_ptr<Tensor> sendbuf,
                            std::shared_ptr<Tensor> tensor,
                            int n,
                            int from_rank) {
//if (var_name == "predictions_kernel_0") {
//  DVLOGR(0, ptre_rank()) << __FUNCTION__ << " " << var_name;
//}
#if 1
  ptre_global.commbuf_table_mu.lock();
  auto ssm = ptre_global.sendbuf_table[var_name].second;
  auto sm = ptre_global.recvbuf_table[var_name].second;
  ptre_global.commbuf_table_mu.unlock();
  ptre_global.avg_mu.lock();
  ptre_global.avg_queue.push(var_name);
  recent_avg_enqueue_rank[var_name] = from_rank;
  ptre_global.avg_mu.unlock();
  ptre_global.avg_cv.notify_one();
  return Status::OK();
#else
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
#endif
}

void AverageThread() {
  DVLOGR(0, ptre_rank()) << __FUNCTION__ << "] Started.";
  while (!ptre_global.shutdown) {
    //std::this_thread::sleep_for(THREAD_SLEEP_DURATION);

    //std::vector<string> tensor_names;
    bool shutdown = false;
    std::unique_lock<std::mutex> lk(ptre_global.avg_mu);
    ptre_global.avg_cv.wait(lk,
        [&] {
          shutdown = ptre_global.shutdown;
          return (!ptre_global.avg_queue.empty() || shutdown);
        });
    if (shutdown) break;
    //while (!ptre_global.avg_queue.empty()) {
    //  tensor_names.push_back(std::move(ptre_global.avg_queue.front()));
    //  ptre_global.avg_queue.pop();
    //}
    auto name = std::move(ptre_global.avg_queue.front());
    ptre_global.avg_queue.pop();
    auto ss = ptre_global.sendbuf_table[name].second;
    ss->mu.lock();
    if (ss->state == SENDBUF_STATE_BUSY) {
      ptre_global.avg_queue.push(std::move(name));
      ss->mu.unlock();
      lk.unlock();
      continue;
    }
    ss->state = SENDBUF_STATE_BUSY;
    ss->mu.unlock();
    lk.unlock();

    //for (auto& name : tensor_names) {
      //ptre_global.commbuf_table_mu.lock();
      auto sb = ptre_global.sendbuf_table[name].first;
      auto rb = ptre_global.recvbuf_table[name].first;
      auto rsm = ptre_global.recvbuf_table[name].second;
      //ptre_global.commbuf_table_mu.unlock();
//DBGR(name, ptre_rank()) << "state=" << rsm->state;
      PerformAverage(sb, rb, 2, rsm, name);
      ss->mu.lock();
      ss->state = SENDBUF_STATE_READY;
      ss->mu.unlock();
    //}
  }
}

// --------------------------------------------------------------------------
void PrepareRdmaPush(RdmaEntry* entry);

void PushThread() {
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

void EnqueueAvgWithTensorIdNumber(const uint32_t id, const int from_rank) {
  ptre_global.id_mu.lock();
  string& name = ptre_global.id_to_name[id];
  ptre_global.id_mu.unlock();
  ptre_global.commbuf_table_mu.lock();
  auto sendbuf = ptre_global.sendbuf_table[name].first;
  auto recvbuf = ptre_global.recvbuf_table[name].first;
  auto rsm = ptre_global.recvbuf_table[name].second;
  ptre_global.commbuf_table_mu.unlock();

#if 0
  bool do_average = false;
  rsm->mu.lock();
  if (rsm->state == RECVBUF_STATE_READY) {
DBGR(name, ptre_rank()) << "state=" << rsm->state;
    rsm->state = RECVBUF_STATE_RECV_DONE;
    do_average = true;
  } else {
DBGR(name, ptre_rank()) << "state=" << rsm->state;
    rsm->state = RECVBUF_STATE_READY;
  }
  rsm->mu.unlock();
  if (do_average) {
    EnqueueTensorAverage(name, sendbuf, recvbuf, 2);
  }
#else
  EnqueueTensorAverage(name, sendbuf, recvbuf, 2, from_rank);
#endif
}

void ProcessRecvWCs(struct ibv_wc* wcs, const int num_wcs) {
#if 1
  assert(!"Not implemented yet.");
#else
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
#endif
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

// --------------------------------------------------------------------------

RemoteAddr GetOrRetrieveRemoteAddress(const int rank, const BufType& type,
                                        const string& name) {
  int ret;
  RemoteAddr addr;
  ret = ptre_global.rdma_mgr->GetRemoteAddress(
      rank, type, name, &addr.remote_addr, &addr.rkey);
  if (ret) {
    GrpcClient* client;
    ptre_global.grpc_client_cache->GetClient(rank, &client);
    ret = client->GetRemoteAddress(
        type, name, &addr.remote_addr, &addr.rkey);
    assert(ret == 0);
    ptre_global.rdma_mgr->SetRemoteAddress(
        rank, type, name, addr.remote_addr, addr.rkey);
  }
  return addr;
}

void PrepareRdmaPush(RdmaEntry* entry) {
  /*
  // Random peer
  if (peer_sel_cnt == 0) {
    //LOG(INFO) << "DEBUG: " << ptre_global.num_tvars;
    std::uniform_int_distribution<int> distribution(0, ptre_size() - 1);
    int peer;
    while (!ptre_global.shutdown) {
      int peer = distribution(gen);
      if (peer == ptre_rank()) continue;
      GrpcClient* client;
      ptre_global.grpc_client_cache->GetClient(peer, &client);
      if (client->AttemptPush()) break;
    }
    peer_selected = peer;
  }
  entry->rank = peer_selected;
  peer_sel_cnt = (peer_sel_cnt + 1) % ptre_global.num_tvars;
  */
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
    state_mr = ptre_global.rdma_mgr->RegisterMR(BUF_TYPE_RECVBUF_STATE_WRITE,
        entry->tensor_name, writebuf, sizeof(int), IBV_ACCESS_LOCAL_WRITE);
  }
  entry->state_mr = state_mr;

  // Remote address of RECVBUF
  entry->tensor_addr = GetOrRetrieveRemoteAddress(entry->rank,
      BUF_TYPE_RECVBUF, entry->tensor_name);

  // Remote address of RECVBUF_STATE
  auto state_addr = GetOrRetrieveRemoteAddress(
      entry->rank, BUF_TYPE_RECVBUF_STATE, entry->tensor_name);
  entry->state_addr = state_addr;

  // Remote tensor ID number
  // TODO: Do all workers always use the same id??
  entry->tensor_id = ptre_global.id_table[entry->tensor_name];

  entry->channel = ptre_global.rdma_mgr->GetChannel(entry->rank);
}

// --------------------------------------------------------------------------

void CheckBufferTable(std::deque<MemcpyRequest>& htod_reqs);

void CheckBcastDone(std::deque<MemcpyRequest>& dtoh_reqs);

void TryOpenCommbuf();

void BackgroundThread() {
  // Operations:
  //  - PollCQ for Recv WC (Remote Push)
  //  - Average
  //  - MemcpyHtoD
  //  - MemcpyDtoH
  //  - Push
  //    - post_send_wr
  //  - PollCQ for Push WC
  //
  // Operations in one step:
  //  ComputeGradient -> MaybeWait -> ApplyGradient \          -> Proceed
  //  Recv -> Average -> MemcpyHtoD /               MemcpyDtoH -> Push

  DVLOGR(0, ptre_rank()) << __FUNCTION__ << "] Started.";
  // RDMA Channels
  std::vector<RdmaChannel*> channels;
  channels.reserve(ptre_size());
  for (int dst = 0; dst < ptre_size(); dst++) {
    channels[dst] = ptre_global.rdma_mgr->GetChannel(dst);
  }
  // WCs
  int num_wcs = 0;
  struct ibv_wc wcs[MAX_CQE_DEFAULT];

  // Grpc Channels
  std::vector<GrpcClient*> grpc_clients;
  grpc_clients.reserve(ptre_size());
  for (int dst = 0; dst < ptre_size(); dst++) {
    GrpcClient* client;
    ptre_global.grpc_client_cache->GetClient(dst, &client);
    grpc_clients[dst] = client;
  }

  while (!ptre_global.shutdown) {
    // MemcpyHtoD
    std::deque<MemcpyRequest> htod_reqs;
    std::deque<MemcpyRequest> htod_finals;
    ptre_global.htod_mu.lock();
    ptre_global.htod_queue.swap(htod_reqs);
    ptre_global.htod_mu.unlock();
    CheckBufferTable(htod_reqs);
    for (auto& req : htod_reqs) {
      OpTrace("TFOp", req.key, "");
      OpTrace("TFOp", req.key, "<-DoneComputeGrad");
      auto sm = ptre_global.recvbuf_table[req.key].second;
      std::lock_guard<std::mutex> guard(sm->mu);
      if (sm->state == RECVBUF_STATE_MEMCPY_READY) {
#ifdef ATOMIC_MODEL
        {
          std::lock_guard<std::mutex> guard(ptre_global.htod_mu);
          bool skip = false;
          if (ptre_global.htod_ever_skipped) {
            skip = true;
          } else {
            ptre_global.htod_ever_performed = true;
          }
          ptre_global.htod_cnt++;
          if (ptre_global.htod_cnt == ptre_global.num_tvars) {
            ptre_global.htod_ever_performed = false;
            ptre_global.htod_ever_skipped = false;
            ptre_global.htod_cnt = 0;
          }
          if (skip) {
            sm->state = RECVBUF_STATE_READY;
            req.callback(Status::OK());
            TryOpenCommbuf();
            continue;
          }
        }
#endif
        htod_finals.push_back(std::move(req));
      } else {
#ifdef SKIP_HTOD_IF_NOT_READY
#ifdef ATOMIC_MODEL
        {
          std::lock_guard<std::mutex> guard(ptre_global.htod_mu);
          if (ptre_global.htod_ever_performed) {
            // Wait
            ptre_global.htod_queue.push_back(std::move(req));
            continue;
          }
          ptre_global.htod_cnt++;
          ptre_global.htod_ever_skipped = true;
          if (ptre_global.htod_cnt == ptre_global.num_tvars) {
            ptre_global.htod_ever_performed = false;
            ptre_global.htod_ever_skipped = false;
            ptre_global.htod_cnt = 0;
          }
        }
#endif
        sm->state = RECVBUF_STATE_READY;
        req.callback(Status::OK());
        TryOpenCommbuf();
#else
        ptre_global.htod_mu.lock();
        ptre_global.htod_queue.push_back(std::move(req));
        ptre_global.htod_mu.unlock();
#endif
      }
    }
    for (auto& req : htod_finals) {
      auto recvbuf = ptre_global.recvbuf_table[req.key].first;
      auto sm = ptre_global.recvbuf_table[req.key].second;
      auto callback = req.callback;
//DBGR(req.key, ptre_rank()) << "HtoD";
      //auto name = ToPersistentCStr(req.key);
      auto& name = req.key;
      MemcpyHostToDevice(req.context, recvbuf, req.tensor,
          [sm, callback, name](const Status& s) {
            std::lock_guard<std::mutex> guard(sm->mu);
            sm->state = RECVBUF_STATE_READY;
            callback(s);
            TryOpenCommbuf();
          });
#ifdef SIMPLE_HTOD_CNT
      int from_rank = recent_avg_enqueue_rank[name];
      if (from_rank >= 0) {
        simple_htod_cnt[name][from_rank]++;
      }
#endif
    }

    // MemcpyDtoH
    std::deque<MemcpyRequest> dtoh_reqs;
    std::deque<MemcpyRequest> dtoh_finals;
    ptre_global.dtoh_mu.lock();
    ptre_global.dtoh_queue.swap(dtoh_reqs);
    ptre_global.dtoh_mu.unlock();
    CheckBcastDone(dtoh_reqs);

    // Wait if pushing
    {
      std::lock_guard<std::mutex> dtoh_guard(ptre_global.dtoh_mu);
      for (auto& req : dtoh_reqs) {
        auto sm = ptre_global.sendbuf_table[req.key].second;
        std::lock_guard<std::mutex> guard(sm->mu);
        if (sm->state == SENDBUF_STATE_BUSY) {
#ifdef SKIP_DTOH_IF_NOT_READY
          req.callback(Status(::tensorflow::error::Code::UNKNOWN, "skip"));
#else
          ptre_global.dtoh_queue.push_back(std::move(req));
#endif
        } else {
          sm->state = SENDBUF_STATE_BUSY;
          dtoh_finals.push_back(std::move(req));
        }
      }
    }
    for (auto& req : dtoh_finals) {
      auto sendbuf = ptre_global.sendbuf_table[req.key].first;
      auto callback = req.callback;
      auto& name = req.key;
      MemcpyDeviceToHost(req.context, req.tensor, sendbuf,
          [callback, name](const Status& s) {
            callback(s);
          });
    }

    // Push
    std::deque<RdmaEntry*> entries;
    ptre_global.push_mu.lock();
    ptre_global.push_queue.swap(entries);
    ptre_global.push_mu.unlock();
#if 1
    {
      std::lock_guard<std::mutex> guard(push_cnt_mu);
      for (auto entry : entries) {
        if (entry->rank < 0) {
          entry->tensor_state->state = SENDBUF_STATE_READY;
          delete entry;
          continue;
        }
        PrepareRdmaPush(entry);
#if 0
        if (push_cnts.find(entry->tensor_name) == push_cnts.end()) {
          push_cnts.emplace(entry->tensor_name, 0);
          //RdmaWrite(entry, false);
          //RdmaWrite(entry);
        } else {
          RdmaWrite(entry);
        }
        push_cnts[entry->tensor_name]++;
  //DBGR(entry->tensor_name, ptre_rank()) << "PUSH";
#else
        RdmaWrite(entry);
#endif
      }
    }

    // Poll Send WC
    for (int dst = 0; dst < ptre_size(); dst++) {
      channels[dst]->PollSendCQ(wcs, &num_wcs, false);
      for (int i = 0; i < num_wcs; i++) {
        assert(wcs[i].status == IBV_WC_SUCCESS);
        RdmaEntry* entry = reinterpret_cast<RdmaEntry*>(wcs[i].wr_id);
        std::lock_guard<std::mutex> guard(entry->tensor_state->mu);
        entry->tensor_state->state = SENDBUF_STATE_READY;
        delete entry;
      }
    }
#endif

    // Recv Remote Write
    //std::vector<uint32_t> ids;
    std::vector<std::pair<uint32_t, int>> ids2;  // tensor_id, from_rank
    for (int dst = 0; dst < ptre_size(); dst++) {
      channels[dst]->PollRecvCQ(wcs, &num_wcs, false);
      for (int i = 0; i < num_wcs; i++) {
        assert(wcs[i].status == IBV_WC_SUCCESS);
        RdmaRecvEntry* entry = reinterpret_cast<RdmaRecvEntry*>(wcs[i].wr_id);
        uint32_t id = ntohl(wcs[i].imm_data);
        //ids.push_back(ntohl(wcs[i].imm_data));
        ids2.emplace_back(id, dst);
        delete entry;
      }
      for (int i = 0; i < num_wcs; i++) {
        PostRecvTensorIdNumber(dst);
      }
    }

    // Average
#if 1
    for (auto& e : ids2) {
      EnqueueAvgWithTensorIdNumber(e.first, e.second);
    }
#elif 0
    for (auto id : ids) {
      string& name = ptre_global.id_to_name[id];
      // DO IT IF NOT SKIPPED
      auto sendbuf = ptre_global.sendbuf_table[name].first;
      auto recvbuf = ptre_global.recvbuf_table[name].first;
      float* sdata = (float*) const_cast<char*>(sendbuf->tensor_data().data());
      float* rdata = (float*) const_cast<char*>(recvbuf->tensor_data().data());
      for (int i = 0; i < recvbuf->NumElements(); i++) {
        rdata[i] = (rdata[i] + sdata[i]) / 2;
      }
      auto sm = ptre_global.recvbuf_table[name].second;
      sm->state = RECVBUF_STATE_MEMCPY_READY;
    }
#elif 1
    for (auto id : ids) {
      string& name = ptre_global.id_to_name[id];
      auto sm = ptre_global.recvbuf_table[name].second;
      sm->state = RECVBUF_STATE_MEMCPY_READY;
    }
#else
    for (auto id : ids) {
      string& name = ptre_global.id_to_name[id];
      auto sm = ptre_global.recvbuf_table[name].second;
      sm->state = RECVBUF_STATE_READY;
    }
#endif
  }
  //DVLOGR(0, ptre_rank()) << __FUNCTION__ << "] End.";
}

void TryOpenCommbuf() {
  await_mu.lock();
  await_cnt++;
  if (await_cnt == ptre_global.num_tvars) {
    ptre_global.commbuf_state->store(COMMBUF_STATE_IDLE);
    await_cnt = 0;
  }
  await_mu.unlock();
}

void CheckBufferTable(std::deque<MemcpyRequest>& htod_reqs) {
  std::deque<MemcpyRequest> reqs;
  for (auto& req : htod_reqs) {
    auto search = ptre_global.recvbuf_table.find(req.key);
    if (search == ptre_global.recvbuf_table.end()) {
      recent_avg_enqueue_rank[req.key] = -1;
      simple_htod_cnt[req.key] = new int[ptre_size()];
      memset(simple_htod_cnt[req.key], 0, ptre_size() * sizeof(int));
      ptre_global.htod_cnts[req.key] = 0;
      // Allocate new sendbuf and recvbuf and their states
      auto new_recvbuf = std::make_shared<Tensor>(
          req.tensor->dtype(), req.tensor->shape());
      auto new_recvbuf_state = std::make_shared<StateMutex>();
      new_recvbuf_state->state = RECVBUF_STATE_READY;
      ptre_global.recvbuf_table.emplace(
          req.key, TensorState(new_recvbuf, new_recvbuf_state));
      auto new_sendbuf = std::make_shared<Tensor>(
          req.tensor->dtype(), req.tensor->shape());
      auto new_sendbuf_state = std::make_shared<StateMutex>();
      ptre_global.sendbuf_table.emplace(
          req.key, TensorState(new_sendbuf, new_sendbuf_state));
      ptre_global.commbuf_table_mu.unlock();
      uint32_t new_id = ptre_global.id_table.size();
      ptre_global.id_to_name.emplace(new_id, req.key);
      ptre_global.id_table.emplace(req.key, new_id);
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
//DBGR(req.key, ptre_rank()) << "NO BUFFER";
      req.callback(Status(::tensorflow::error::Code::UNKNOWN, "skip"));
    } else {
      if (!ptre_global.num_tvars_initialized) {
        LOG(INFO) << "DEBUG, num_tvars=" << ptre_global.htod_cnts.size();
        ptre_global.num_tvars = ptre_global.htod_cnts.size();
        ptre_global.num_tvars_initialized = true;
      }
      ptre_global.htod_cnts[req.key]++;
      if (ptre_global.htod_cnts[req.key] < 2) {
//DBGR(req.key, ptre_rank()) << "CNT < 2";
        req.callback(Status(::tensorflow::error::Code::UNKNOWN, "skip"));
      } else {
        reqs.push_back(std::move(req));
      }
    }
  }
  htod_reqs.swap(reqs);
}

void CheckBcastDone(std::deque<MemcpyRequest>& dtoh_reqs) {
  std::lock_guard<std::mutex> guard(ptre_global.bcast_mu);
  std::deque<MemcpyRequest> reqs;
  for (auto& req : dtoh_reqs) {
#if 1
    if (ptre_global.bcast_done.find(req.key) == ptre_global.bcast_done.end()) {
//DBGR(req.key, ptre_rank()) << "BCAST NOT DONE";
      req.callback(Status(::tensorflow::error::Code::UNKNOWN, "skip"));
#else
    if (push_cnts.find(req.key) == push_cnts.end()) {
      push_cnts.emplace(req.key, 0);
    } else {
      push_cnts[req.key]++;
    }
    if (push_cnts[req.key] < 3) {
      req.callback(Status(::tensorflow::error::Code::UNKNOWN, "skip"));
#endif
    } else {
      reqs.push_back(std::move(req));
    }
  }
  dtoh_reqs.swap(reqs);
}

}  // namespace common
}  // namespace ptre
