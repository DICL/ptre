#ifndef PTRE_COMMON_PTRE_GLOBAL_H_
#define PTRE_COMMON_PTRE_GLOBAL_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "ptre/common/buffer_table.h"
#include "ptre/common/common.h"
#include "ptre/common/message.h"
#include "ptre/common/cm/consensus_manager.h"
#include "ptre/common/communication/grpc/grpc_client_cache.h"
#include "ptre/common/communication/rdma/grpc_client.h"
#include "ptre/common/communication/rdma/grpc_server.h"
#include "ptre/common/communication/rdma/pull_job.h"
#include "ptre/common/communication/rdma/rdma_mgr.h"
#include "ptre/common/communication/rdma/rdma_task.h"
#include "ptre/common/communication/tcp/tcp_grpc_client.h"
#include "ptre/common/communication/tcp/tcp_service_impl.h"
#include "ptre/common/rdma/rdma_context.h"
#include "third_party/minitrace/minitrace.h"

namespace ptre {
namespace common {

using std::string;
using MessageTable =
    std::unordered_map<string, std::unordered_map<int, Request>>;
using TensorTable = std::unordered_map<string, TensorTableEntry>;
using TensorState =
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<StateMutex>>;
using CommBufTable = std::unordered_map<string, TensorState>;

struct PtreGlobal {
  std::shared_ptr<std::atomic<int>> commbuf_state;
  bool num_tvars_initialized = false;
  int num_tvars = -1;

  std::unordered_map<string, std::unordered_map<string, int>> op_tracers;

  std::mutex htod_mu;
  std::mutex dtoh_mu;
  std::deque<MemcpyRequest> htod_queue;
  std::deque<MemcpyRequest> dtoh_queue;
  std::unordered_map<string, int> htod_cnts;

  std::mutex bcast_mu;
  std::unordered_map<string, bool> bcast_done;

#ifdef ATOMIC_MODEL
  //std::mutex htod_mu;
  int num_htod = 0;
  int htod_cnt = 0;
  bool htod_ever_skipped = false;
  bool htod_ever_performed = false;
#endif

  std::thread memcpy_thread;
  std::deque<MemcpyRequest> memcpy_queue;
  std::mutex memcpy_mu;
  TensorTable memcpy_table;

  std::mutex commbuf_table_mu;
  CommBufTable sendbuf_table;
  CommBufTable recvbuf_table;

  std::mutex id_mu;
  // unique uint32_t id to tensor name
  std::unordered_map<uint32_t, string> id_to_name;
  std::unordered_map<string, uint32_t> id_table;

  std::shared_ptr<BufferTable> buf_table;

  std::mutex push_mu;
  std::thread push_thread;
  std::deque<RdmaEntry*> push_queue;

  std::deque<string> pull_queue;
  std::mutex pull_mu;
  TensorTable pull_table;

  std::thread enq_avg_thread;

  std::vector<std::thread> avg_threads;
  std::thread avg_thread;
  std::mutex avg_mu;
  std::condition_variable avg_cv;
  std::queue<string> avg_queue;

  std::thread polling_thread;
  std::thread polling_recv_thread;

  ConsensusManager* cm = nullptr;
  RdmaMgr* rdma_mgr = nullptr;
  RdmaContext* rdma_ctx = nullptr;
  std::mutex mu;
  std::mutex mu_modelaverage;
  std::mutex mu_pull;

  std::queue<Request> message_queue;
  std::queue<Request> message_queue_modelaverage;
  std::queue<Request> message_queue_pull;
  MessageTable message_table;
  TensorTable tensor_table;
  TensorTable tensor_table_modelaverage;
  std::thread background_thread;
  std::thread background_thread_modelaverage;
  std::thread background_thread_pull;

  std::vector<std::thread> polling_threads;


  std::mutex q_mu;
  std::queue<int> q;
  //std::queue<std::shared_ptr<PushRequest>> req_q;
  //std::mutex req_q_mu;

  // Task Oriented
  //std::mutex push_q_mu;
  //std::vector<int> push_dsts;
  //std::queue<std::shared_ptr<PushTaskV2>> push_q;

  // Grpc Service
  RdmaServiceImpl grpc_service;
  // Tcp Grpc Service
  TcpServiceImpl tcp_grpc_service;
  // Grpc Server
  std::unique_ptr<grpc::Server> grpc_server = nullptr;
  std::atomic<bool> shutdown;
  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::vector<std::thread> push_threads;
  std::vector<std::thread> send_polling_threads;
  std::vector<std::thread> recv_polling_threads;
  //std::vector<std::thread> polling_threads;
  std::vector<std::thread> aggregation_threads;
  //std::vector<std::thread> receive_threads;
  Eigen::ThreadPool* eigen_pool;
  Eigen::ThreadPool* agg_eigen_pool;
  Eigen::ThreadPool* reduce_eigen_pool;

  int size;
  int rank;
  int local_size;
  int local_rank;
  PtreWorker this_worker;
  std::vector<PtreWorker> workers;
  std::vector<PtreNode> nodes;
  std::vector<std::string> grpc_hosts;
  std::shared_ptr<GrpcClientCache<GrpcClient>> grpc_client_cache = nullptr;
  std::shared_ptr<GrpcClientCache<TcpGrpcClient>> tcp_grpc_client_cache =
      nullptr;

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

  //std::mutex push_mu;

  // Counter
  std::vector<std::vector<int>> rcv_cnts;
  std::atomic<int> agg_cnt_total;
  std::atomic<int> rcv_cnt_total;
  std::atomic<int> send_cnt_total;
  std::vector<std::map<string, int>> agg_cnts;
  int agg_cnts_last = 1;

  std::map<string, int> push_success_cnt;

  //std::vector<std::mutex*> qp_mus;

  std::mutex rpn_checker_mu;
  std::map<uint64_t, string> rpn_checker;

  std::map<int, PullJob*> pull_jobs;
  std::mutex job_table_mu;
  std::queue<PullTask*> agg_q;
  std::mutex agg_q_mu;
  std::map<int, std::map<string, uint64_t>> last_key;
  std::map<int, std::map<string, int>> peer_agg_cnt;

  PtreGlobal();
  ~PtreGlobal();
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_PTRE_GLOBAL_H_
