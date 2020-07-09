#ifndef PTRE_COMMON_PTRE_GLOBAL_H_
#define PTRE_COMMON_PTRE_GLOBAL_H_

#include <mutex>
#include <queue>
#include <vector>
#include <thread>
#include <atomic>

#include "ptre/common/cm/consensus_manager.h"
#include "ptre/common/communication/grpc/grpc_client_cache.h"
#include "ptre/common/communication/rdma/grpc_client.h"
#include "ptre/common/communication/rdma/grpc_server.h"
#include "ptre/common/communication/rdma/pull_job.h"
#include "ptre/common/communication/rdma/rdma_mgr.h"
#include "ptre/common/communication/rdma/rdma_task.h"
#include "ptre/common/message.h"
#include "ptre/common/rdma/rdma_context.h"

namespace ptre {
namespace common {

using std::string;
using MessageTable =
    std::unordered_map<string, std::unordered_map<int, Request>>;
//using MessageTable = std::unordered_map<string, std::vector<Request>>;

struct PtreGlobal {
  PtreGlobal();
  ~PtreGlobal();

  ConsensusManager* cm = nullptr;
  RdmaMgr* rdma_mgr = nullptr;
  RdmaContext* rdma_ctx = nullptr;
  std::mutex mu;

  std::queue<Request> message_queue;
  //std::unique_ptr<MessageTable> message_table;
  MessageTable message_table;
  std::thread background_thread;

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

  std::map<int, PullJob*> pull_jobs;
  std::mutex job_table_mu;
  std::queue<PullTask*> agg_q;
  std::mutex agg_q_mu;
  std::map<int, std::map<string, uint64_t>> last_key;
  std::map<int, std::map<string, int>> peer_agg_cnt;

  std::mutex tensor_table_mu;
  std::map<string, TensorTableEntry> tensor_table;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_PTRE_GLOBAL_H_
