#include "ptre/common/ptre_global.h"

namespace ptre {
namespace common {

PtreGlobal::PtreGlobal() {
  ma_op_cnt = 0;
  ma_op_cnt2 = 0;
  reduce_op_cnt0 = 0;
  reduce_op_cnt1 = 0;
  for (int i = 0; i < num_copy_cnt; i++) {
    copy_cnt[i] = 0;
  }

  // Counters
  agg_cnt_total = 0;
  rcv_cnt_total = 0;
  send_cnt_total = 0;
}

PtreGlobal::~PtreGlobal() {
  shutdown = true;

  if (polling_threads.size() > 0) {
    DVLOG(0) << "Join Polling Threads(num_threads="<< polling_threads.size()
        << ")";
    for (auto& t: polling_threads) {
      if (t.joinable()) t.join();
    }
  }

  if (push_threads.size() > 0) {
    DVLOG(0) << "Join Push Threads: " << push_threads.size();
    for (auto& t : push_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  if (send_polling_threads.size() > 0) {
    DVLOG(0) << "Join Send Polling Threads: " << send_polling_threads.size();
    for (auto& t : send_polling_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  if (recv_polling_threads.size() > 0) {
    DVLOG(0) << "Join Recv Polling Threads: " << recv_polling_threads.size();
    for (auto& t : recv_polling_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  DVLOG(0) << "Join Grpc Server Thread";
  if (grpc_server_thread.joinable()) {
    grpc_server_thread.join();
  }

  if (aggregation_threads.size() > 0) {
    DVLOG(0) << "Join Aggregation Threads: " << aggregation_threads.size();
    for (auto& t : aggregation_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }
  //for (auto& t : receive_threads) {
  //  if (t.joinable()) {
  //    t.join();
  //  }
  //}
  DVLOG(0) << "Delete EigenThreadpools";
  delete eigen_pool;
  delete agg_eigen_pool;
  delete reduce_eigen_pool;
  DVLOG(0) << "Destruction Done.";
  //if (qp_recover_thread.joinable()) {
  //  qp_recover_thread.join();
  //}
  /*
  if (rdma_mgr != nullptr) {
    delete rdma_mgr;
  }
  */
}

}  // namespace common
}  // namespace ptre
