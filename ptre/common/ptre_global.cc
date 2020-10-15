#include "ptre/common/ptre_global.h"

#include "third_party/minitrace/minitrace.h"

namespace ptre {
namespace common {

PtreGlobal::PtreGlobal() {
#ifdef MTR_ENABLED
  mtr_init("/tmp/ptre_trace.json");
#endif
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

  commbuf_state = std::make_shared<std::atomic<int>>(0);
}

PtreGlobal::~PtreGlobal() {
  //shutdown.store(true);
  shutdown = true;

  //LOG(INFO) << "Joining Background Thread";
  if (background_thread.joinable()) {
    //background_thread.detach();
    background_thread.join();
  }

  DVLOG(0) << "Shuttingdown Grpc Server";
  if (grpc_server != nullptr) {
    grpc_server->Shutdown();
  }

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

  avg_cv.notify_all();
  for (auto& t : avg_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
  //for (auto& t : receive_threads) {
  //  if (t.joinable()) {
  //    t.join();
  //  }
  //}
#if 0
  DVLOG(0) << "Delete EigenThreadpools";
  delete eigen_pool;
  delete agg_eigen_pool;
  delete reduce_eigen_pool;
#endif
  DVLOG(0) << "Destruction Done.";
  //if (qp_recover_thread.joinable()) {
  //  qp_recover_thread.join();
  //}
  /*
  if (rdma_mgr != nullptr) {
    delete rdma_mgr;
  }
  */
#ifdef MTR_ENABLED
  for (auto& cat : op_tracers) {
    for (auto& title : cat.second) {
      MTR_FINISH(cat.first.c_str(), title.first.c_str(), &title.second);
    }
  }
  mtr_flush();
  mtr_shutdown();
#endif
}

}  // namespace common
}  // namespace ptre
