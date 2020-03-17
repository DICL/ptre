#include "ptre/communication/rdma/rdma_worker.h"

#include <infiniband/verbs.h>

namespace ptre {

/// Thread safe.
void RdmaWorker::ProcessTaskQueue() {
  RdmaTask* t;
  q_->wait_and_pop(t);
  if (t->type == CAS1) {
    int ret = ibv_post_send(t->qp, t->wr, &(t->bad_wr));
  } else if (t->type == CAS2) {
    struct ibv_send_wr* wr = t->wr;
    float* read_buf = (float*) wr->sg_list[0].addr;
    if (wr->wr.atomic.compare_add == *((uint64_t*) read_buf)) {
      /// Compare And Swap Successfully done.
      delete t;
    } else {
      /// Retry Compare And Swap.
      wr->wr.atomic.compare_add = *((uint64_t*) read_buf);
      float* sums = reinterpret_cast<float*>(&wr->wr.atomic.swap);
      sums[0] = read_buf[0] + t->flat[0];
      sums[1] = read_buf[1] + t->flat[1];
      int ret = ibv_post_send(t->qp, wr, &(t->bad_wr));
    }
  }
}

//void RdmaWorker::ProcessCompletionQueue() {
//}


}  // namespace ptre
