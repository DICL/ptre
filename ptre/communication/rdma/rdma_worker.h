#ifndef PTRE_COMMUNICATION_RDMA_RDMA_WORKER_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_WORKER_H_

#include <memory>

#include "ptre/communication/rdma/rdma_task.h"
#include "ptre/lib/concurrent_queue.h"

namespace ptre {

class RdmaWorker {
 public:
  RdmaWorker(std::shared_ptr<ConcurrentQueue<RdmaTask*>> q) {
    q_ = q;
  }
  void ProcessTaskQueue();
 private:
  /// Not owned.
  std::shared_ptr<ConcurrentQueue<RdmaTask*>> q_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_WORKER_H_
