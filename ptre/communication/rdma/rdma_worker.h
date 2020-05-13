#ifndef PTRE_COMMUNICATION_RDMA_RDMA_WORKER_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_WORKER_H_

#include <vector>
#include <infiniband/verbs.h>

namespace ptre {

class RdmaWorker {
 public:
  RdmaWorker() {
  }
  void PollReceiveCQ();

 private:
  std::vector<struct ibv_cq*> rcv_cqs_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_WORKER_H_
