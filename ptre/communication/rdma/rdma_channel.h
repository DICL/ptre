#ifndef PTRE_COMMUNICATION_RDMA_RDMA_CHANNEL_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_CHANNEL_H_

#include <mutex>

#include "ptre/communication/rdma/rdma.h"


namespace ptre {

class RdmaChannel {
 public:
  RdmaChannel(struct ibv_context* ctx, struct ibv_qp* qp);
  int PostSend(struct ibv_send_wr& wr);
  int PostRecv(struct ibv_recv_wr& wr);

 private:
  //int dst_;
  struct ibv_context* ctx_;
  std::mutex mu_;
  struct ibv_qp* qp_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_CHANNEL_H_
