#ifndef PTRE_COMMON_COMMUNICATION_RDMA_RDMA_CHANNEL_H_
#define PTRE_COMMON_COMMUNICATION_RDMA_RDMA_CHANNEL_H_

#include <mutex>

#include "ptre/common/communication/rdma/rdma.h"


namespace ptre {
namespace common {

class RdmaChannel {
 public:
  RdmaChannel(struct ibv_context* ctx, struct ibv_qp* qp);
  int PostSend(struct ibv_send_wr& wr);
  int PostRecv(struct ibv_recv_wr& wr, const bool locking = true);
  int PollSendCQ(struct ibv_wc* wcs, int* num_wcs,
                 const bool use_locking = true);
  int PollRecvCQ(struct ibv_wc* wcs, int* num_wcs, bool locking = true);
  int Recover();

 private:
  //int dst_;
  struct ibv_context* ctx_;
  std::mutex mu_;
  struct ibv_qp* qp_;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_COMMUNICATION_RDMA_RDMA_CHANNEL_H_
