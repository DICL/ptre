#include "ptre/communication/rdma/rdma_channel.h"

namespace ptre {

int RdmaChannel::PostSend(struct ibv_send_wr& wr) {
  int ret;
  struct ibv_send_wr* bad_wr;
  mu_.lock();
  ret = ibv_post_send(qp_, &wr, &bad_wr);
  mu_.unlock();
  return ret;
}

int RdmaChannel::PostRecv(struct ibv_recv_wr& wr) {
  int ret;
  struct ibv_recv_wr* bad_wr;
  mu_.lock();
  ret = ibv_post_recv(qp_, &wr, &bad_wr);
  mu_.unlock();
  return ret;
}

}  // namespace ptre
