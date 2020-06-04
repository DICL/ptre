#include "ptre/communication/rdma/rdma_channel.h"

namespace ptre {

RdmaChannel::RdmaChannel(struct ibv_context* ctx, struct ibv_qp* qp) {
  ctx_ = ctx;
  qp_ = qp;
}

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
