#include "ptre/common/communication/rdma/rdma_channel.h"

namespace ptre {
namespace common {

RdmaChannel::RdmaChannel(struct ibv_context* ctx, struct ibv_qp* qp) {
  ctx_ = ctx;
  qp_ = qp;
}

int RdmaChannel::PostSend(struct ibv_send_wr& wr) {
  int ret;
  struct ibv_send_wr* bad_wr;
  mu_.lock();
  ret = ibv_post_send(qp_, &wr, &bad_wr);
  if (ret) {
    LOG(ERROR) << "Failed ibv_post_send @ " << __PRETTY_FUNCTION__
        << ": ret=" << ret;
    exit(1);
  }
  mu_.unlock();
  return ret;
}

int RdmaChannel::PostRecv(struct ibv_recv_wr& wr, const bool locking) {
  int ret;
  struct ibv_recv_wr* bad_wr;
  if (locking) mu_.lock();
  ret = ibv_post_recv(qp_, &wr, &bad_wr);
  if (ret) {
    LOG(ERROR) << "Failed ibv_post_recv @ " << __PRETTY_FUNCTION__
        << ": ret=" << ret;
    exit(1);
  }
  if (locking) mu_.unlock();
  return ret;
}

int PollCQInternal(struct ibv_cq* cq, struct ibv_wc* wcs, int* num_wcs) {
  *num_wcs = ibv_poll_cq(cq, MAX_CQE_DEFAULT, wcs);
  assert(*num_wcs >= 0);
  return 0;
}

int RdmaChannel::PollSendCQ(struct ibv_wc* wcs, int* num_wcs,
                            const bool use_locking) {
  if (use_locking) mu_.lock();
  std::lock_guard<std::mutex> guard(mu_);
  struct ibv_cq* cq = qp_->send_cq;
  int ret = PollCQInternal(cq, wcs, num_wcs);
  if (use_locking) mu_.unlock();
  return ret;
}

int RdmaChannel::PollRecvCQ(struct ibv_wc* wcs, int* num_wcs, bool locking) {
  if (locking) {
    mu_.lock();
  }
  struct ibv_cq* cq = qp_->recv_cq;
  int ret = PollCQInternal(cq, wcs, num_wcs);
  if (locking) {
    mu_.unlock();
  }
  return ret;
}

int RdmaChannel::Recover() {
  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
  int ret = ibv_query_qp(qp_, &attr,
        IBV_QP_STATE
      | IBV_QP_AV
      | IBV_QP_DEST_QPN,
      &init_attr);
  if (ret) {
    LOG(ERROR) << "Failed to query QP: ret=" << ret << ", "
        << std::strerror(ret);
    return 1;
  }
  if (attr.qp_state != IBV_QPS_RTS) {
    uint32_t dest_qp_num = attr.dest_qp_num;
    uint16_t dlid = attr.ah_attr.dlid;
    if (rdma_qp_reset_to_rts(qp_, dest_qp_num, dlid)) {
      return 1;
    }
  }
  return 0;
}

}  // namespace common
}  // namespace ptre
