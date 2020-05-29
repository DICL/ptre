#include "ptre/communication/rdma/rdma_task.h"

#include <arpa/inet.h>

#include "ptre/communication/push_variable.h"
#include "ptre/communication/rdma/rdma.h"

namespace ptre {

int RPNTask::PostRead() {
  state_ = STATE_READ;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) rmr_->addr;
  sge.length = rmr_->length;
  sge.lkey = rmr_->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = id_;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_permit_addr_;
  wr.wr.rdma.rkey = permit_rkey_;
  struct ibv_qp* qp = rdma_mgr_->qp(dst_);
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret) {
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": Failed to ibv_post_send";
    return 1;
  }
  return 0;
}

int RPNTask::PostWrite() {
  state_ = STATE_WRITE;

  auto pvar = rdma_mgr_->push_variable(var_name_);
  if (!pvar || !pvar->GetState()) {
    return 1;
  }
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) wmr_->addr;
  sge.length = wmr_->length;
  sge.lkey = wmr_->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = id_;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data = htonl(var_idx_);
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_var_addr_;
  wr.wr.rdma.rkey = var_rkey_;
  struct ibv_qp* qp = rdma_mgr_->qp(dst_);
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret) {
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": Failed to ibv_post_send";
    return 1;
  }
  return 0;
}

int RPNTask::state() {
  return state_;
}

int RPNTask::permit() {
  return permit_;
}

int RecvTask::PostRecv() {
  int ret;
  struct ibv_qp* qp = rdma_mgr_->qp(dst_);
  struct ibv_recv_wr* bad_wr;
  ret = ibv_post_recv(qp, &wr_, &bad_wr);
  if (ret) {
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": Failed to ibv_post_send";
    return 1;
  }
  return 0;
}

} // namespace ptre
