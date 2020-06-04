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
    if (ret == 12) {
      return 12;
    }
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": Failed to ibv_post_send, ret=" << ret;
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
    if (ret == 12) {
      return 12;
    }
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": Failed to ibv_post_send, ret=" << ret;
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

string RPNTask::var_name() {
  return var_name_;
}

int RecvTask::PostRecv() {
  int ret;
  struct ibv_qp* qp = rdma_mgr_->qp(dst_);
  struct ibv_recv_wr* bad_wr;
  ret = ibv_post_recv(qp, &wr_, &bad_wr);
  if (ret) {
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": Failed to ibv_post_recv, ret=" << ret;
    if (ret == 12) {
      struct ibv_qp_attr attr;
      struct ibv_qp_init_attr init_attr;
      int qp_ret = ibv_query_qp(qp, &attr, IBV_QP_STATE | IBV_QP_CAP,
          &init_attr);
      LOG(INFO) << "qp_state=" << attr.qp_state
          << ", max_send_wr=" << init_attr.cap.max_send_wr
          << ", max_recv_wr=" << init_attr.cap.max_recv_wr;
      usleep(1000 * 1000);
    }
    return 1;
  }
  return 0;
}

// PullTask
PullTask::PullTask(RdmaManager* rdma_mgr, int dst, RemoteVariable* var,
                   void* job_handle) {
  rdma_mgr_ = rdma_mgr;
  dst_ = dst;
  var_name_ = var->name();
  job_handle_ = job_handle;

  state_ = STATE_INIT;

  memset(&key_read_, 0, sizeof(key_read_));
  tensor_ = new Tensor(var->dtype(), var->shape());
  memset(&validation_read_, 0, sizeof(validation_read_));
  // MR
  key_mr_ = ibv_reg_mr(rdma_mgr_->pd(), (void*) &key_read_, sizeof(key_read_),
      IBV_ACCESS_LOCAL_WRITE);
  tensor_mr_ = ibv_reg_mr(rdma_mgr_->pd(),
      (void*) tensor_->tensor_data().data(), tensor_->AllocatedBytes(),
      IBV_ACCESS_LOCAL_WRITE);
  validation_mr_ = ibv_reg_mr(rdma_mgr_->pd(), (void*) &validation_read_,
      sizeof(validation_read_), IBV_ACCESS_LOCAL_WRITE);
  // Remote Addr
  rdma_mgr_->GetRemoteAddress(dst_, BUF_TYPE_PULL_KEY, var_name_,
      &key_remote_addr_, &key_rkey_);
  for (int i = 0; i < 2; i++) {
    rdma_mgr_->GetRemoteAddress(dst_, BUF_TYPE_PULL_TENSOR_A, var_name_,
        &tensor_remote_addrs_[i], &tensor_rkeys_[i]);
  }
}

PullTask::~PullTask() {
  ibv_dereg_mr(key_mr_);
  ibv_dereg_mr(tensor_mr_);
  ibv_dereg_mr(validation_mr_);
  delete tensor_;
}

int PullTask::GetState() {
  std::lock_guard<std::mutex> guard(mu_);
  return state_;
}

int PullTask::state() {
  return state_;
}

int PullTask::PostReadKey() {
  std::lock_guard<std::mutex> guard(mu_);
  if (state_ == STATE_STOPPED) {
    state_ = STATE_ABORTED;
    return 1;
  }
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) key_mr_->addr;
  sge.length = key_mr_->length;
  sge.lkey = key_mr_->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) this;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = key_remote_addr_;
  wr.wr.rdma.rkey = key_rkey_;
  state_ = STATE_KEY_READ;
  int ret = channel_->PostSend(wr);
  if (ret) {
    state_ = STATE_ABORTED;
    return 1;
  }
  return 0;
}

void PullTask::PostReadTensor() {
  std::lock_guard<std::mutex> guard(mu_);
  if (state_ == STATE_STOPPED) {
    state_ = STATE_ABORTED;
    return;
  }
  int indicator = key_read_.indicator;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) tensor_mr_->addr;
  sge.length = tensor_mr_->length;
  sge.lkey = tensor_mr_->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) this;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = tensor_remote_addrs_[indicator];
  wr.wr.rdma.rkey = tensor_rkeys_[indicator];
  state_ = STATE_TENSOR_READ;
  int ret = channel_->PostSend(wr);
  if (ret) {
    state_ = STATE_ABORTED;
  }
}

void PullTask::PostReadValidation() {
  std::lock_guard<std::mutex> guard(mu_);
  if (state_ == STATE_STOPPED) {
    state_ = STATE_ABORTED;
    return;
  }
  int indicator = key_read_.indicator;
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) validation_mr_->addr;
  sge.length = validation_mr_->length;
  sge.lkey = validation_mr_->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) this;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = key_remote_addr_ + sizeof(bool)
      + sizeof(uint64_t) * indicator;
  wr.wr.rdma.rkey = key_rkey_;
  state_ = STATE_KEY_READ;
  int ret = channel_->PostSend(wr);
  if (ret) {
    state_ = STATE_ABORTED;
  }
}

bool PullTask::IsTensorValid() {
  std::lock_guard<std::mutex> guard(mu_);
  if (state_ == STATE_STOPPED) {
    state_ = STATE_ABORTED;
    return false;
  }
  int indicator = key_read_.indicator;
  uint64_t key_before;
  if (indicator == 0) {
    key_before = key_read_.key_a;
  } else {
    key_before = key_read_.key_b;
  }
  bool ret = (key_before == validation_read_);
  if (ret) {
    state_ = STATE_VALID;
  } else {
    state_ = STATE_INVALID;
  }
  return ret;
}

Tensor* PullTask::tensor() {
  return tensor_;
}

} // namespace ptre
