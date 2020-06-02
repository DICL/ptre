#ifndef PTRE_COMMUNICATION_RDMA_RDMA_TASK_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_TASK_H_

#include "ptre/communication/rdma/rdma.h"
#include "ptre/communication/rdma/rdma_manager.h"

namespace ptre {

// Read, Push and Notify
class RPNTask {
 public:
  // State of this task
  //  0: Init
  //  1: Read IP
  //  2: Read Done
  //  3: Write IP
  //  4: Write Done
  enum TaskState {
    STATE_INIT,
    STATE_READ,
    STATE_WRITE,
  };

  RPNTask(RdmaManager* rdma_mgr, int dst, const string& var_name) {
    id_ = (uint64_t) this;
    rdma_mgr_ = rdma_mgr;
    comm_rank_ = rdma_mgr_->rank();
    dst_ = dst;
    var_name_ = var_name;
    var_idx_ = rdma_mgr_->var_name_to_index(var_name);

    permit_ = -1;
    rmr_ = ibv_reg_mr(rdma_mgr_->pd(), (void*) &permit_, sizeof(permit_),
        IBV_ACCESS_LOCAL_WRITE);
    wmr_ = rdma_mgr_->GetMR(BUF_TYPE_SEND_BUF, var_name_);
    rdma_mgr_->GetRemoteAddress(dst_, BUF_TYPE_PUSH_PERMIT, var_name_,
        &remote_permit_addr_, &permit_rkey_);
    rdma_mgr_->GetRemoteAddress(dst_, BUF_TYPE_RECV_BUF, var_name_,
        &remote_var_addr_, &var_rkey_);

    state_ = STATE_INIT;
  }
  ~RPNTask() {
    ibv_dereg_mr(rmr_);
  }
  int PostRead();
  int PostWrite();
  int state();
  int permit();
  string var_name();

 private:
  uint64_t id_;
  //QPMgr* qp_mgr_;
  RdmaManager* rdma_mgr_;
  int comm_rank_;
  int dst_;
  string var_name_;
  int var_idx_;

  int permit_;
  struct ibv_mr* rmr_;
  struct ibv_mr* wmr_;
  uint64_t remote_permit_addr_;
  uint32_t permit_rkey_;
  uint64_t remote_var_addr_;
  uint32_t var_rkey_;

  int state_;
};

class RecvTask {
 public:
  RecvTask(RdmaManager* rdma_mgr, int dst) {
    id_ = (uint64_t) this;
    rdma_mgr_ = rdma_mgr;
    dst_ = dst;
    buf_ = malloc(1);
    mr_ = ibv_reg_mr(rdma_mgr_->pd(), buf_, 1, IBV_ACCESS_LOCAL_WRITE);
    memset(&sge_, 0, sizeof(sge_));
    sge_.addr = (uint64_t) mr_->addr;
    sge_.length = mr_->length;
    sge_.lkey = mr_->lkey;
    memset(&wr_, 0, sizeof(wr_));
    wr_.wr_id = id_;
    wr_.sg_list = &sge_;
    wr_.num_sge = 1;
  }
  ~RecvTask() {
    ibv_dereg_mr(mr_);
    free(buf_);
  }
  int PostRecv();

 private:
  uint64_t id_;
  RdmaManager* rdma_mgr_;
  int dst_;
  // WR
  void* buf_;
  struct ibv_mr* mr_;
  struct ibv_sge sge_;
  struct ibv_recv_wr wr_;
};

} // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_TASK_H_
