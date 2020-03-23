#include "ptre/communication/rdma/rdma_agg_writer.h"

namespace ptre {

void cas() {
}

RdmaAggWriter::RdmaAggWriter(int dst_rank, struct ibv_pd* pd,
                struct ibv_qp* qp, struct ibv_cq* cq,
                const std::vector<string>& names,
                const std::vector<struct ibv_mr*>& agg_buf_state_rmrs,
                const std::vector<struct ibv_mr*>& agg_buf_rmrs,
                const std::vector<struct ibv_mr*>& send_buf_mrs)
    : dst_rank_(dst_rank), n_(names.size()), pd_(pd), qp_(qp), cq_(cq) {
  // Init names
  for (int i = 0; i < names.size(); i++) {
    const string& name = names[i];
    //names_.push_back(name);
    name_to_index_.emplace(name, i);
  }
  /// Init State Read Buf and its MR for CAS
  state_read_bufs_ = (uint64_t*) malloc(sizeof(uint64_t) * n_);
  for (int i = 0; i < n_; i++) {
    state_read_bufs_[i] = StatefulAggBuf::kRecvReady;
  }
  int ibv_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  for (int i = 0; i < n_; i++) {
    struct ibv_mr* state_read_buf_mr = ibv_reg_mr(pd_,
        (void*) (state_read_bufs_ + i), sizeof(uint64_t), ibv_access_flags);
    state_read_buf_mrs_.push_back(state_read_buf_mr);
  }
  /// Init Remote MRs
  for (int i = 0; i < n_; i++) {
    agg_buf_state_rmrs_.push_back(agg_buf_state_rmrs[i]);
    agg_buf_rmrs_.push_back(agg_buf_rmrs[i]);
    send_buf_mrs_.push_back(send_buf_mrs[i]);
  }
}

int RdmaAggWriter::WriteAggBuf(const string& name) {
  // TODO: Check send buf state
  int idx = name_to_index_[name];

  /// STEP 1: CAS
  /// Get State Read Buf MR
  struct ibv_mr* state_read_buf_mr = state_read_buf_mrs_[idx];
  uint64_t* state_read_buf = (uint64_t*) state_read_buf_mr->addr;
  /// Init Scatter & Gather List
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) state_read_buf_mr->addr;
  sge.length = state_read_buf_mr->length;
  sge.lkey = state_read_buf_mr->lkey;
  /// Get Remote AggBuf State Mr
  struct ibv_mr* state_rmr = agg_buf_state_rmrs_[idx];
  /// Init WR
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.atomic.remote_addr = (uint64_t) state_rmr->addr;
  wr.wr.atomic.compare_add = StatefulAggBuf::kRecvReady;
  wr.wr.atomic.swap = StatefulAggBuf::kRecvInProgress;
  wr.wr.atomic.rkey = state_rmr->rkey;
  /// Repeat CAS until we get kRecvReady
  bool cas_done = false;
  while (!cas_done) {
    wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WR_ID_CAS_TENSOR_AGG_STATE, nullptr);
    struct ibv_send_wr* bad_wr;
    int ret = ibv_post_send(qp_, &wr, &bad_wr);
    struct ibv_wc wc;
    ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
    if (*state_read_buf == StatefulAggBuf::kRecvReady) {
      cas_done = true;
    }
  }

  /// STEP 2: Write
  /// Get Local Send Buf MR
  struct ibv_mr* send_buf_mr = send_buf_mrs_[idx];
  /// Init Scatter & Gather List
  memset(&sge, 0, sizeof(struct ibv_sge));
  sge.addr = (uint64_t) send_buf_mr->addr;
  sge.length = send_buf_mr->length;
  sge.lkey = send_buf_mr->lkey;
  /// Get Remote AggBuf MR
  struct ibv_mr* rmr = agg_buf_rmrs_[idx];
  /// Init WR
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) new RdmaWrId(RDMA_WRITE_ID_TENSOR_WRITE, nullptr);
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uint64_t) rmr->addr;
  wr.wr.rdma.rkey = rmr->rkey;
  /// Write Send Buf to Remote AggBuf
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(qp_, &wr, &bad_wr);
  struct ibv_wc wc;
  ptre_poll_cq(cq_, 1, &wc);  // delete RdmaWrId
}

}  // namespcae ptre
