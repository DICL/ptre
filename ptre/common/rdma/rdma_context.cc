#include "ptre/common/rdma/rdma_context.h"

namespace ptre {
namespace common {

RdmaContext::RdmaContext(RdmaMgr* rdma_mgr, struct ibv_mr* send_mr,
                         struct ibv_mr* recv_mr) {
  comm_size_ = rdma_mgr->comm_size();
  comm_rank_ = rdma_mgr->comm_rank();
  pd_ = rdma_mgr->pd();
  for (int i = 0; i < comm_size_; i++) {
    channel_table_[i] = rdma_mgr->GetChannel(i);
  }
  send_mr_ = send_mr;
  recv_mr_ = recv_mr;
}

RdmaChannel* RdmaContext::get_channel(int comm_rank) {
  auto search = channel_table_.find(comm_rank);
  assert(search != channel_table_.end());
  return search->second;
}

}  // namespace common
}  // namespace ptre
