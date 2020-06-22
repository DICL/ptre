#ifndef PTRE_COMMON_RDMA_RDMA_CONTEXT_H_
#define PTRE_COMMON_RDMA_RDMA_CONTEXT_H_

#include <map>

#include <infiniband/verbs.h>

#include "ptre/common/communication/rdma/rdma_mgr.h"
#include "ptre/common/communication/rdma/rdma_channel.h"

namespace ptre {
namespace common {

class RdmaContext {
 public:
  RdmaContext(RdmaMgr* rdma_mgr, struct ibv_mr* send_mr = NULL,
              struct ibv_mr* recv_mr = NULL);
  struct ibv_pd* pd() { return pd_; }
  int comm_rank() { return comm_rank_; }
  int comm_size() { return comm_size_; }
  RdmaChannel* get_channel(int comm_rank);
  struct ibv_mr* send_mr() { return send_mr_; }
  struct ibv_mr* recv_mr() { return recv_mr_; }

 protected:
  int comm_rank_;
  int comm_size_;
  struct ibv_pd* pd_;
  std::map<int, RdmaChannel*> channel_table_;
  struct ibv_mr* send_mr_;
  struct ibv_mr* recv_mr_;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_RDMA_RDMA_CONTEXT_H_
