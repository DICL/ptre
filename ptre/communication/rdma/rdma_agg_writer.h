#ifndef PTRE_COMMUNICATION_RDMA_RDMA_AGG_WRITER_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_AGG_WRITER_H_

#include <vector>
#include <string>
#include <map>

#include <ptre/cm/tensor_aggregator.h>
#include <ptre/communication/rdma/rdma.h>

namespace ptre {

using std::string;

class RdmaAggWriter {
 public:
  RdmaAggWriter(int dst_rank, struct ibv_pd* pd,
                struct ibv_cq* cq, struct ibv_qp* qp,
                const std::vector<string>& names,
                const std::vector<RemoteMR>& agg_buf_state_rmrs,
                const std::vector<RemoteMR>& agg_buf_rmrs,
                const std::vector<struct ibv_mr*>& send_buf_mrs);
  int WriteToAggBuf(const string& name);

 private:
  int dst_rank_;
  int n_;
  std::map<string, int> name_to_index_;
  /// Protection Domain
  /// TODO: const?
  struct ibv_pd* pd_;
  /// Completion Queue
  /// TODO: const?
  struct ibv_cq* cq_;
  /// QP
  /// TODO: const?
  struct ibv_qp* qp_;
  /// State Read Bufs
  /// Owned. No need of exchanging rkeys.
  uint64_t* state_read_bufs_;
  std::vector<struct ibv_mr*> state_read_buf_mrs_;
  /// Remote AggBuf State MRs
  std::vector<RemoteMR> agg_buf_state_rmrs_;
  /// Remote AggBufs
  std::vector<RemoteMR> agg_buf_rmrs_;
  /// Local Send Bufs
  std::vector<struct ibv_mr*> send_buf_mrs_;
};

}  // namespcae ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_AGG_WRITER_H_
