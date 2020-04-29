#ifndef PTRE_COMMUNICATION_RDMA_RDMA_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_H_

#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <iostream>

#include <infiniband/verbs.h>

#include "tensorflow/core/platform/logging.h"

namespace ptre {
#define IB_PORT 1
#define QUEUE_DEPTH_DEFAULT 1024
#define MAX_CONCURRENT_WRITES 1000
#define TIMEOUT_DEFAULT 14
#define RETRY_CNT_DEFAULT 7

using std::cerr;
using std::endl;

struct RemoteTensorId {
  int dst_rank;
  std::string name;
};
bool operator<(const RemoteTensorId& a, const RemoteTensorId& b) {
  return (a.dst_rank < b.dst_rank ||
          (a.dst_rank == b.dst_rank && a.name < b.name));
}

struct RdmaEnv {
  ibv_device **dev_list;
  ibv_context *context;
  ibv_pd *pd;
  ibv_port_attr port_attr;
  ibv_device_attr dev_attr;
  union ibv_gid gid;
};

enum RdmaWrIdType {
  RDMA_WRITE_ID_TENSOR_WRITE,
  RDMA_WRITE_ID_INCOMING_FLAG_WRITE,
  RDMA_WR_ID_READ_TWO,
  RDMA_WR_ID_CAS_TWO,
  RDMA_WR_ID_CAS_TENSOR_AGG_STATE,
  RDMA_WR_ID_WRITE_TENSOR_AGG_STATE,
  RDMA_WR_ID_CAS,
  RDMA_WR_ID_READ
};

class RdmaWrId {
 public:
  RdmaWrId(RdmaWrIdType write_type, void* write_context)
      : write_type(write_type), write_context(write_context) {}

  RdmaWrIdType write_type;
  void* write_context;
};

struct RemoteMR {
  uint64_t remote_addr;
  //size_t length;
  uint32_t rkey;
};

//struct RdmaTensorBuf {
//  void* buf;
//  size_t length;
//  uint64_t state;
//};

//struct SRdmaTensorChannel {
//  ibv_qp* qp;
//  uint32_t qpn;
//  uint32_t lkey;
//  RemoteMR rmr;
//};

/// local_tensor_send_buf ----- remote_tensor_recv_buf
///                       \---- remote_tensor_recv_buf
/// local_tensor_recv_buf ----- remote_tensor_send_buf
///                       \---- remote_tensor_send_buf
class RemoteTensorChannel {
 private:
  ibv_mr* mr_;
  RemoteMR rmr_;
};

int init_rdma_env(RdmaEnv& env);
struct ibv_cq* ptre_rdma_create_cq(RdmaEnv* rdma_env, int comp_vector);
struct ibv_qp* ptre_rdma_create_qp(RdmaEnv* rdma_env, struct ibv_cq* send_cq,
    struct ibv_cq* recv_cq);
int ptre_rdma_connect_qp(struct ibv_qp* qp, uint32_t dest_qp_num,
    uint64_t global_subnet_prefix, uint64_t global_interface_id, uint16_t dlid,
    uint32_t my_psn, uint32_t remote_psn);
int ptre_rdma_connect_qp_local(struct ibv_qp* qp, uint32_t dest_qp_num,
    uint16_t dlid,
    uint32_t my_psn, uint32_t remote_psn);
void rdma_qp_reset_to_rts(struct ibv_qp* qp, uint32_t remote_qpn,
    uint16_t remote_lid, uint32_t remote_psn = 0, uint32_t my_psn = 0);

void ptre_poll_cq(struct ibv_cq* cq, int num_comps,
                                struct ibv_wc* wcs, int caller_id = 0);

int post_write(size_t buffer_size, uint64_t src_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               ibv_qp* qp);
int post_fetch_and_add(size_t buffer_size, uint64_t src_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               struct ibv_qp *qp);
int post_atomic_cmp_and_swp(size_t buffer_size,
    uint64_t local_addr,
    uint32_t lkey,
    uint64_t remote_addr,
               uint32_t rkey,
               struct ibv_send_wr& wr,
               uint64_t wr_id,
               struct ibv_qp *qp, uint64_t compare_add, uint64_t swap);
int post_atomic_add(size_t buffer_size, uint64_t src_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               struct ibv_qp *qp,
               uint64_t compare_add, uint64_t swap);
int post_read(size_t buffer_size, uint64_t local_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               struct ibv_qp *qp);
void rdma_modify_qp_rts(struct ibv_qp* qp, uint32_t remote_qpn,
    uint32_t remote_psn, uint16_t remote_lid, uint32_t my_psn);
void rdma_poll_cq(struct ibv_cq* cq, int num_comps,
                                struct ibv_wc* wcs);
uint64_t rdma_cas(uint64_t compare, uint64_t swap, struct ibv_qp* qp,
    struct ibv_cq* cq, struct ibv_mr* read_buf_mr, uint64_t remote_addr,
    uint32_t rkey, struct ibv_pd* pd);
}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_H_
