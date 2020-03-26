#ifndef PTRE_COMMUNICATION_RDMA_RDMA_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_H_

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
  //ibv_cq* cq;
  //ibv_wc wc[MAX_CONCURRENT_WRITES * 2];
  //std::thread polling_thread;
};

enum RdmaWrIdType {
  RDMA_WRITE_ID_TENSOR_WRITE,
  RDMA_WRITE_ID_INCOMING_FLAG_WRITE,
  RDMA_WR_ID_READ_TWO,
  RDMA_WR_ID_CAS_TWO,
  RDMA_WR_ID_CAS_TENSOR_AGG_STATE
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

class RdmaTensorChannel {
 public:
  RdmaTensorChannel(const RdmaEnv* env, const RemoteTensorId& id);
  void Connect(uint32_t dlid);

 private:
  const RdmaEnv* env_;
  RemoteTensorId id_;
  ibv_qp* qp_ = nullptr;
  bool connected_ = false;
};

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

static inline void ptre_poll_cq(struct ibv_cq* cq, int num_comps,
                                struct ibv_wc* wcs) {
  int cnt = 0;
  while (cnt < num_comps) {
    struct ibv_wc& wc = wcs[cnt];
    int new_comps = ibv_poll_cq(cq, num_comps - cnt, &wc);
    if (new_comps > 0) {
      for (int i = 0; i < new_comps; i++) {
        struct ibv_wc& curr_wc = wcs[cnt + i];
        if (curr_wc.status < 0) {
          std::cerr << "Bad wc status " << curr_wc.status << endl;
        }
        RdmaWrId* wr_id = reinterpret_cast<RdmaWrId*>(curr_wc.wr_id);
        //std::cout << "WorkCompletion (RdmaWrIdType=" << wr_id->write_type
        //    << ")\n";
        delete wr_id;
      }
      cnt += new_comps;
    }
  }
}

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
}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_H_
