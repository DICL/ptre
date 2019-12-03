#ifndef PTRE_COMMUNICATION_RDMA_RDMA_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include <infiniband/verbs.h>

namespace ptre {
#define IB_PORT 1
#define QUEUE_DEPTH_DEFAULT 1024
#define MAX_CONCURRENT_WRITES 1000
#define TIMEOUT_DEFAULT 14
#define RETRY_CNT_DEFAULT 7

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

enum RdmaWriteIDType {
  RDMA_WRITE_ID_TENSOR_WRITE,
  RDMA_WRITE_ID_INCOMING_FLAG_WRITE
};

class RdmaWriteID {
 public:
  RdmaWriteID(RdmaWriteIDType write_type, void* write_context)
      : write_type(write_type), write_context(write_context) {}

  RdmaWriteIDType write_type;
  void* write_context;
};

struct RemoteMR {
  uint64_t remote_addr;
  uint32_t rkey;
};

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

int post_write(size_t buffer_size, uint64_t src_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               ibv_qp* qp);
}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_H_
