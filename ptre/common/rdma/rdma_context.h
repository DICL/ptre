#ifndef PTRE_COMMON_RDMA_RDMA_CONTEXT_H_
#define PTRE_COMMON_RDMA_RDMA_CONTEXT_H_

#include <map>

#include <infiniband/verbs.h>

#include "ptre/common/communication/rdma/rdma_mgr.h"
#include "ptre/common/communication/rdma/rdma_channel.h"

namespace ptre {
namespace common {

class MRCache {
 public:
  struct ibv_mr* RegisterSendMR(struct ibv_pd* pd, void* buf, size_t length);
  struct ibv_mr* RegisterRecvMR(struct ibv_pd* pd, void* buf, size_t length);
  void DeregisterSendMR(void* sendbuf);
  void DeregisterRecvMR(void* recvbuf);
  bool HasSendMR(void* buf);
  bool HasRecvMR(void* buf);
  struct ibv_mr* send_mr(const void* buf);
  struct ibv_mr* recv_mr(const void* buf);

 private:
  std::map<const void*, struct ibv_mr*> send_mr_table_;
  std::map<const void*, struct ibv_mr*> recv_mr_table_;
};

class RdmaContext {
 public:
  enum RemoteAddrType {
    REMOTE_ADDR_ALLREDUCE_INTERMEDIATE_BUF,
    REMOTE_ADDR_ALLREDUCE_RECV_BUF
  };
  RdmaContext(RdmaMgr* rdma_mgr, struct ibv_mr* send_mr = NULL,
              struct ibv_mr* recv_mr = NULL);
  struct ibv_pd* pd() { return pd_; }
  int comm_rank() { return comm_rank_; }
  int comm_size() { return comm_size_; }
  RdmaChannel* get_channel(int comm_rank);
  void RegisterSendBuffer(void* sendbuf, size_t length);
  struct ibv_mr* RegisterRecvBuffer(void* recvbuf, size_t length);
  void DeregisterSendBuffer(void* sendbuf);
  void DeregisterRecvBuffer(void* recvbuf);
  struct ibv_mr* send_mr(const void* buf);
  struct ibv_mr* recv_mr(const void* buf);
  void allreduce_set_intermediate_buf(const void* ptr, char* inbuf);
  char* allreduce_intermediate_buf(const void* ptr);
  void set_remote_addr(int type, const void* ptr,
                       const RemoteAddr& ra);
  int get_remote_addr(int type, const void* ptr, RemoteAddr* out);

 protected:
  int comm_rank_;
  int comm_size_;
  struct ibv_pd* pd_;
  std::map<int, RdmaChannel*> channel_table_;
  MRCache mr_cache_;
  std::map<const void*, char*> inbuf_table_;
  std::map<int, std::map<const void*, RemoteAddr>> remote_addr_table_;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_RDMA_RDMA_CONTEXT_H_
