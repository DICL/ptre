#include "ptre/common/rdma/rdma_context.h"

namespace ptre {
namespace common {

struct ibv_mr* MRCache::RegisterSendMR(struct ibv_pd* pd, void* buf,
                                       size_t length) {
  struct ibv_mr* mr = ibv_reg_mr(pd, buf, length, IBV_ACCESS_REMOTE_READ);
  send_mr_table_[buf] = mr;
  return mr;
}

struct ibv_mr* MRCache::RegisterRecvMR(struct ibv_pd* pd, void* buf,
                                       size_t length) {
  struct ibv_mr* mr = ibv_reg_mr(pd, buf, length,
      IBV_ACCESS_LOCAL_WRITE
      | IBV_ACCESS_REMOTE_WRITE);
  recv_mr_table_[buf] = mr;
  return mr;
}
void MRCache::DeregisterSendMR(void* sendbuf) {
  ibv_dereg_mr(send_mr(sendbuf));
}

void MRCache::DeregisterRecvMR(void* recvbuf) {
  ibv_dereg_mr(recv_mr(recvbuf));
}

bool MRCache::HasSendMR(void* buf) {
  return (send_mr_table_.find(buf) != send_mr_table_.end());
}

bool MRCache::HasRecvMR(void* buf) {
  return (recv_mr_table_.find(buf) != recv_mr_table_.end());
}

struct ibv_mr* MRCache::send_mr(const void* buf) {
  auto search = send_mr_table_.find(buf);
  if (search == send_mr_table_.end()) return NULL;
  return search->second;
}

struct ibv_mr* MRCache::recv_mr(const void* buf) {
  auto search = recv_mr_table_.find(buf);
  if (search == recv_mr_table_.end()) return NULL;
  return search->second;
}

RdmaContext::RdmaContext(RdmaMgr* rdma_mgr, struct ibv_mr* send_mr,
                         struct ibv_mr* recv_mr) {
  comm_size_ = rdma_mgr->comm_size();
  comm_rank_ = rdma_mgr->comm_rank();
  pd_ = rdma_mgr->pd();
  for (int i = 0; i < comm_size_; i++) {
    channel_table_[i] = rdma_mgr->GetChannel(i);
  }
}

RdmaChannel* RdmaContext::get_channel(int comm_rank) {
  auto search = channel_table_.find(comm_rank);
  if (search == channel_table_.end()) {
    LOG(ERROR) << "Channel not found: rank=" << comm_rank;
    return NULL;
  }
  return search->second;
}

void RdmaContext::RegisterSendBuffer(void* sendbuf, size_t length) {
  mr_cache_.RegisterSendMR(pd_, sendbuf, length);
}

struct ibv_mr* RdmaContext::RegisterRecvBuffer(void* recvbuf, size_t length) {
  return mr_cache_.RegisterRecvMR(pd_, recvbuf, length);
}

void RdmaContext::DeregisterSendBuffer(void* sendbuf) {
  mr_cache_.DeregisterSendMR(sendbuf);
}

void RdmaContext::DeregisterRecvBuffer(void* recvbuf) {
  mr_cache_.DeregisterRecvMR(recvbuf);
}

struct ibv_mr* RdmaContext::send_mr(const void* buf) {
  return mr_cache_.send_mr(buf);
}

struct ibv_mr* RdmaContext::recv_mr(const void* buf) {
  return mr_cache_.recv_mr(buf);
}

void RdmaContext::allreduce_set_intermediate_buf(const void* ptr, char* inbuf) {
  inbuf_table_[ptr] = inbuf;
}

char* RdmaContext::allreduce_intermediate_buf(const void* ptr) {
  auto search = inbuf_table_.find(ptr);
  if (search == inbuf_table_.end()) return NULL;
  return search->second;
}

void RdmaContext::set_remote_addr(int type, const void* ptr,
                                  const RemoteAddr& ra) {
  remote_addr_table_[type][ptr] = ra;
}

int RdmaContext::get_remote_addr(int type, const void* ptr, RemoteAddr* out) {
  auto search = remote_addr_table_[type].find(ptr);
  if (search == remote_addr_table_[type].end()) return 1;
  *out = search->second;
  return 0;
}

}  // namespace common
}  // namespace ptre
