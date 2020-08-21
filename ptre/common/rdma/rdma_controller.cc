#include "ptre/common/common.h"
#include "ptre/common/communication/rdma/rdma.h"

#include <arpa/inet.h>

namespace ptre {
namespace common {

Status PostRecvWithImm(RdmaRecvEntry* entry) {
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) entry;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  int ret = entry->channel->PostRecv(wr);
  assert(ret == 0);
  return Status::OK();
}

Status RdmaRead(RdmaEntry* entry) {
#if 0
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) entry->state_mr->addr;
  sge.length = sizeof(int);
  sge.lkey = entry->state_mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) entry;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = entry.read_addr->remote_addr;
  wr.wr.rdma.rkey = entry.read_addr->rkey;

  entry.channel->PostSend(wr);
#endif
  return Status(::tensorflow::error::Code::UNIMPLEMENTED, __PRETTY_FUNCTION__);
}

Status RdmaWrite(RdmaEntry* entry) {
  struct ibv_mr* mr;
  RemoteAddr addr;
  assert(entry->state == RDMA_OP_STATE_WRITE_TENSOR);
  switch (entry->state) {
    case RDMA_OP_STATE_WRITE_TENSOR:
      mr = entry->tensor_mr;
      addr = entry->tensor_addr;
      break;
    case RDMA_OP_STATE_WRITE_STATE:
      mr = entry->state_mr;
      addr = entry->state_addr;
      break;
    default:
      exit(1);
      break;
  }

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) mr->addr;
  sge.length = mr->length;
  sge.lkey = mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) entry;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(entry->tensor_id);
  wr.wr.rdma.remote_addr = addr.remote_addr;
  wr.wr.rdma.rkey = addr.rkey;

  entry->channel->PostSend(wr);

  return Status::OK();
}

}  // namespace common
}  // namespace ptre
