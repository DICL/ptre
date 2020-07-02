#include "ptre/common/rdma/rdma_mpi.h"

#include <unistd.h>
#include <arpa/inet.h>

#include "ptre/common/logging.h"
#include "ptre/common/message.h"


namespace ptre {
namespace common {

int RdmaWait(RdmaRequest* request, Status* status) {
  int ret, join_ret;
  join_ret = request->Join();
  struct ibv_mr* mr = request->mr();
  if (mr != NULL) {
    ret = ibv_dereg_mr(mr);
    if (ret) {
      LOG(ERROR) << "Failed to deregister MR @ " << __PRETTY_FUNCTION__;
      return 1;
    }
  }
  //ret = request->status();
  return join_ret;
}

int RdmaIsend(const void* buf, int count, DataType datatype, int dest, int tag,
              RdmaContext* ctx, RdmaRequest* request) {
  int ret;
  size_t length = count * DataType_Size(datatype);

  struct ibv_mr* mr = ctx->send_mr(buf);
  if (mr == NULL) {
    mr = ibv_reg_mr(ctx->pd(), const_cast<void*>(buf), length, 0);
    request->set_mr(mr);
  }

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) buf;
  sge.length = length;
  sge.lkey = mr->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) request;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  auto channel = ctx->get_channel(dest);
  ret = channel->PostSend(wr);
  return ret;
}

int RdmaSend(const void* buf, int count, DataType datatype, int dest, int tag,
             RdmaContext* ctx) {
  int ret;
  RdmaRequest request;
  ret = RdmaIsend(buf, count, datatype, dest, tag, ctx, &request);
  if (ret) {
    LOG(ERROR) << "RdmaIsend returned " << ret << " @ " << __PRETTY_FUNCTION__;
  }

  ret = RdmaWait(&request, NULL);
  if (ret) {
    LOG(ERROR) << "RdmaWait returned " << ret << " @ " << __PRETTY_FUNCTION__;
  }

  return 0;
}

int RdmaIrecv(void* buf, int count, DataType datatype, int source, int tag,
              RdmaContext* ctx, RdmaRequest* request) {
  int ret;
  size_t length = count * DataType_Size(datatype);

  struct ibv_mr* mr = ctx->recv_mr(buf);
//LOG(INFO) << __FUNCTION__ << "0000";
  if (mr == NULL) {
    mr = ibv_reg_mr(ctx->pd(), buf, length, IBV_ACCESS_LOCAL_WRITE);
    request->set_mr(mr);
  }

//LOG(INFO) << __FUNCTION__ << "1000";
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) buf;
  sge.length = length;
  sge.lkey = mr->lkey;
//LOG(INFO) << __FUNCTION__ << "2000";
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) request;
  wr.sg_list = &sge;
  wr.num_sge = 1;

//LOG(INFO) << __FUNCTION__ << "3000";
  auto channel = ctx->get_channel(source);
//LOG(INFO) << __FUNCTION__ << "4000";
  ret = channel->PostRecv(wr);
//LOG(INFO) << __FUNCTION__ << "5000";
  return ret;
}

int RdmaRecv(void* buf, int count, DataType datatype, int source, int tag,
             RdmaContext* ctx, Status* status) {
  int ret;
  RdmaRequest request;
  ret = RdmaIrecv(buf, count, datatype, source, tag, ctx, &request);
  if (ret) {
    LOG(ERROR) << "RdmaIrecv returned " << ret << " @ " << __PRETTY_FUNCTION__;
    return 1;
  }

  ret = RdmaWait(&request, status);
  if (ret) {
    LOG(ERROR) << "RdmaWait returned " << ret << " @ " << __PRETTY_FUNCTION__;
    return 1;
  }
  return 0;
}

int RdmaIwriteWithImm(const void* buf, uint32_t imm_data, RemoteAddr ra,
                     int count, DataType dtype, int dst, int tag,
                     RdmaContext* ctx, RdmaRequest* request,
                     struct ibv_mr* send_mr) {
  int ret;
  size_t length = count * DataType_Size(dtype);

  struct ibv_mr* mr = send_mr;
  if (mr == NULL) {
    mr = ctx->send_mr(buf);
    if (mr == NULL) {
      mr = ibv_reg_mr(ctx->pd(), const_cast<void*>(buf), length, 0);
      if (mr == NULL) {
        LOG(ERROR) << "Failed to register MR";
        return 1;
      }
      request->set_mr(mr);
    }
  }

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) buf;
  sge.length = length;
  sge.lkey = mr->lkey;
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) request;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = htonl(imm_data);
  wr.wr.rdma.remote_addr = ra.remote_addr;
  wr.wr.rdma.rkey = ra.rkey;

  auto channel = ctx->get_channel(dst);
  ret = channel->PostSend(wr);
  if (ret) {
    LOG(ERROR) << "Failed PostSend";
    return 1;
  }
  return 0;
}

int RdmaWriteWithImm(const void* buf, uint32_t imm_data, RemoteAddr ra,
                     int count, DataType dtype, int dst, int tag,
                     RdmaContext* ctx) {
  int ret;
  RdmaRequest req;
  ret = RdmaIwriteWithImm(buf, imm_data, ra, count, dtype, dst, tag, ctx, &req);
  if (ret) {
    LOG(ERROR) << "RdmaIwriteWithImm returned " << ret;
    return 1;
  }
  ret = RdmaWait(&req, NULL);
  if (ret) {
    LOG(ERROR) << "RdmaWait returned " << ret;
    return 1;
  }
  return 0;
}

int RdmaRecvWithImm(void* buf, uint32_t* out_imm_data, int count,
                    DataType dtype, int src, int tag, RdmaContext* ctx,
                    Status* status) {
  int ret;
  RdmaRequest request;
  size_t length = count * DataType_Size(dtype);

  struct ibv_mr* mr;
  if (buf != NULL) {
    mr = ctx->recv_mr(buf);
    if (mr == NULL) {
      mr = ibv_reg_mr(ctx->pd(), buf, length,
          IBV_ACCESS_LOCAL_WRITE
          | IBV_ACCESS_REMOTE_WRITE
          | IBV_ACCESS_REMOTE_READ);
      request.set_mr(mr);
    }
  }

  struct ibv_sge sge;
  if (buf != NULL) {
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t) buf;
    sge.length = length;
    sge.lkey = mr->lkey;
  }
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) &request;
  wr.sg_list = (buf != NULL) ? &sge : NULL;
  wr.num_sge = (buf != NULL) ? 1 : 0;

  auto channel = ctx->get_channel(src);
  ret = channel->PostRecv(wr);
  if (ret) {
    LOG(ERROR) << "PostRecv Failed @ " << __PRETTY_FUNCTION__
        << ": ret=" << ret;
    return 1;
  }

  ret = RdmaWait(&request, NULL);

  *out_imm_data = request.imm_data();
  return 0;
}


int RdmaSendrecv(const void* sendbuf, int sendcount, DataType sendtype,
                 int dest, int sendtag, void* recvbuf, int recvcount,
                 DataType recvtype, int source, int recvtag, RdmaContext* ctx,
                 Status* status) {
  int ret;
  RdmaRequest request;
  ret = RdmaIrecv(recvbuf, recvcount, recvtype, source, recvtag, ctx, &request);
  if (ret) {
    LOG(ERROR) << "RdmaIrecv returned " << ret << " @ " << __PRETTY_FUNCTION__;
  }

  ret = RdmaSend(sendbuf, sendcount, sendtype, dest, sendtag, ctx);
  if (ret) {
    LOG(ERROR) << "RdmaSend returned " << ret << " @ " << __PRETTY_FUNCTION__;
  }

  ret = RdmaWait(&request, status);
  if (ret) {
    LOG(ERROR) << "RdmaWait returned " << ret << " @ " << __PRETTY_FUNCTION__;
  }

  return 0;
}

// TODO: Optimize this using a tree structure
int RdmaBcast(void* buffer, int count, DataType datatype, int root,
              RdmaContext* ctx) {
  int ret, comm_rank, comm_size;
  comm_size = ctx->comm_size();
  comm_rank = ctx->comm_rank();

  if (comm_rank == root) {
    for (int i = 0; i < comm_size; i++) {
      if (i == comm_rank) continue;
      ret = RdmaSend(buffer, count, datatype, i, 0, ctx);
      if (ret) {
        LOG(ERROR) << "RdmaSend returned " << ret << " @ "
            << __PRETTY_FUNCTION__;
      }
    }
  } else {
    ret = RdmaRecv(buffer, count, datatype, root, 0, ctx, NULL);
    if (ret) {
      LOG(ERROR) << "RdmaRecv returned " << ret << " @ " << __PRETTY_FUNCTION__;
    }
  }

  return 0;
}

/*
int RdmaBarrier(RdmaContext* ctx) {
  int ret, my_rank, comm_size;
  comm_size = ctx->comm_size();
  if (comm_size == 1) return;

  my_rank = ctx->comm_rank();
  int mask = 0x1;
  while (mask < size) {
    int dst = (my_rank + mask) % size;
    RdmaSend(dst, NULL, 0, "PtreBarrier");
    int src = (my_rank - mask + size) % size;
    PtreRecv(src, NULL, 0, "PtreBarrier");
    mask <<= 1;
  }
}
*/



// TODO: Optimize this using a tree structure
int RdmaReduce(const void* sendbuf, void* recvbuf, int count, DataType datatype,
               ReduceOp op, int root, RdmaContext* ctx) {
  int ret, comm_rank, comm_size;
  comm_size = ctx->comm_size();
  comm_rank = ctx->comm_rank();
  char* inbuf;
  size_t dtsize;

  dtsize = DataType_Size(datatype);

  inbuf = (char*) malloc(count * dtsize);
  if (comm_rank == root) {
    if (sendbuf != COMM_IN_PLACE) {
      memcpy(recvbuf, sendbuf, count * dtsize);
    }

    for (int i = 0; i < comm_size; i++) {
      if (i == comm_rank) continue;
      ret = RdmaRecv(inbuf, count, datatype, i, 0, ctx, NULL);
      if (ret) {
        LOG(ERROR) << "RdmaRecv returned " << ret << " @ "
            << __PRETTY_FUNCTION__;
      }

      // TODO: Apply DataType other than float
      float* tmp_arr_a = (float*) recvbuf;
      float* tmp_arr_b = (float*) inbuf;
      for (int idx = 0; idx < count; idx++) {
        tmp_arr_a[idx] += tmp_arr_b[idx];
      }
    }
  } else {
    ret = RdmaSend(sendbuf, count, datatype, root, 0, ctx);
    if (ret) {
      LOG(ERROR) << "RdmaSend returned " << ret << " @ " << __PRETTY_FUNCTION__;
    }
  }

  if (inbuf != NULL) free(inbuf);

  return 0;
}

int RdmaAllreduce(const void* sendbuf, void* recvbuf, int count,
                  DataType datatype, ReduceOp op, RdmaContext* ctx) {
  return RdmaAllreduceRing(sendbuf, recvbuf, count, datatype, op, ctx);
}

int RdmaAllreduceNonOverlapping(const void* sendbuf, void* recvbuf, int count,
                                DataType datatype, ReduceOp op,
                                RdmaContext* ctx) {
  int ret, comm_rank;
  comm_rank = ctx->comm_rank();

  if (sendbuf == COMM_IN_PLACE) {
    if (comm_rank == 0) {
      ret = RdmaReduce(COMM_IN_PLACE, recvbuf, count, datatype, op, 0, ctx);
    } else {
      ret = RdmaReduce(recvbuf, NULL, count, datatype, op, 0, ctx);
    }
  } else {
    ret = RdmaReduce(sendbuf, recvbuf, count, datatype, op, 0, ctx);
  }
  assert(ret == 0);

  return RdmaBcast(recvbuf, count, datatype, 0, ctx);
}

int RdmaAllreduceRing(const void* sendbuf, void* recvbuf, int count,
                      DataType datatype, ReduceOp op, RdmaContext* ctx) {
LOG(INFO) << __FUNCTION__ << "1000";
  int ret, line, comm_rank, comm_size, k, recv_from, send_to, block_count, inbi;
  int early_segcount, late_segcount, split_rank, max_segcount;
  char *tmpsend = NULL, *tmprecv = NULL;
  char* inbuf[2] = {NULL, NULL};
  size_t true_lb, true_extnt, lb, extnt;
  size_t block_offset, max_real_segsize;
  RdmaRequest* reqs[2];
  size_t dtsize;

  comm_size = ctx->comm_size();
  comm_rank = ctx->comm_rank();
  dtsize = DataType_Size(datatype);

LOG(INFO) << __FUNCTION__ << "2000";
  // Special case for comm_size == 1
  if (comm_size == 1) {
    DVLOG(0) << "Special case for comm_size == 1";
    if (sendbuf != COMM_IN_PLACE) {
      memcpy(recvbuf, sendbuf, count * DataType_Size(datatype));
    }
    return 0;
  }

LOG(INFO) << __FUNCTION__ << "3000";
  // Special case for count less than comm_size - use simple allreduce
  if (count < comm_size) {
    return RdmaAllreduceNonOverlapping(sendbuf, recvbuf, count, datatype, op,
        ctx);
  }

LOG(INFO) << __FUNCTION__ << "4000";
  // Compute Block Count
  early_segcount = late_segcount = count / comm_size;
  split_rank = count % comm_size;
  if (split_rank != 0) {
    early_segcount += 1;
  }
  max_segcount = early_segcount;
  max_real_segsize = max_segcount * dtsize;
LOG(INFO) << __FUNCTION__ << "4100";
  inbuf[0] = (char*) malloc(max_real_segsize);
  if (inbuf[0] == NULL) return 1;
  if (comm_size > 2) {
LOG(INFO) << __FUNCTION__ << "4200";
    inbuf[1] = (char*) malloc(max_real_segsize);
    if (inbuf[1] == NULL) return 1;
  }

LOG(INFO) << std::endl << __FUNCTION__ << "\n***sendbuf=" << (uint64_t) sendbuf << ", recvbuf=" << (uint64_t) recvbuf << ", count=" << count << ", dtsize=" << DataType_Size(datatype);
  if (sendbuf != COMM_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * DataType_Size(datatype));
  }

LOG(INFO) << __FUNCTION__ << "5000";
  // Computation Loop
  send_to = (comm_rank + 1) % comm_size;
  recv_from = (comm_rank + comm_size - 1) % comm_size;

  inbi = 0;
  reqs[inbi] = new RdmaRequest();
  ret = RdmaIrecv((void*) inbuf[inbi], max_segcount, datatype, recv_from, 0,
      ctx, reqs[inbi]);
  //assert(ret == 0);
  if (comm_rank < split_rank) {
    block_offset = comm_rank * early_segcount;
    block_count = early_segcount;
  } else {
    block_offset = comm_rank * late_segcount + split_rank;
    block_count = late_segcount;
  }
  tmpsend = ((char*) recvbuf) + block_offset * dtsize;
  ret = RdmaSend((void*) tmpsend, block_count, datatype, send_to, 0, ctx);
  //assert(ret == 0);

LOG(INFO) << __FUNCTION__ << "6000";
  for (k = 2; k < comm_size; k++) {
    const int prevblock = (comm_rank + comm_size - k + 1) % comm_size;

    inbi = inbi ^ 0x1;

    reqs[inbi] = new RdmaRequest();
    ret = RdmaIrecv((void*) inbuf[inbi], max_segcount, datatype, recv_from, 0,
        ctx, reqs[inbi]);
    //assert(ret == 0);

    ret = RdmaWait(reqs[inbi ^ 0x1], NULL);
    //assert(ret == 0);
    delete reqs[inbi ^ 0x1];

    if (prevblock < split_rank) {
      block_offset = prevblock * early_segcount;
      block_count = early_segcount;
    } else {
      block_offset = prevblock * late_segcount + split_rank;
      block_count = late_segcount;
    }
    tmprecv = ((char*) recvbuf) + block_offset * dtsize;
    // TODO: Apply DataType other than float
#if 1
    float* tmp_arr_a = (float*) tmprecv;
    float* tmp_arr_b = (float*) inbuf[inbi ^ 0x1];
    for (int idx = 0; idx < block_count; idx++) {
      tmp_arr_a[idx] += tmp_arr_b[idx];
    }
#endif

    //ctx->RegisterSendBuffer((void*) tmprecv, block_count * dtsize);
    ret = RdmaSend((void*) tmprecv, block_count, datatype, send_to, 0, ctx);
    //assert(ret == 0);
  }

  ret = RdmaWait(reqs[inbi], NULL);
  //assert(ret == 0);
  delete reqs[inbi];

  recv_from = (comm_rank + 1) % comm_size;
  if (recv_from < split_rank) {
    block_offset = recv_from * early_segcount;
    block_count = early_segcount;
  } else {
    block_offset = recv_from * late_segcount + split_rank;
    block_count = late_segcount;
  }
  tmprecv = ((char*) recvbuf) + block_offset * dtsize;
  // TODO: Apply DataType other than float
#if 1
  float* tmp_arr_a = (float*) tmprecv;
  float* tmp_arr_b = (float*) inbuf[inbi];
  for (int idx = 0; idx < block_count; idx++) {
    tmp_arr_a[idx] += tmp_arr_b[idx];
  }
#endif

LOG(INFO) << __FUNCTION__ << "7000";
  // Distribution Loop
  send_to = (comm_rank + 1) % comm_size;
  recv_from = (comm_rank + comm_size - 1) % comm_size;
  for (k = 0; k < comm_size - 1; k++) {
    const int recv_data_from = (comm_rank + comm_size - k) % comm_size;
    const int send_data_from = (comm_rank + 1 + comm_size - k) % comm_size;
    const int send_block_offset = (send_data_from < split_rank)
        ? (send_data_from * early_segcount)
        : (send_data_from * late_segcount + split_rank);
    const int recv_block_offset = (recv_data_from < split_rank)
        ? (recv_data_from * early_segcount)
        : (recv_data_from * late_segcount + split_rank);
    block_count = (send_data_from < split_rank)
        ? early_segcount : late_segcount;

    tmprecv = (char*) recvbuf + recv_block_offset * dtsize;
    tmpsend = (char*) recvbuf + send_block_offset * dtsize;

    ret = RdmaSendrecv((void*) tmpsend, block_count, datatype, send_to, 0,
        (void*) tmprecv, max_segcount, datatype, recv_from, 0, ctx, NULL);
    //assert(ret == 0);
  }

  if (inbuf[0] != NULL) free(inbuf[0]);
  if (inbuf[1] != NULL) free(inbuf[1]);

  return 0;
}

}  // namespace common
}  // namespace ptre
