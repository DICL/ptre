#include "ptre/common/rdma/rdma_mpi_v2.h"

#include "ptre/common/rdma/rdma_mpi.h"

#include <infiniband/verbs.h>

namespace ptre {
namespace common {

int RdmaInitAllreduceV2(const void* sendbuf, const void* recvbuf,
                        const void* inbuf, int count, DataType datatype,
                        RdmaContext* ctx) {
  int comm_size = ctx->comm_size();
  int comm_rank = ctx->comm_rank();
  size_t dtsize = DataType_Size(datatype);
  int send_to = (comm_rank + 1) % comm_size;
  int recv_from = (comm_rank + comm_size - 1) % comm_size;

  ctx->allreduce_set_intermediate_buf((void*) sendbuf, (char*) inbuf);
  struct ibv_mr* mr = ctx->RegisterRecvBuffer((void*) inbuf, count * dtsize);
  RemoteAddr my = { (uint64_t) mr->addr, mr->rkey };
  RemoteAddr ra;
  RET_OK(RdmaSendrecv((void*) &my, sizeof(RemoteAddr), DataType::DT_STRING,
        recv_from, 0, (void*) &ra, sizeof(RemoteAddr), DataType::DT_STRING,
        send_to, 0, ctx, NULL));
  ctx->set_remote_addr(RdmaContext::REMOTE_ADDR_ALLREDUCE_INTERMEDIATE_BUF,
      (void*) recvbuf, ra);

  mr = ctx->RegisterRecvBuffer((void*) recvbuf, count * dtsize);
  my = { (uint64_t) mr->addr, mr->rkey };
  RET_OK(RdmaSendrecv((void*) &my, sizeof(RemoteAddr), DataType::DT_STRING,
        recv_from, 0, (void*) &ra, sizeof(RemoteAddr), DataType::DT_STRING,
        send_to, 0, ctx, NULL));
  ctx->set_remote_addr(RdmaContext::REMOTE_ADDR_ALLREDUCE_RECV_BUF,
      (void*) recvbuf, ra);
}

/*
int RdmaFinalizeAllreduceV2(const void* sendbuf, const void* recvbuf,
                        const void* inbuf, int count, DataType datatype,
                        RdmaContext* ctx) {
                        */

int RdmaAllreduceV2(const void* sendbuf, void* recvbuf, int count,
                  DataType datatype, ReduceOp op, RdmaContext* ctx) {
  return RdmaAllreduceRingV2(sendbuf, recvbuf, count, datatype, op, ctx);
}

int RdmaAllreduceRingV2(const void* sendbuf, void* recvbuf, int count,
                        DataType datatype, ReduceOp op,
                        RdmaContext* ctx) {
  int ret;
  int comm_rank, comm_size;
  comm_size = ctx->comm_size();
  comm_rank = ctx->comm_rank();
  // Special case for comm_size == 1
  if (comm_size == 1) {
    if (sendbuf != COMM_IN_PLACE) {
      memcpy(recvbuf, sendbuf, count * DataType_Size(datatype));
    }
    return 0;
  }
  // Special case for count less than comm_size - use simple allreduce
  if (count < comm_size) {
    return RdmaAllreduceNonOverlapping(sendbuf, recvbuf, count, datatype, op,
        ctx);
  }

  size_t dtsize;
  char* inbuf;
  int early_segcount;
  int late_segcount;
  int split_rank;
  int max_segcount;
  char* tmprecvs[comm_size];
  char* tmpinbufs[comm_size];
  size_t byte_offsets[comm_size];
  size_t block_counts[comm_size];
  RemoteAddr remote_recvs[comm_size];
  RemoteAddr remote_inbufs[comm_size];
  RdmaRequest reqs[2][comm_size];
  int send_to = (comm_rank + 1) % comm_size;
  int recv_from = (comm_rank + comm_size - 1) % comm_size;
  uint32_t curr;
  struct ibv_mr* recv_mr;
  struct ibv_mr* inbuf_mr;

  // Init Recv buffer and Intermediate buffer
  dtsize = DataType_Size(datatype);
  if (sendbuf != COMM_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * dtsize);
  }
  inbuf = ctx->allreduce_intermediate_buf((void*) sendbuf);
  if (inbuf == NULL) {
    inbuf = (char*) malloc(count * dtsize);
  }

  // Compute Block Count and Offsets
  early_segcount = late_segcount = count / comm_size;
  split_rank = count % comm_size;
  if (split_rank != 0) {
    early_segcount += 1;
  }
  max_segcount = early_segcount;
  //max_real_segsize = max_segcount * dtsize;
  for (int i = 0; i < comm_size; i++) {
    size_t block_offset;
    if (i < split_rank) {
      block_offset = i * early_segcount;
      block_counts[i] = early_segcount;
    } else {
      block_offset = i * late_segcount + split_rank;
      block_counts[i] = late_segcount;
    }
    byte_offsets[i] = block_offset * dtsize;
    tmprecvs[i] = ((char*) recvbuf) + block_offset * dtsize;
    tmpinbufs[i] = inbuf + block_offset * dtsize;
  }

  // Remote Address
  RemoteAddr recv_ra;
  ret = ctx->get_remote_addr(
      RdmaContext::REMOTE_ADDR_ALLREDUCE_RECV_BUF, (void*) recvbuf, &recv_ra);
  if (ret) {
    recv_mr = ctx->recv_mr(recvbuf);
    if (recv_mr == NULL) {
      // Register MR
      recv_mr = ibv_reg_mr(ctx->pd(), recvbuf, count * dtsize,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
    RemoteAddr my = { (uint64_t) recv_mr->addr, recv_mr->rkey };
    // Exchange address and rkey
    RET_OK(RdmaSendrecv((void*) &my, sizeof(RemoteAddr), DataType::DT_STRING,
          recv_from, 0, (void*) &recv_ra, sizeof(RemoteAddr),
          DataType::DT_STRING, send_to, 0, ctx, NULL));
  }
  for (int i = 0; i < comm_size; i++) {
    remote_recvs[i].remote_addr = recv_ra.remote_addr + byte_offsets[i];
    remote_recvs[i].rkey = recv_ra.rkey;
  }
  RemoteAddr inbuf_ra;
  ret = ctx->get_remote_addr(
      RdmaContext::REMOTE_ADDR_ALLREDUCE_INTERMEDIATE_BUF, (void*) recvbuf,
      &inbuf_ra);
  if (ret) {
    inbuf_mr = ctx->recv_mr((void*) inbuf);
    if (inbuf_mr == NULL) {
      // Register MR
      inbuf_mr = ibv_reg_mr(ctx->pd(), (void*) inbuf, count * dtsize,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
    RemoteAddr my = { (uint64_t) inbuf_mr->addr, inbuf_mr->rkey };
    // Exchange address and rkey
    RET_OK(RdmaSendrecv((void*) &my, sizeof(RemoteAddr), DataType::DT_STRING,
          recv_from, 0, (void*) &inbuf_ra, sizeof(RemoteAddr),
          DataType::DT_STRING, send_to, 0, ctx, NULL));
  }
  for (int i = 0; i < comm_size; i++) {
    remote_inbufs[i].remote_addr = inbuf_ra.remote_addr + byte_offsets[i];
    remote_inbufs[i].rkey = inbuf_ra.rkey;
  }

  // Computation Loop
  curr = comm_rank;
  RET_OK(RdmaIwriteWithImm((void*) tmprecvs[curr], curr, remote_inbufs[curr],
        block_counts[curr], datatype, send_to, 0, ctx, &reqs[0][curr],
        recv_mr));
  for (int k = 0; k < comm_size - 2; k++) {
    RET_OK(RdmaRecvWithImm(NULL, &curr, 0, DataType::DT_STRING, recv_from, 0,
          ctx, NULL));
    // TODO: Apply DataType other than float
    float* seg_recv = (float*) tmprecvs[curr];
    float* seg_inbuf = (float*) tmpinbufs[curr];
    for (int idx = 0; idx < block_counts[curr]; idx++) {
      seg_recv[idx] += seg_inbuf[idx];
    }
    RET_OK(RdmaIwriteWithImm((void*) tmprecvs[curr], curr, remote_inbufs[curr],
          block_counts[curr], datatype, send_to, 0, ctx, &reqs[0][curr],
          recv_mr));
  }
  RET_OK(RdmaRecvWithImm(NULL, &curr, 0, DataType::DT_STRING, recv_from, 0,
        ctx, NULL));
  // TODO: Apply DataType other than float
  float* seg_recv = (float*) tmprecvs[curr];
  float* seg_inbuf = (float*) tmpinbufs[curr];
  for (int idx = 0; idx < block_counts[curr]; idx++) {
    seg_recv[idx] += seg_inbuf[idx];
  }

  // Distribution Loop
  curr = (comm_rank + 1) % comm_size;
  for (int k = 0; k < comm_size - 1; k++) {
    RET_OK(RdmaIwriteWithImm((void*) tmprecvs[curr], curr, remote_recvs[curr],
          block_counts[curr], datatype, send_to, 0, ctx, &reqs[1][curr],
          recv_mr));
    RET_OK(RdmaRecvWithImm(NULL, &curr, 0, DataType::DT_STRING, recv_from, 0,
          ctx, NULL));
  }

  // Finalize Async Rdma Requests
  // reqs[0][(comm_rank + 1) % comm_size].Done();
  // reqs[1][(comm_rank + 2) % comm_size].Done();
  for (int i = 0; i < 2; i++) {
    reqs[i][(comm_rank + i + 1) % comm_size].Done();
    for (int j = 0; j < comm_size; j++) {
      RET_OK(reqs[i][j].Join());
    }
  }

  // Deregister MR
  //for (int i = 0; i < comm_size; i++) {
  //  ctx->DeregisterRecvBuffer((void*) tmprecvs[i]);
  //}
  if (ctx->recv_mr(recvbuf) == NULL && recv_mr != NULL) {
    ibv_dereg_mr(recv_mr);
  }
  if (ctx->recv_mr((void*) inbuf) == NULL && inbuf_mr != NULL) {
    ibv_dereg_mr(inbuf_mr);
  }
  if (ctx->allreduce_intermediate_buf((void*) sendbuf) == NULL
      && inbuf != NULL) {
    //ctx->DeregisterRecvBuffer((void*) inbuf);
    free(inbuf);
  }
  return 0;
}

}  // namespace common
}  // namespace ptre
