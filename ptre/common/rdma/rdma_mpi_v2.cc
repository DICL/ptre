#include "ptre/common/rdma/rdma_mpi_v2.h"

#include "ptre/common/rdma/rdma_mpi.h"

#include <infiniband/verbs.h>

namespace ptre {
namespace common {

int RdmaAllreduceV2(const void* sendbuf, void* recvbuf, int count,
                  DataType datatype, ReduceOp op, RdmaContext* ctx) {
  return RdmaAllreduceRingV2(sendbuf, recvbuf, count, datatype, op, ctx);
}

int RdmaAllreduceRingV2(const void* sendbuf, void* recvbuf, int count,
                        DataType datatype, ReduceOp op,
                        RdmaContext* ctx) {
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
  size_t block_counts[comm_size];
  RemoteAddr my_recvs[comm_size];
  RemoteAddr my_inbufs[comm_size];
  RemoteAddr remote_recvs[comm_size];
  RemoteAddr remote_inbufs[comm_size];
  RdmaRequest reqs[2][comm_size];
  int send_to = (comm_rank + 1) % comm_size;
  int recv_from = (comm_rank + comm_size - 1) % comm_size;
  uint32_t curr;

  // Init Recv buffer and Intermediate buffer
  dtsize = DataType_Size(datatype);
  if (sendbuf != COMM_IN_PLACE) {
    memcpy(recvbuf, sendbuf, count * dtsize);
  }
  inbuf = (char*) malloc(count * dtsize);

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
    tmprecvs[i] = ((char*) recvbuf) + block_offset * dtsize;
    tmpinbufs[i] = inbuf + block_offset * dtsize;
  }

  // Register MR
  for (int i = 0; i < comm_size; i++) {
    struct ibv_mr* m;
    m = ctx->RegisterRecvBuffer((void*) tmprecvs[i], block_counts[i] * dtsize);
    if (m == NULL) {
      LOG(ERROR) << "recv_mr is NULL";
      return 1;
    }
    my_recvs[i].remote_addr = (uint64_t) tmprecvs[i];
    my_recvs[i].rkey = m->rkey;

    m = ctx->RegisterRecvBuffer((void*) tmpinbufs[i], block_counts[i] * dtsize);
    if (m == NULL) {
      LOG(ERROR) << "recv_mr is NULL";
      return 1;
    }
    my_inbufs[i].remote_addr = (uint64_t) tmpinbufs[i];
    my_inbufs[i].rkey = m->rkey;
  }
  // Exchange address and rkey
  RET_OK(RdmaSendrecv((void*) my_recvs, sizeof(RemoteAddr) * comm_size,
        DataType::DT_STRING, recv_from, 0,
        (void*) remote_recvs, sizeof(RemoteAddr) * comm_size,
        DataType::DT_STRING, send_to, 0, ctx, NULL));
  RET_OK(RdmaSendrecv((void*) my_inbufs, sizeof(RemoteAddr) * comm_size,
        DataType::DT_STRING, recv_from, 0,
        (void*) remote_inbufs, sizeof(RemoteAddr) * comm_size,
        DataType::DT_STRING, send_to, 0, ctx, NULL));

  // Computation Loop
  curr = comm_rank;
  RET_OK(RdmaIwriteWithImm((void*) tmprecvs[curr], curr, remote_inbufs[curr],
        block_counts[curr], datatype, send_to, 0, ctx, &reqs[0][curr]));
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
          block_counts[curr], datatype, send_to, 0, ctx, &reqs[0][curr]));
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
          block_counts[curr], datatype, send_to, 0, ctx, &reqs[1][curr]));
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
  for (int i = 0; i < comm_size; i++) {
    ctx->DeregisterRecvBuffer((void*) tmprecvs[i]);
    ctx->DeregisterRecvBuffer((void*) tmpinbufs[i]);
  }
  if (inbuf != NULL) free(inbuf);
  return 0;
}

}  // namespace common
}  // namespace ptre
