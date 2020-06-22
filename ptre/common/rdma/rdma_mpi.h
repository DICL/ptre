#ifndef PTRE_COMMON_RDMA_RDMA_MPI_H_
#define PTRE_COMMON_RDMA_RDMA_MPI_H_

#include <infiniband/verbs.h>

#include "ptre/common/common.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/rdma/rdma_request.h"

namespace ptre {
namespace common {

int RdmaWait(RdmaRequest* request, Status* status);

int RdmaIsend(const void* buf, int count, DataType datatype, int dest, int tag,
              RdmaContext* ctx, RdmaRequest* request);

int RdmaSend(const void* buf, int count, DataType datatype, int dest, int tag,
             RdmaContext* ctx);

int RdmaIrecv(void* buf, int count, DataType datatype, int source, int tag,
              RdmaContext* ctx, RdmaRequest* request);

int RdmaRecv(void* buf, int count, DataType datatype, int source, int tag,
             RdmaContext* ctx, Status* status);

int RdmaSendrecv(const void* sendbuf, int sendcount, DataType sendtype,
                 int dest, int sendtag, void* recvbuf, int recvcount,
                 DataType recvtype, int source, int recvtag, RdmaContext* ctx,
                 Status* status);

int RdmaBcast(void* buffer, int count, DataType datatype, int root,
              RdmaContext* ctx);

int RdmaReduce(const void* sendbuf, void* recvbuf, int count, DataType datatype,
               ReduceOp op, int root, RdmaContext* ctx);

int RdmaAllreduce(const void* sendbuf, void* recvbuf, int count,
                  DataType datatype, ReduceOp op, RdmaContext* ctx);

int RdmaAllreduceNonOverlapping(const void* sendbuf, void* recvbuf, int count,
                                DataType datatype, ReduceOp op,
                                RdmaContext* ctx);

int RdmaAllreduceRing(const void* sendbuf, void* recvbuf, int count,
                      DataType datatype, ReduceOp op, RdmaContext* ctx);

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_RDMA_RDMA_MPI_H_
