#ifndef PTRE_COMMON_RDMA_RDMA_MPI_V2_H_
#define PTRE_COMMON_RDMA_RDMA_MPI_V2_H_

#include "ptre/common/common.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/rdma/rdma_request.h"

namespace ptre {
namespace common {

int RdmaInitAllreduceV2(const void* sendbuf, const void* recvbuf,
                        const void* inbuf, int count, DataType datatype,
                        RdmaContext* ctx);

int RdmaAllreduceV2(const void* sendbuf, void* recvbuf, int count,
                  DataType datatype, ReduceOp op, RdmaContext* ctx);

int RdmaAllreduceRingV2(const void* sendbuf, void* recvbuf, int count,
                        DataType datatype, ReduceOp op,
                        RdmaContext* ctx);

}  // namespace common
}  // namespace ptre


#endif  // PTRE_COMMON_RDMA_RDMA_MPI_V2_H_
