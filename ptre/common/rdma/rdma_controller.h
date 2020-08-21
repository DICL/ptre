#ifndef PTRE_COMMON_RDMA_RDMA_CONTROLLER_H_
#define PTRE_COMMON_RDMA_RDMA_CONTROLLER_H_

#include "ptre/common/common.h"

namespace ptre {
namespace common {

Status PostRecvWithImm(RdmaRecvEntry* entry);

Status RdmaRead(RdmaEntry* entry);

Status RdmaWrite(RdmaEntry* entry);

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_RDMA_RDMA_CONTROLLER_H_
