#ifndef PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "ptre/protobuf/rdma_service.pb.h"

namespace ptre {

class RdmaManager {
 private:
  std::unordered_map<int, RdmaChannel> remotes_;
  //std::vector<Channel> channels_;
  //std::vector<MemoryRegion> mrs_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_MANAGER_H_
