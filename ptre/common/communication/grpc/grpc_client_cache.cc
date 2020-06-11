#include "ptre/common/communication/grpc/grpc_client_cache.h"

namespace ptre {
namespace common {

//GrpcClientCache::~GrpcClientCache() {
//}
GrpcClientCache::GrpcClientCache(int rank,
                                 const std::vector<string>& hostnames) {
  rank_ = rank;
  for (auto it : hostnames) {
    hostnames_.push_back(it);
  }
}

int GrpcClientCache::GetClient(int target, GrpcClient** client) {
  auto it = clients_.find(target);
  if (it == clients_.end()) {
    auto worker = std::unique_ptr<GrpcClient>(new GrpcClient(rank_, target,
          hostnames_[target]));
    it = clients_.emplace(target, std::move(worker)).first;
  }
  *client = it->second.get();
  return 0;
}

}  // namespace common
}  // namespace ptre
