#include "ptre/common/communication/grpc/grpc_client_cache.h"

#include "ptre/common/communication/rdma/grpc_client.h"
#include "ptre/common/communication/tcp/tcp_grpc_client.h"

namespace ptre {
namespace common {

//GrpcClientCache::~GrpcClientCache() {
//}
template <typename T>
GrpcClientCache<T>::GrpcClientCache(int rank,
                                    const std::vector<string>& hostnames) {
  rank_ = rank;
  for (auto it : hostnames) {
    hostnames_.push_back(it);
  }
}

template <typename T>
int GrpcClientCache<T>::GetClient(int target, T** client) {
  auto it = clients_.find(target);
  if (it == clients_.end()) {
    auto worker = std::unique_ptr<T>(new T(rank_, target,
          hostnames_[target]));
    it = clients_.emplace(target, std::move(worker)).first;
  }
  *client = it->second.get();
  return 0;
}

// Forward declarations.
template class GrpcClientCache<GrpcClient>;
template class GrpcClientCache<TcpGrpcClient>;

}  // namespace common
}  // namespace ptre
