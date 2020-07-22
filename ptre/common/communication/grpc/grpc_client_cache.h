#ifndef PTRE_COMMON_COMMUNICATION_GRPC_CLIENT_CACHE_H_
#define PTRE_COMMON_COMMUNICATION_GRPC_CLIENT_CACHE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

//#include <grpcpp/grpcpp.h>


namespace ptre {
namespace common {

//typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;
using std::string;

template <typename T>
class GrpcClientCache {
 public:
  GrpcClientCache(int rank, const std::vector<string>& hostnames);
  //~GrpcClientCache();
  int GetClient(int target, T** client);

 private:
  int rank_;
  std::map<int, std::unique_ptr<T>> clients_;
  std::vector<std::string> hostnames_;
};

}  // namespace common
}  // namespace ptre


#endif  // PTRE_COMMON_COMMUNICATION_GRPC_CLIENT_CACHE_H_
