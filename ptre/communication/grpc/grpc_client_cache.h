#ifndef PTRE_COMMUNICATION_GRPC_CLIENT_CACHE_H_
#define PTRE_COMMUNICATION_GRPC_CLIENT_CACHE_H_

#include <map>
#include <memory>
#include <string>

//#include <grpcpp/grpcpp.h>

#include "ptre/communication/rdma/grpc_client.h"

namespace ptre {

//typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;
using std::string;

class GrpcClientCache {
 public:
  GrpcClientCache(int rank, const std::vector<string>& hostnames);
  //~GrpcClientCache();
  int GetClient(int target, GrpcClient** client);

 private:
  int rank_;
  std::map<int, std::unique_ptr<GrpcClient>> clients_;
  std::vector<std::string> hostnames_;
};

}  // namespace ptre


#endif  // PTRE_COMMUNICATION_GRPC_CLIENT_CACHE_H_
