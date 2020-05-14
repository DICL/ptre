#ifndef PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_
#define PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_

#include <memory>
#include <string>
#include <map>

#include <grpcpp/grpcpp.h>

#include "ptre/protobuf/rdma_service.grpc.pb.h"
#include "ptre/communication/rdma/rdma_manager.h"

namespace ptre {

using std::string;

class GrpcClient {
 public:
  //GrpcClient(std::shared_ptr<::grpc::Channel> channel);
  GrpcClient(int src_rank, int dst_rank, const std::string& hostname);
  ~GrpcClient();
  int GetRemoteAddress(const std::string& name);
  int GetRemoteParamAddress();
  int GetRemoteEnv();
  bool AttemptPush(int vstep);
  int NotifyPushDone();
  bool Barrier();
  void SetRdmaManager(RdmaManager* rdma_manager);

  int GetRemoteAddressV2(const BufType type, const string& name,
                         RemoteMR* const rmr);

  int Recv(char* buf, size_t len, const string& name);

 private:
  int client_rank_;
  int dst_rank_;
  std::string hostname_;
  std::unique_ptr<Rdma::Stub> stub_;
  RdmaManager* rdma_manager_ = nullptr;
};

//class GrpcClientCache {
// public:
//  GrpcClientCache(int rank) : rank_(rank) {}
//  GrpcClient* GetClient(int dst_rank);
//
// private:
//  int rank_;
//  std::map<int, GrpcClient*> cache_;
//};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_GRPC_CLIENT_H_
