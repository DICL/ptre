#ifndef PTRE_COMMON_COMMUNICATION_RDMA_GRPC_CLIENT_H_
#define PTRE_COMMON_COMMUNICATION_RDMA_GRPC_CLIENT_H_

#include <memory>
#include <string>
#include <map>

#include <grpcpp/grpcpp.h>

#include "ptre/protobuf/rdma_service.grpc.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace ptre {
namespace common {

using std::string;

class GrpcClient {
 public:
  //GrpcClient(std::shared_ptr<::grpc::Channel> channel);
  GrpcClient(int src_rank, int dst_rank, const std::string& hostname);
  ~GrpcClient();
  int GetLID(uint16_t* remote_lid);
  int GetQPAttr(uint32_t* remote_qpn, uint32_t* remote_psn);
  int GetRemoteAddress(const BufType type, const std::string& name,
      uint64_t* out_remote_addr, uint32_t* out_rkey);
  int GetRemoteParamAddress();
  int GetRemoteEnv();
  bool AttemptPush(int vstep = -1);
  int NotifyPushDone(const string& var_name);
  bool Barrier();

  int Recv(char* buf, size_t len, const string& name);
  int GetPermit(const string& name);
  int AttemptPushVar(const string& var_name);
  int CancelPushVar(const string& var_name);

 private:
  int comm_rank_;
  int dst_rank_;
  std::string hostname_;
  std::unique_ptr<Rdma::Stub> stub_;
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

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_COMMUNICATION_RDMA_GRPC_CLIENT_H_
