#ifndef PTRE_COMMUNICATION_RDMA_GRPC_SERVER_H_
#define PTRE_COMMUNICATION_RDMA_GRPC_SERVER_H_

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "ptre/communication/rdma/rdma.h"
#include "ptre/protobuf/rdma_service.grpc.pb.h"
#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/rdma_manager.h"

namespace ptre {

using std::string;

class RdmaServiceImpl final : public Rdma::Service {
 public:
  grpc::Status GetRemoteAddress(grpc::ServerContext* context,
                                const GetRemoteAddressRequest* request,
                                GetRemoteAddressResponse* response) override;
  grpc::Status GetRemoteParamAddress(grpc::ServerContext* context,
                            const GetRemoteParamAddressRequest* request,
                            GetRemoteParamAddressResponse* response) override;
  grpc::Status GetRemoteEnv(grpc::ServerContext* context,
                                const GetRemoteEnvRequest* request,
                                GetRemoteEnvResponse* response) override;
  grpc::Status AttemptPush(grpc::ServerContext* context,
      const AttemptPushRequest* request,
      AttemptPushResponse* response) override;
  grpc::Status NotifyPushDone(grpc::ServerContext* context,
      const NotifyPushDoneRequest* request,
      NotifyPushDoneResponse* response) override;
  grpc::Status Barrier(grpc::ServerContext* context,
      const BarrierRequest* request,
      BarrierResponse* response) override;
  grpc::Status GetRemoteAddressV2(grpc::ServerContext* context,
      const GetRemoteAddressV2Request* request,
      GetRemoteAddressV2Response* response) override;

  void SetRdmaManager(RdmaManager* rdma_manager);
  void SetConsensusManager(ConsensusManager* cm);
  void SetBarrierVariable(bool* barrier_variable);

 private:
  bool* barrier_variable_ = nullptr;
  RdmaManager* rdma_manager_ = nullptr;  // not owned.
  ConsensusManager* cm_ = nullptr;  // not owned.

  // Rdma Attributes
  //uint32_t lid_;
  //std::map<string, uint32_t> qpns_;
  //std::map<string, uint64_t> remote_addrs_;
  //std::map<string, uint32_t> rkeys_;
};

class GrpcServer {
 public:
  //~GrpcServer();
  //void SetRdmaManager(RdmaManager* rdma_manager);
  static void RunServer(RdmaManager* rdma_manager);

 private:
  //std::unique_ptr<grpc::Server> server_;
  //std::unique_ptr<std::thread> t_ = nullptr;
  //RdmaManager* rdma_manager_;
};

}  // namespace ptre


#endif  // PTRE_COMMUNICATION_RDMA_GRPC_SERVER_H_
