#ifndef PTRE_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_
#define PTRE_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_

#include <string>

#include "ptre/protobuf/tcp_service.grpc.pb.h"
#include "ptre/cm/consensus_manager.h"

namespace ptre {

using std::string;

class TcpServiceImpl final : public Tcp::Service {
 public:
  TcpServiceImpl(ConsensusManager* cm);
  grpc::Status PushTensor(grpc::ServerContext* context,
                          const PushTensorRequest* request,
                          PushTensorResponse* response) override;

 private:
  ConsensusManager* cm_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_
