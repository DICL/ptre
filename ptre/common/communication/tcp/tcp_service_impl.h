#ifndef PTRE_COMMON_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_
#define PTRE_COMMON_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_

#include <string>

#include "ptre/protobuf/tcp_service.grpc.pb.h"
#include "ptre/common/cm/consensus_manager.h"

namespace ptre {
namespace common {

using std::string;

class TcpServiceImpl final : public Tcp::Service {
 public:
  TcpServiceImpl(ConsensusManager* cm);
  grpc::Status PullTensor(grpc::ServerContext* context,
                          const PullTensorRequest* request,
			  PullTensorResponse* response) override;
  grpc::Status PushTensor(grpc::ServerContext* context,
                          const PushTensorRequest* request,
                          PushTensorResponse* response) override;

 private:
  ConsensusManager* cm_;
};

}  // namespace common
}  // namespace ptre
#endif  // PTRE_COMMON_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_
