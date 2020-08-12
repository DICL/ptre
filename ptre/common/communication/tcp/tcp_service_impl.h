#ifndef PTRE_COMMON_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_
#define PTRE_COMMON_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_

#include <mutex>
#include <string>

#include "ptre/protobuf/tcp_service.grpc.pb.h"
#include "ptre/common/common.h"
#include "ptre/common/cm/consensus_manager.h"

namespace ptre {
namespace common {

using std::string;
using TensorState =
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<StateMutex>>;
using CommBufTable = std::unordered_map<string, TensorState>;

class TcpServiceImpl final : public Tcp::Service {
 public:
  void SetConsensusManager(ConsensusManager* cm);
  void SetCommBufTables(CommBufTable* sbt, CommBufTable* rbt,
                        std::mutex* mu);
  grpc::Status PullTensor(grpc::ServerContext* context,
                          const PullTensorRequest* request,
			  PullTensorResponse* response) override;
  grpc::Status PushTensor(grpc::ServerContext* context,
                          const PushTensorRequest* request,
                          PushTensorResponse* response) override;

 private:
  ConsensusManager* cm_ = nullptr;
  CommBufTable* sendbuf_table_;
  CommBufTable* recvbuf_table_;
  std::mutex* commbuf_table_mu_;
};

}  // namespace common
}  // namespace ptre
#endif  // PTRE_COMMON_COMMUNICATION_TCP_TCP_SERVICE_IMPL_H_
