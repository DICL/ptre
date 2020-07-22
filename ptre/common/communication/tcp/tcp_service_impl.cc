#include "ptre/common/communication/tcp/tcp_service_impl.h"

namespace ptre {
namespace common {

void TcpServiceImpl::SetConsensusManager(ConsensusManager* cm) {
  cm_ = cm;
}

grpc::Status TcpServiceImpl::PullTensor(grpc::ServerContext* context,
                                        const PullTensorRequest* request,
					PullTensorResponse* response) {
  if (cm_ == nullptr) return grpc::Status::CANCELLED;
  int rank = request->src_rank();
  response->set_tensor_name(request->tensor_name());
  auto t = cm_->ready_tensor(request->tensor_name());
  response->set_buf(
      static_cast<const void*>(t->tensor_data().data()),
      t->AllocatedBytes());
  response->set_status(0);
  return grpc::Status::OK;
}

grpc::Status TcpServiceImpl::PushTensor(grpc::ServerContext* context,
                                        const PushTensorRequest* request,
                                        PushTensorResponse* response) {
  int rank = request->src_rank();
  string tensor_name = request->tensor_name();
  string buf = request->buf();
  Tensor recv_tensor = cm_->global_consensus(tensor_name);
  std::copy(buf.begin(), buf.end(),
            const_cast<char*>(recv_tensor.tensor_data().data()));

/// message PushTensorResponse {
///   int32 dst_rank = 1;
///   string tensor_name = 2;
///   int32 status = 3;
/// }
  response->set_status(0);
  return grpc::Status::OK;
}
}
}
