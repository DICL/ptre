#include "ptre/communication/tcp/tcp_service_impl.h"

namespace ptre {

TcpServiceImpl::TcpServiceImpl(ConsensusManager* cm)
    : Tcp::Service(), cm_(cm) {}

grpc::Status TcpServiceImpl::PushTensor(grpc::ServerContext* context,
                                        const PushTensorRequest* request,
                                        PushTensorResponse* response) {
  int rank = request->src_rank();
  string tensor_name = request->name();
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
