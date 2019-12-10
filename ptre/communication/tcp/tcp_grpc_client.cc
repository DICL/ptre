#include "ptre/communication/tcp/tcp_grpc_client.h"

namespace ptre {
namespace {
using grpc::ClientContext;
}  // namespace

TcpGrpcClient::TcpGrpcClient(int src_rank, int dst_rank, const string& hostname)
    : src_rank_(src_rank), dst_rank_(dst_rank), hostname_(hostname) {
  std::shared_ptr<::grpc::Channel> channel = grpc::CreateChannel(hostname,
      grpc::InsecureChannelCredentials());
  stub_ = Tcp::NewStub(channel);
}

int TcpGrpcClient::PushTensor(const string& name, const Tensor& tensor) {
  PushTensorRequest request;
  request.set_src_rank(src_rank_);
  request.set_name(name);
  size_t buffer_size = (size_t) tensor.TotalBytes();
  request.set_buf(tensor.tensor_data().data(), buffer_size);  // COPY

  PushTensorResponse response;
  ClientContext context;
  grpc::Status status = stub_->PushTensor(&context, request, &response);
  if (status.ok()) {
    return 0;
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return -1;
  }
}

}  // namespace ptre
