#include "ptre/common/communication/tcp/tcp_grpc_client.h"

namespace ptre {
namespace common {

namespace {
using grpc::ClientContext;
}  // namespace

TcpGrpcClient::TcpGrpcClient(int src_rank, int dst_rank, const string& hostname)
    : src_rank_(src_rank), dst_rank_(dst_rank), hostname_(hostname) {
  grpc::ChannelArguments ch_args;
  ch_args.SetMaxReceiveMessageSize(-1);//max msg size
  std::shared_ptr<::grpc::Channel> channel = grpc::CreateCustomChannel(hostname,
      grpc::InsecureChannelCredentials(), ch_args);
  stub_ = Tcp::NewStub(channel);
}

int TcpGrpcClient::PullTensor(const string& tensor_name, uint64_t step,
                              Tensor& out) {
  PullTensorRequest request;
  request.set_src_rank(src_rank_);
  request.set_tensor_name(tensor_name);
  request.set_src_step(step);
  request.set_sync_mode(P2P_SYNC_MODE_STEP);

  PullTensorResponse response;
  ClientContext context;
  grpc::Status status = stub_->PullTensor(&context, request, &response);
  if (status.ok()) {
    if (response.status() == 0) {
      std::copy(response.buf().begin(), response.buf().end(),
          const_cast<char*>(out.tensor_data().data()));
    }
    return response.status();
  } else {
    std::cout << status.error_code() << ": " << status.error_message() << std::endl;
    return -1;
  }
}

int TcpGrpcClient::PushTensor(const string& tensor_name, const Tensor& tensor) {
  PushTensorRequest request;
  request.set_src_rank(src_rank_);
  request.set_tensor_name(tensor_name);
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

}  // namespace common
}  // namespace ptre
