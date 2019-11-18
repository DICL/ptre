#include "tensorflow/ptre/rpc/ptre_service.h"

#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/channel_interface.h"
#include "grpcpp/impl/codegen/client_unary_call.h"
#include "grpcpp/impl/codegen/method_handler_impl.h"
#include "grpcpp/impl/codegen/rpc_service_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/sync_stream.h"

namespace tensorflow {

namespace grpc {

static const char* grpcPtreService_method_names[] = {
    "/tensorflow.PtreService/GetRemoteAddress",
};

std::unique_ptr<PtreService::Stub> PtreService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<PtreService::Stub> stub(new PtreService::Stub(channel));
  return stub;
}

PtreService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_GetRemoteAddress_(grpcPtreService_method_names[0],
                                  ::grpc::internal::RpcMethod::NORMAL_RPC,
                                  channel) {}

::grpc::Status PtreService::Stub::GetRemoteAddress(
    ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
    GetRemoteAddressResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(
      channel_.get(), rpcmethod_GetRemoteAddress_, context, request, response);
}

PtreService::AsyncService::AsyncService() {
  for (int i = 0; i < 1; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcPtreService_method_names[i],
        ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

PtreService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
