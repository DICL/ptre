// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: tcp_service.proto

#include "tcp_service.pb.h"
#include "tcp_service.grpc.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/channel_interface.h>
#include <grpcpp/impl/codegen/client_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/rpc_service_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/sync_stream.h>
namespace ptre {

static const char* Tcp_method_names[] = {
  "/ptre.Tcp/PushTensor",
};

std::unique_ptr< Tcp::Stub> Tcp::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< Tcp::Stub> stub(new Tcp::Stub(channel));
  return stub;
}

Tcp::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_PushTensor_(Tcp_method_names[0], ::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status Tcp::Stub::PushTensor(::grpc::ClientContext* context, const ::ptre::PushTensorRequest& request, ::ptre::PushTensorResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_PushTensor_, context, request, response);
}

void Tcp::Stub::experimental_async::PushTensor(::grpc::ClientContext* context, const ::ptre::PushTensorRequest* request, ::ptre::PushTensorResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_PushTensor_, context, request, response, std::move(f));
}

void Tcp::Stub::experimental_async::PushTensor(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::ptre::PushTensorResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_PushTensor_, context, request, response, std::move(f));
}

void Tcp::Stub::experimental_async::PushTensor(::grpc::ClientContext* context, const ::ptre::PushTensorRequest* request, ::ptre::PushTensorResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_PushTensor_, context, request, response, reactor);
}

void Tcp::Stub::experimental_async::PushTensor(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::ptre::PushTensorResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_PushTensor_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::ptre::PushTensorResponse>* Tcp::Stub::AsyncPushTensorRaw(::grpc::ClientContext* context, const ::ptre::PushTensorRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::ptre::PushTensorResponse>::Create(channel_.get(), cq, rpcmethod_PushTensor_, context, request, true);
}

::grpc::ClientAsyncResponseReader< ::ptre::PushTensorResponse>* Tcp::Stub::PrepareAsyncPushTensorRaw(::grpc::ClientContext* context, const ::ptre::PushTensorRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::ptre::PushTensorResponse>::Create(channel_.get(), cq, rpcmethod_PushTensor_, context, request, false);
}

Tcp::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      Tcp_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< Tcp::Service, ::ptre::PushTensorRequest, ::ptre::PushTensorResponse>(
          std::mem_fn(&Tcp::Service::PushTensor), this)));
}

Tcp::Service::~Service() {
}

::grpc::Status Tcp::Service::PushTensor(::grpc::ServerContext* context, const ::ptre::PushTensorRequest* request, ::ptre::PushTensorResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace ptre

