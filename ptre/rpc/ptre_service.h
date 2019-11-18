#ifndef TENSORFLOW_PTRE_RPC_PTRE_SERVICE_H_
#define TENSORFLOW_PTRE_RPC_PTRE_SERVICE_H_

//#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/async_stream.h"
#include "grpcpp/impl/codegen/async_unary_call.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/impl/codegen/rpc_method.h"
#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/stub_options.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "tensorflow/ptre/protobuf/ptre_service.pb.h"

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace tensorflow {

namespace grpc {

class PtreService final {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status GetRemoteAddress(
        ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
        GetRemoteAddressResponse* response) = 0;
  };
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status GetRemoteAddress(
        ::grpc::ClientContext* context, const GetRemoteAddressRequest& request,
        GetRemoteAddressResponse* response) GRPC_OVERRIDE;

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    const ::grpc::internal::RpcMethod rpcmethod_GetRemoteAddress_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr< ::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class AsyncService : public ::grpc::Service {
   public:
    AsyncService();
    virtual ~AsyncService();
    void RequestGetRemoteAddress(
        ::grpc::ServerContext* context, GetRemoteAddressRequest* request,
        ::grpc::ServerAsyncResponseWriter<GetRemoteAddressResponse>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
};

}  // namespace grpc

}  // namespace tensorflow

#endif  // TENSORFLOW_PTRE_RPC_PTRE_SERVICE_H_
