// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: rdma_service.proto
#ifndef GRPC_rdma_5fservice_2eproto__INCLUDED
#define GRPC_rdma_5fservice_2eproto__INCLUDED

#include "rdma_service.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/method_handler_impl.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class Channel;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace ptre {

class Rdma final {
 public:
  static constexpr char const* service_full_name() {
    return "ptre.Rdma";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status GetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::ptre::GetRemoteEnvResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteEnvResponse>> AsyncGetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteEnvResponse>>(AsyncGetRemoteEnvRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteEnvResponse>> PrepareAsyncGetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteEnvResponse>>(PrepareAsyncGetRemoteEnvRaw(context, request, cq));
    }
    virtual ::grpc::Status GetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::ptre::GetRemoteAddressResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteAddressResponse>> AsyncGetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteAddressResponse>>(AsyncGetRemoteAddressRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteAddressResponse>> PrepareAsyncGetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteAddressResponse>>(PrepareAsyncGetRemoteAddressRaw(context, request, cq));
    }
    class experimental_async_interface {
     public:
      virtual ~experimental_async_interface() {}
      virtual void GetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response, std::function<void(::grpc::Status)>) = 0;
      virtual void GetRemoteEnv(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::ptre::GetRemoteEnvResponse* response, std::function<void(::grpc::Status)>) = 0;
      virtual void GetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response, std::function<void(::grpc::Status)>) = 0;
      virtual void GetRemoteAddress(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::ptre::GetRemoteAddressResponse* response, std::function<void(::grpc::Status)>) = 0;
    };
    virtual class experimental_async_interface* experimental_async() { return nullptr; }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteEnvResponse>* AsyncGetRemoteEnvRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteEnvResponse>* PrepareAsyncGetRemoteEnvRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteAddressResponse>* AsyncGetRemoteAddressRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::ptre::GetRemoteAddressResponse>* PrepareAsyncGetRemoteAddressRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status GetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::ptre::GetRemoteEnvResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteEnvResponse>> AsyncGetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteEnvResponse>>(AsyncGetRemoteEnvRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteEnvResponse>> PrepareAsyncGetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteEnvResponse>>(PrepareAsyncGetRemoteEnvRaw(context, request, cq));
    }
    ::grpc::Status GetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::ptre::GetRemoteAddressResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteAddressResponse>> AsyncGetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteAddressResponse>>(AsyncGetRemoteAddressRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteAddressResponse>> PrepareAsyncGetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteAddressResponse>>(PrepareAsyncGetRemoteAddressRaw(context, request, cq));
    }
    class experimental_async final :
      public StubInterface::experimental_async_interface {
     public:
      void GetRemoteEnv(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response, std::function<void(::grpc::Status)>) override;
      void GetRemoteEnv(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::ptre::GetRemoteEnvResponse* response, std::function<void(::grpc::Status)>) override;
      void GetRemoteAddress(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response, std::function<void(::grpc::Status)>) override;
      void GetRemoteAddress(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::ptre::GetRemoteAddressResponse* response, std::function<void(::grpc::Status)>) override;
     private:
      friend class Stub;
      explicit experimental_async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class experimental_async_interface* experimental_async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class experimental_async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteEnvResponse>* AsyncGetRemoteEnvRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteEnvResponse>* PrepareAsyncGetRemoteEnvRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteEnvRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteAddressResponse>* AsyncGetRemoteAddressRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::ptre::GetRemoteAddressResponse>* PrepareAsyncGetRemoteAddressRaw(::grpc::ClientContext* context, const ::ptre::GetRemoteAddressRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_GetRemoteEnv_;
    const ::grpc::internal::RpcMethod rpcmethod_GetRemoteAddress_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response);
    virtual ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_GetRemoteEnv : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_GetRemoteEnv() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_GetRemoteEnv() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetRemoteEnv(::grpc::ServerContext* context, ::ptre::GetRemoteEnvRequest* request, ::grpc::ServerAsyncResponseWriter< ::ptre::GetRemoteEnvResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithAsyncMethod_GetRemoteAddress : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithAsyncMethod_GetRemoteAddress() {
      ::grpc::Service::MarkMethodAsync(1);
    }
    ~WithAsyncMethod_GetRemoteAddress() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetRemoteAddress(::grpc::ServerContext* context, ::ptre::GetRemoteAddressRequest* request, ::grpc::ServerAsyncResponseWriter< ::ptre::GetRemoteAddressResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_GetRemoteEnv<WithAsyncMethod_GetRemoteAddress<Service > > AsyncService;
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_GetRemoteEnv : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithCallbackMethod_GetRemoteEnv() {
      ::grpc::Service::experimental().MarkMethodCallback(0,
        new ::grpc::internal::CallbackUnaryHandler< ::ptre::GetRemoteEnvRequest, ::ptre::GetRemoteEnvResponse>(
          [this](::grpc::ServerContext* context,
                 const ::ptre::GetRemoteEnvRequest* request,
                 ::ptre::GetRemoteEnvResponse* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   return this->GetRemoteEnv(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithCallbackMethod_GetRemoteEnv() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_GetRemoteAddress : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithCallbackMethod_GetRemoteAddress() {
      ::grpc::Service::experimental().MarkMethodCallback(1,
        new ::grpc::internal::CallbackUnaryHandler< ::ptre::GetRemoteAddressRequest, ::ptre::GetRemoteAddressResponse>(
          [this](::grpc::ServerContext* context,
                 const ::ptre::GetRemoteAddressRequest* request,
                 ::ptre::GetRemoteAddressResponse* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   return this->GetRemoteAddress(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithCallbackMethod_GetRemoteAddress() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  typedef ExperimentalWithCallbackMethod_GetRemoteEnv<ExperimentalWithCallbackMethod_GetRemoteAddress<Service > > ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_GetRemoteEnv : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_GetRemoteEnv() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_GetRemoteEnv() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithGenericMethod_GetRemoteAddress : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithGenericMethod_GetRemoteAddress() {
      ::grpc::Service::MarkMethodGeneric(1);
    }
    ~WithGenericMethod_GetRemoteAddress() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_GetRemoteEnv : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithRawMethod_GetRemoteEnv() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_GetRemoteEnv() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetRemoteEnv(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawMethod_GetRemoteAddress : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithRawMethod_GetRemoteAddress() {
      ::grpc::Service::MarkMethodRaw(1);
    }
    ~WithRawMethod_GetRemoteAddress() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestGetRemoteAddress(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(1, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_GetRemoteEnv : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithRawCallbackMethod_GetRemoteEnv() {
      ::grpc::Service::experimental().MarkMethodRawCallback(0,
        new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
          [this](::grpc::ServerContext* context,
                 const ::grpc::ByteBuffer* request,
                 ::grpc::ByteBuffer* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   this->GetRemoteEnv(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithRawCallbackMethod_GetRemoteEnv() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void GetRemoteEnv(::grpc::ServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_GetRemoteAddress : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    ExperimentalWithRawCallbackMethod_GetRemoteAddress() {
      ::grpc::Service::experimental().MarkMethodRawCallback(1,
        new ::grpc::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
          [this](::grpc::ServerContext* context,
                 const ::grpc::ByteBuffer* request,
                 ::grpc::ByteBuffer* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   this->GetRemoteAddress(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithRawCallbackMethod_GetRemoteAddress() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void GetRemoteAddress(::grpc::ServerContext* context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GetRemoteEnv : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_GetRemoteEnv() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::ptre::GetRemoteEnvRequest, ::ptre::GetRemoteEnvResponse>(std::bind(&WithStreamedUnaryMethod_GetRemoteEnv<BaseClass>::StreamedGetRemoteEnv, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_GetRemoteEnv() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetRemoteEnv(::grpc::ServerContext* context, const ::ptre::GetRemoteEnvRequest* request, ::ptre::GetRemoteEnvResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGetRemoteEnv(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::ptre::GetRemoteEnvRequest,::ptre::GetRemoteEnvResponse>* server_unary_streamer) = 0;
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_GetRemoteAddress : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service *service) {}
   public:
    WithStreamedUnaryMethod_GetRemoteAddress() {
      ::grpc::Service::MarkMethodStreamed(1,
        new ::grpc::internal::StreamedUnaryHandler< ::ptre::GetRemoteAddressRequest, ::ptre::GetRemoteAddressResponse>(std::bind(&WithStreamedUnaryMethod_GetRemoteAddress<BaseClass>::StreamedGetRemoteAddress, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_GetRemoteAddress() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status GetRemoteAddress(::grpc::ServerContext* context, const ::ptre::GetRemoteAddressRequest* request, ::ptre::GetRemoteAddressResponse* response) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedGetRemoteAddress(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::ptre::GetRemoteAddressRequest,::ptre::GetRemoteAddressResponse>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_GetRemoteEnv<WithStreamedUnaryMethod_GetRemoteAddress<Service > > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_GetRemoteEnv<WithStreamedUnaryMethod_GetRemoteAddress<Service > > StreamedService;
};

}  // namespace ptre


#endif  // GRPC_rdma_5fservice_2eproto__INCLUDED
