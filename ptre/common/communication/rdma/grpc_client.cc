#include "ptre/common/communication/rdma/grpc_client.h"

#include <iostream>
#include <sstream>

namespace ptre {
namespace common {

namespace {
using grpc::ClientContext;
}  // namespace

std::string grpc_target(int dst_rank) {
  /// 172.30.1.1 ib001
  /// 172.20.1.1 dumbo001
  std::stringstream ss;
  ss << "172.20.1." << (dst_rank + 1) << ":50051";
  return ss.str();
}

GrpcClient::GrpcClient(int src_rank, int dst_rank, const std::string& hostname)
    : comm_rank_(src_rank), dst_rank_(dst_rank), hostname_(hostname) {
  //std::string target(grpc_target(dst_rank_));
  //std::cout << "target: " << hostname << std::endl;
  grpc::ChannelArguments ch_args;
  ch_args.SetMaxReceiveMessageSize(-1);
  std::shared_ptr<grpc::Channel> ch = grpc::CreateCustomChannel(hostname,
      grpc::InsecureChannelCredentials(), ch_args);
  stub_ = Rdma::NewStub(ch);
  /*
  std::shared_ptr<::grpc::Channel> channel = grpc::CreateChannel(hostname,
      grpc::InsecureChannelCredentials());
  stub_ = Rdma::NewStub(channel);
  */
}

GrpcClient::~GrpcClient() {
  stub_.reset();
}

int GrpcClient::GetLID(uint16_t* remote_lid) {
  GetLIDRequest req;
  GetLIDResponse res;
  ClientContext ctx;
  grpc::Status status = stub_->GetLID(&ctx, req, &res);
  if (status.ok()) {
    *remote_lid = res.lid();
    return 0;
  } else {
    return 1;
  }
}

int GrpcClient::GetQPAttr(uint32_t* remote_qpn, uint32_t* remote_psn) {
  GetQPAttrRequest req;
  req.set_src_rank(comm_rank_);

  GetQPAttrResponse res;

  ClientContext ctx;
  grpc::Status status = stub_->GetQPAttr(&ctx, req, &res);
  if (status.ok()) {
    *remote_qpn = res.qpn();
    *remote_psn = res.psn();
    return 0;
  } else {
    return 1;
  }
}

int GrpcClient::GetRemoteAddress(const BufType type, const std::string& name,
    uint64_t* out_remote_addr, uint32_t* out_rkey) {
  GetRemoteAddressRequest req;
  req.set_buf_type(type);
  req.set_var_name(name);

  GetRemoteAddressResponse res;

  ClientContext ctx;
  grpc::Status status = stub_->GetRemoteAddress(&ctx, req, &res);

  if (status.ok()) {
    *out_remote_addr = res.remote_addr();
    *out_rkey = res.rkey();
    return 0;
  } else {
    return 1;
  }
}

bool GrpcClient::AttemptPush(int vstep) {
  AttemptPushRequest request;
  AttemptPushResponse response;
  ClientContext context;
  request.set_rank(comm_rank_);
  request.set_vstep(vstep);
  grpc::Status status = stub_->AttemptPush(&context, request, &response);
  if (status.ok()) {
    return response.available();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return false;
  }
}

int GrpcClient::NotifyPushDone(const string& var_name) {
  NotifyPushDoneRequest request;
  NotifyPushDoneResponse response;
  ClientContext context;
  request.set_src_rank(comm_rank_);
  request.set_var_name(var_name);
  grpc::Status status = stub_->NotifyPushDone(&context, request, &response);
  //std::cout << "\n Client NotifyPushDone\n";
}

bool GrpcClient::Barrier() {
  BarrierRequest request;
  BarrierResponse response;

  ClientContext context;
  grpc::Status status = stub_->Barrier(&context, request, &response);

  if (status.ok()) {
    return response.entered();
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return false;
  }
}

int GrpcClient::Recv(char* buf, size_t len, const string& name) {
  RecvRequest request;
  RecvResponse response;
  ClientContext context;
  request.set_dst_rank(comm_rank_);
  request.set_len(len);
  request.set_name(name);
  grpc::Status status = stub_->Recv(&context, request, &response);
  if (status.ok()) {
    /*
    LOG(INFO) << "GOT " << name << ": var[0]=" << ((float*) response.buf().data())[0];
    */
    memcpy((void*) buf, (void*) response.buf().data(), len);
    return 0;
  } else {
    LOG(ERROR) << "dst_rank=" << dst_rank_ << " error_code="
        << status.error_code() << ": " << status.error_message();
    return -1;
  }
}

int GrpcClient::GetPermit(const string& name) {
  GetPermitRequest req;
  req.set_var_name(name);

  GetPermitResponse res;

  ClientContext ctx;
  grpc::Status status = stub_->GetPermit(&ctx, req, &res);

  if (status.ok()) {
    return res.permit();
  } else {
    return -1;
  }
}

int GrpcClient::AttemptPushVar(const string& var_name) {
  AttemptPushVarRequest req;
  req.set_src_rank(comm_rank_);
  req.set_var_name(var_name);

  AttemptPushVarResponse res;
  ClientContext ctx;
  grpc::Status status = stub_->AttemptPushVar(&ctx, req, &res);

  //LOG(INFO) << "result=" << res.result();
  if (status.ok() && res.result() == 1) {
    return 0;
  } else {
    return 1;
  }
}

int GrpcClient::CancelPushVar(const string& var_name) {
  CancelPushVarRequest req;
  req.set_src_rank(comm_rank_);
  req.set_var_name(var_name);

  CancelPushVarResponse res;
  ClientContext ctx;
  grpc::Status status = stub_->CancelPushVar(&ctx, req, &res);

  if (status.ok()) {
    return 0;
  } else {
    return 1;
  }
}

//GrpcClient* GrpcClientCache::GetClient(int dst_rank) {
//  if (cache_.find(rank) == cache_.end()) {
//    auto client = new GrpcClient(rank_, dst_rank,
//        GrpcClient grpc_client(ptre_global.rank, i, grpc_hosts_[i]);
//}

}  // namespace common
}  // namespace ptre
