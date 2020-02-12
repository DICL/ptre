#include "ptre/communication/rdma/grpc_client.h"

#include <iostream>
#include <sstream>

namespace ptre {

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
    : src_rank_(src_rank), dst_rank_(dst_rank), hostname_(hostname) {
  //std::string target(grpc_target(dst_rank_));
  std::cout << "target: " << hostname << std::endl;
  std::shared_ptr<::grpc::Channel> channel = grpc::CreateChannel(hostname,
      grpc::InsecureChannelCredentials());
  stub_ = Rdma::NewStub(channel);
}

GrpcClient::~GrpcClient() {
  stub_.reset();
}

void GrpcClient::SetRdmaManager(RdmaManager* rdma_manager) {
  rdma_manager_ = rdma_manager;
}

int GrpcClient::GetRemoteAddress(const std::string& name) {
  GetRemoteAddressRequest request;
  request.set_rank(src_rank_);
  request.set_tensor_name(name);

  GetRemoteAddressResponse response;

  ClientContext context;
  grpc::Status status = stub_->GetRemoteAddress(&context, request, &response);

	if (status.ok()) {
    rdma_manager_->SetRemoteMR(dst_rank_, name, response.mr()[0].remote_addr(),
                               response.mr()[0].rkey());
    return 0;
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return -1;
  }
}

int GrpcClient::GetRemoteParamAddress() {
  GetRemoteParamAddressRequest request;
  request.set_rank(src_rank_);

  GetRemoteParamAddressResponse response;

  ClientContext context;
  grpc::Status status = stub_->GetRemoteParamAddress(&context, request, &response);
  if (status.ok()) {
    rdma_manager_->SetRemoteParamMR(dst_rank_, response.mr()[0].remote_addr(),
                                    response.mr()[0].rkey());
    return 0;
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return -1;
  }
}

int GrpcClient::GetRemoteEnv() {
  GetRemoteEnvRequest request;
  request.set_rank(src_rank_);

  GetRemoteEnvResponse response;

  ClientContext context;
  grpc::Status status = stub_->GetRemoteEnv(&context, request, &response);

	if (status.ok()) {
    rdma_manager_->SetDlid(dst_rank_, response.lid());
    rdma_manager_->set_qpn(dst_rank_, response.qpn());
    rdma_manager_->set_snp(dst_rank_, response.snp());
    rdma_manager_->set_iid(dst_rank_, response.iid());
    return 0;
  } else {
    std::cout << status.error_code() << ": " << status.error_message()
              << std::endl;
    return -1;
  }
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

//GrpcClient* GrpcClientCache::GetClient(int dst_rank) {
//  if (cache_.find(rank) == cache_.end()) {
//    auto client = new GrpcClient(rank_, dst_rank, 
//        GrpcClient grpc_client(ptre_global.rank, i, grpc_hosts_[i]);
//}

}  // namespace ptre
