#include "ptre/communication/rdma/grpc_server.h"

#include <string>

namespace ptre {

//GrpcServer::~GrpcServer() {
//  if (t_ != nullptr) {
//    t_->join();
//  }
//}

grpc::Status RdmaServiceImpl::GetRemoteAddress(grpc::ServerContext* context,
                                const GetRemoteAddressRequest* request,
                                GetRemoteAddressResponse* response) {
  /// need rank
  /// 
  int rank = rdma_manager_->rank();
  std::string tensor_name = request->tensor_name();
  RemoteMR rmr = rdma_manager_->GetRemoteMR(tensor_name);

  response->set_rank(rank);
  response->set_tensor_name(tensor_name);
  MemoryRegion* mr_proto = response->add_mr();
  mr_proto->set_remote_addr(rmr.remote_addr);
  mr_proto->set_rkey(rmr.rkey);
  return grpc::Status::OK;
}

grpc::Status RdmaServiceImpl::GetRemoteEnv(grpc::ServerContext* context,
                                           const GetRemoteEnvRequest* request,
                                           GetRemoteEnvResponse* response) {
  int src_rank = request->rank();
  int rank = rdma_manager_->rank();
  RdmaEnv* env = rdma_manager_->rdma_env();
  
  response->set_rank(rank);
  response->set_lid(env->port_attr.lid);
  response->set_qpn(rdma_manager_->qp(src_rank)->qp_num);
  response->set_snp(env->gid.global.subnet_prefix);
  response->set_iid(env->gid.global.interface_id);

  return grpc::Status::OK;
}

void RdmaServiceImpl::SetRdmaManager(RdmaManager* rdma_manager) {
  rdma_manager_ = rdma_manager;
}

//void GrpcServer::SetRdmaManager(RdmaManager* rdma_manager) {
//  rdma_manager_ = rdma_manager;
//}

/* static */
void GrpcServer::RunServer(RdmaManager* rdma_manager) {
  RdmaServiceImpl service;
  service.SetRdmaManager(rdma_manager);
  std::string server_address("0.0.0.0:50051");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  //server_ = std::move(std::unique_ptr<grpc::Server>(builder.BuildAndStart()));
  auto server = builder.BuildAndStart();
  server->Wait();
}

}  // namespace ptre
