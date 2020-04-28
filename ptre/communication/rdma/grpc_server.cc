#include "ptre/communication/rdma/grpc_server.h"
#include "tensorflow/core/platform/logging.h"

#include <iostream>

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

grpc::Status RdmaServiceImpl::GetRemoteParamAddress(grpc::ServerContext* context,
                                const GetRemoteParamAddressRequest* request,
                                GetRemoteParamAddressResponse* response) {
  int src_rank = request->rank();
  int rank = rdma_manager_->rank();
  RemoteMR rpmr = rdma_manager_->GetRemoteParamMR();

  response->set_rank(rank);
  MemoryRegion* mr_proto = response->add_mr();
  mr_proto->set_remote_addr(rpmr.remote_addr);
  mr_proto->set_rkey(rpmr.rkey);
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

grpc::Status RdmaServiceImpl::AttemptPush(grpc::ServerContext* context,
                                      const AttemptPushRequest* request,
                                      AttemptPushResponse* response) {
  int src_rank = request->rank();
  int src_vstep = request->vstep();
  if (cm_->CanReceive(src_rank, src_vstep)) {
    response->set_available(true);
  } else {
    response->set_available(false);
  }

  return grpc::Status::OK;
}

grpc::Status RdmaServiceImpl::AckPushDone(grpc::ServerContext* context,
                                      const AckPushDoneRequest* request,
                                      AckPushDoneResponse* response) {
  //std::cout << "\nServer got AckPushDone\n";
  int src_rank = request->rank();
  cm_->FinalizeRecv(src_rank);
  return grpc::Status::OK;
}

grpc::Status RdmaServiceImpl::Barrier(grpc::ServerContext* context,
                                      const BarrierRequest* request,
                                      BarrierResponse* response) {
  int this_rank = cm_->rank();
  //std::cout << "[RANK:" << this_rank << "] Server::Barrier: is_entered?=" << *barrier_variable_ << std::endl;
  response->set_entered(*barrier_variable_);
  return grpc::Status::OK;
}

grpc::Status RdmaServiceImpl::GetRemoteAddressV2(grpc::ServerContext* context,
    const GetRemoteAddressV2Request* request,
    GetRemoteAddressV2Response* response) {
  int src_rank = request->rank();
  BufType type = request->type();
  string name = request->name();
  struct ibv_mr* mr = rdma_manager_->GetMR(type, name);
  response->set_rank(rdma_manager_->rank());
  response->set_type(type);
  response->set_name(name);
  MemoryRegion* mr_proto = response->add_mr();
  uint64_t mr_addr = (uint64_t) mr->addr;
  mr_proto->set_remote_addr(mr_addr);
  mr_proto->set_rkey(mr->rkey);
  return grpc::Status::OK;
}

void RdmaServiceImpl::SetRdmaManager(RdmaManager* rdma_manager) {
  rdma_manager_ = rdma_manager;
}

void RdmaServiceImpl::SetConsensusManager(ConsensusManager* cm) {
  cm_ = cm;
}

void RdmaServiceImpl::SetBarrierVariable(bool* barrier_variable) {
    barrier_variable_ = barrier_variable;
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
