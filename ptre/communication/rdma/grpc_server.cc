#include "ptre/communication/rdma/grpc_server.h"
#include "tensorflow/core/platform/logging.h"

#include <iostream>

namespace ptre {

//GrpcServer::~GrpcServer() {
//  if (t_ != nullptr) {
//    t_->join();
//  }
//}

grpc::Status RdmaServiceImpl::GetRemoteAddress(grpc::ServerContext* ctx,
                                const GetRemoteAddressRequest* req,
                                GetRemoteAddressResponse* res) {
  struct ibv_mr* mr = rdma_manager_->GetMR(req->buf_type(), req->var_name());
  res->set_remote_addr((uint64_t) mr->addr);
  res->set_rkey(mr->rkey);
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

grpc::Status RdmaServiceImpl::NotifyPushDone(grpc::ServerContext* context,
                                      const NotifyPushDoneRequest* request,
                                      NotifyPushDoneResponse* response) {
  //std::cout << "\nServer got NotifyPushDone\n";
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

grpc::Status RdmaServiceImpl::Recv(grpc::ServerContext* context,
    const RecvRequest* request, RecvResponse* response) {
  int dst_rank = request->dst_rank();
  size_t len = request->len();
  string name = request->name();
  mu_.lock();
  if (send_q_cache_.find(dst_rank) == send_q_cache_.end()) {
    send_q_cache_.emplace(dst_rank,
        std::map<string, ConcurrentQueue<string>*>());
  }
  auto&& q_map = send_q_cache_[dst_rank];
  if (q_map.find(name) == q_map.end()) {
    q_map[name] = new ConcurrentQueue<string>();
  }
  auto&& q = q_map[name];
  mu_.unlock();
  string send_buf;
  q->wait_and_pop(send_buf);
  response->set_buf(std::move(send_buf));
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

void RdmaServiceImpl::Send(int dst_rank, char* buf, size_t len,
    const string& name) {
  mu_.lock();
  if (send_q_cache_.find(dst_rank) == send_q_cache_.end()) {
    send_q_cache_.emplace(dst_rank,
        std::map<string, ConcurrentQueue<string>*>());
  }
  auto&& q_map = send_q_cache_[dst_rank];
  if (q_map.find(name) == q_map.end()) {
    q_map[name] = new ConcurrentQueue<string>();
  }
  auto&& q = q_map[name];
  mu_.unlock();
  string send_buf(buf, len);
  q->push(std::move(send_buf));
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
