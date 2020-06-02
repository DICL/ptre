#include "ptre/communication/rdma/grpc_server.h"
#include "tensorflow/core/platform/logging.h"

#include <iostream>

namespace ptre {

//GrpcServer::~GrpcServer() {
//  if (t_ != nullptr) {
//    t_->join();
//  }
//}
grpc::Status RdmaServiceImpl::GetLID(grpc::ServerContext* ctx,
                                const GetLIDRequest* req,
                                GetLIDResponse* res) {
  if (rdma_manager_ != nullptr) {
    res->set_lid(rdma_manager_->port_attr().lid);
    return grpc::Status::OK;
  } else {
    return grpc::Status::CANCELLED;
  }
}

grpc::Status RdmaServiceImpl::GetQPAttr(grpc::ServerContext* ctx,
                                const GetQPAttrRequest* req,
                                GetQPAttrResponse* res) {
  if (rdma_manager_ != nullptr) {
    int src_rank = req->src_rank();
    struct ibv_qp* qp = rdma_manager_->qp(src_rank);
    if (qp) {
      res->set_qpn(qp->qp_num);
      // TODO: Support Custom PSN
      res->set_psn(0);
      return grpc::Status::OK;
    }
  }
  return grpc::Status::CANCELLED;
}

grpc::Status RdmaServiceImpl::GetRemoteAddress(grpc::ServerContext* ctx,
                                const GetRemoteAddressRequest* req,
                                GetRemoteAddressResponse* res) {
  if (rdma_manager_ != nullptr) {
    struct ibv_mr* mr = rdma_manager_->GetMR(req->buf_type(), req->var_name());
    if (mr) {
      res->set_remote_addr((uint64_t) mr->addr);
      res->set_rkey(mr->rkey);
      return grpc::Status::OK;
    }
  }
  return grpc::Status::CANCELLED;
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
  usleep(100);
  int src_rank = request->src_rank();
  auto rvar = cm_->remote_variable(request->var_name());
  if (rvar) {
    struct ibv_mr* mr = rdma_manager_->GetMR(ptre::BUF_TYPE_RECV_BUF, request->var_name());
    LOG(INFO) << "\n"
        << "WRITTEN " << request->var_name() << ":\n"
        << "rcv[0]=" << ((float*) rvar->rcv_data())[0] <<"\n"
        << "rcv[15]=" << ((float*) rvar->rcv_data())[15] <<"\n"
        << "rcv[16]=" << ((float*) rvar->rcv_data())[16] <<"\n"
        << "addr=" << mr->addr << ", rkey=" << mr->rkey << ", rcv_buf=" << rvar->rcv_data();
    rvar->NewIncoming(src_rank);
  }
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
  //LOG(INFO) << "WILLBESENT " << name << ": var[0]=" << ((float*) send_buf.data())[0];
  response->set_buf(send_buf);
  return grpc::Status::OK;
}


grpc::Status RdmaServiceImpl::GetPermit(grpc::ServerContext* context,
    const GetPermitRequest* request, GetPermitResponse* response) {
  auto rvar = cm_->remote_variable(request->var_name());
  if (rvar) {
    response->set_permit(rvar->permit());
  }
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
  //q->push(std::move(send_buf));
  q->push(send_buf);
}

grpc::Status RdmaServiceImpl::AttemptPushVar(grpc::ServerContext* context,
                                      const AttemptPushVarRequest* request,
                                      AttemptPushVarResponse* response) {
  //LOG(INFO) << "Got AttemptPushVar: " << request->var_name() << ", src=" << request->src_rank();
  int src_rank = request->src_rank();
  string var_name = request->var_name();

  auto rvar = cm_->remote_variable(var_name);
  if (rvar) {
    rvar->EnqueueSenderCandidate(src_rank);
    response->set_result(1);
  } else {
    response->set_result(0);
  }

  return grpc::Status::OK;
}

grpc::Status RdmaServiceImpl::CancelPushVar(grpc::ServerContext* context,
                                      const CancelPushVarRequest* request,
                                      CancelPushVarResponse* response) {
  int src_rank = request->src_rank();
  string var_name = request->var_name();
  auto rvar = cm_->remote_variable(var_name);
  if (rvar) {
    rvar->PopSenderCandidate(src_rank);
  }
  return grpc::Status::OK;
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
