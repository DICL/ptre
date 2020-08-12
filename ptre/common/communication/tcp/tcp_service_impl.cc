#include "ptre/common/communication/tcp/tcp_service_impl.h"

#include <chrono>
#include <mutex>
#include <thread>

#include "ptre/common/common.h"
#include "ptre/common/cm/ready_tensor.h"

namespace ptre {
namespace common {

void TcpServiceImpl::SetConsensusManager(ConsensusManager* cm) {
  cm_ = cm;
}

void TcpServiceImpl::SetCommBufTables(CommBufTable* sbt, CommBufTable* rbt,
                                      std::mutex* mu) {
  sendbuf_table_ = sbt;
  recvbuf_table_ = rbt;
  commbuf_table_mu_ = mu;
}

inline void SetBuf(PullTensorResponse* res, ReadyTensor* t) {
  res->set_buf(static_cast<const void*>(t->tensor_data().data()),
      t->AllocatedBytes());
}

grpc::Status TcpServiceImpl::PullTensor(grpc::ServerContext* context,
                                        const PullTensorRequest* request,
					PullTensorResponse* response) {
#if 0
  if (cm_ == nullptr) return grpc::Status::CANCELLED;
  int rank = request->src_rank();
  response->set_tensor_name(request->tensor_name());
  ReadyTensor* t = cm_->ready_tensor(request->tensor_name());
  if (request->sync_mode() == P2P_SYNC_MODE_STEP) {
    std::mutex& mu = t->mu();
    std::condition_variable& cv = t->cv();
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return (t->step() == request->src_step()); });
    SetBuf(response, t);
    response->set_status(0);
    lk.unlock();
  } else if (request->sync_mode() == P2P_SYNC_MODE_STEP_ASYNC) {
    std::lock_guard<std::mutex> guard(t->mu());
    if (t->step() >= request->src_step()) {
      SetBuf(response, t);
      response->set_status(0);
    } else {
      response->set_status(1);
    }
  } else {
    std::lock_guard<std::mutex> guard(t->mu());
    SetBuf(response, t);
  }
#elif 0
  auto& name = request->tensor_name();
  auto sm = ptre_global_->result_state[name];
  std::lock_guard<std::mutex> guard(sm->mu);
  if (sm->state == 1 || sm->state == 2) {
  response->set_tensor_name(request->tensor_name());
  ptre_global_->comm_buf_table[request->tensor_name()];
  res->set_buf(static_cast<const void*>(t->tensor_data().data()),
      t->AllocatedBytes());
#else
  auto& name = request->tensor_name();
  commbuf_table_mu_->lock();
  auto search = sendbuf_table_->find(name);
  commbuf_table_mu_->unlock();
  if (search == sendbuf_table_->end()) {
//if (name == "predictions_kernel_0") {
//  DVLOGR(0, cm_->rank()) << __FUNCTION__ << " sendbuf not set " << name;
//}
    response->set_status(1);
    return grpc::Status::OK;
  }
  auto sm = search->second.second;
  if (sm->state != SENDBUF_STATE_READY) {
//if (name == "predictions_kernel_0") {
//  DVLOGR(0, cm_->rank()) << __FUNCTION__ << " memcpy not done " << name;
//}
    response->set_status(1);
    return grpc::Status::OK;
  }
  auto t = search->second.first;
  response->set_buf(static_cast<const void*>(t->tensor_data().data()),
      t->AllocatedBytes());
  response->set_status(0);
  sm->mu.lock();
  sm->state = SENDBUF_STATE_INIT;
  sm->mu.unlock();
#endif
  return grpc::Status::OK;
}

grpc::Status TcpServiceImpl::PushTensor(grpc::ServerContext* context,
                                        const PushTensorRequest* request,
                                        PushTensorResponse* response) {
  int rank = request->src_rank();
  string tensor_name = request->tensor_name();
  string buf = request->buf();
  Tensor recv_tensor = cm_->global_consensus(tensor_name);
  std::copy(buf.begin(), buf.end(),
            const_cast<char*>(recv_tensor.tensor_data().data()));

/// message PushTensorResponse {
///   int32 dst_rank = 1;
///   string tensor_name = 2;
///   int32 status = 3;
/// }
  response->set_status(0);
  return grpc::Status::OK;
}
}
}
