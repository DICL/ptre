#include "ptre/common/communication/tcp/tcp_service_impl.h"

#include <mutex>

#include "ptre/common/cm/ready_tensor.h"

namespace ptre {
namespace common {

void TcpServiceImpl::SetConsensusManager(ConsensusManager* cm) {
  cm_ = cm;
}

inline void SetBuf(PullTensorResponse* res, ReadyTensor* t) {
  res->set_buf(static_cast<const void*>(t->tensor_data().data()),
      t->AllocatedBytes());
}

grpc::Status TcpServiceImpl::PullTensor(grpc::ServerContext* context,
                                        const PullTensorRequest* request,
					PullTensorResponse* response) {
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
