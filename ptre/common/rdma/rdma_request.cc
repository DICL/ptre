#include "ptre/common/rdma/rdma_request.h"

namespace ptre {
namespace common {

RdmaRequest::RdmaRequest() {
  mr_ = NULL;
  done_ = false;
}

int RdmaRequest::Join() {
#ifdef RDMA_REQUEST_BUSY_WAIT
  bool ret = false;
  do {
    mu_.lock();
    if (done_) ret = true;
    mu_.unlock();
  } while (!ret);
#else
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return done_; });
  lk.unlock();
#endif
  return status_;
}

void RdmaRequest::Done() {
  {
    std::lock_guard<std::mutex> guard(mu_);
    status_ = 0;
    done_ = true;
  }
#ifndef RDMA_REQUEST_BUSY_WAIT
  cv_.notify_one();
#endif
}

void RdmaRequest::DoneFailure() {
  {
    std::lock_guard<std::mutex> guard(mu_);
    status_ = 1;
    done_ = true;
  }
#ifndef RDMA_REQUEST_BUSY_WAIT
  cv_.notify_one();
#endif
}

void RdmaRequest::set_mr(struct ibv_mr* mr) {
  mr_ = mr;
}

}  // namespace common
}  // namespace ptre
