#include "ptre/common/rdma/rdma.h"

namespace ptre {
namespace common {

RdmaRequest::RdmaRequest() {
  mr_ = NULL;
  done_ = false;
}

int RdmaRequest::Join() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return done_; });
  lk.unlock();
  return status_;
}

void RdmaRequest::Done() {
  {
    std::lock_guard<std::mutex> guard(mu_);
    status_ = 0;
    done_ = true;
  }
  cv_.notify_all();
}

void RdmaRequest::DoneFailure() {
  {
    std::lock_guard<std::mutex> guard(mu_);
    status_ = 1;
    done_ = true;
  }
  cv_.notify_all();
}

void RdmaRequest::set_mr(struct ibv_mr* mr) {
  mr_ = mr;
}

}  // namespace common
}  // namespace ptre
