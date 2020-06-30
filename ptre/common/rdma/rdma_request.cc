#include "ptre/common/rdma/rdma_request.h"
//#include "ptre/lib/cache_ctl.h"

namespace ptre {
namespace common {

RdmaRequest::RdmaRequest() {
  mr_ = NULL;
  //done_ = false;
  status_ = -1;
}

void RdmaRequest::Clear() {
  mr_ = NULL;
  status_ = -1;
}

int RdmaRequest::Join() {
#ifdef RDMA_REQUEST_BUSY_WAIT
  while (status_ < 0) continue;
#else
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return (status_ >= 0); });
  lk.unlock();
#endif
  return status_;
}

void RdmaRequest::Done() {
  std::lock_guard<std::mutex> guard(mu_);
  status_ = 0;
  //cache_ctl::clflush((char*) &status_, sizeof(status_));
  //done_ = true;
  //cache_ctl::clflush((char*) &done_, sizeof(done_));
#ifndef RDMA_REQUEST_BUSY_WAIT
  cv_.notify_one();
#endif
}

void RdmaRequest::DoneFailure() {
  std::lock_guard<std::mutex> guard(mu_);
  status_ = 1;
  //done_ = true;
#ifndef RDMA_REQUEST_BUSY_WAIT
  cv_.notify_one();
#endif
}

void RdmaRequest::set_mr(struct ibv_mr* mr) {
  mr_ = mr;
}

void RdmaRequest::set_imm_data(uint32_t imm_data) {
  imm_data_ = imm_data;
}

}  // namespace common
}  // namespace ptre
