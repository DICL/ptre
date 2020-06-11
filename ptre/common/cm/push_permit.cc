#include "ptre/common/cm/push_permit.h"

#include <algorithm>
#include <thread>
#include <chrono>

#include "ptre/lib/cache_ctl.h"

namespace ptre {

//void PermitScheduler::EnqueueRecvTask(int src_rank, int idx) {
//  auto&& p = buf_table_[idx];
//  p->Enqueue(src_rank);
//}

Permit::Permit() {
  buf_ = (int*) malloc(sizeof(int));
  *buf_ = -1;
}

Permit::Permit(Allocator* a) {
  buf_ = (int*) a->Allocate(sizeof(int));
  *buf_ = -1;
}

bool Contains(std::deque<int>& dq, int elem) {
  return std::find(dq.begin(), dq.end(), elem) != dq.end();
}

int Permit::Enqueue(int src_rank, int rcv_state) {
  if (rcv_state == 1 && !Contains(dq_, src_rank)) {
    if (checker_.find(src_rank) == checker_.end()) {
      checker_[src_rank] = true;
      dq_.push_back(src_rank);
      return 0;
    }
  } else {
    if (!Contains(dq_pending_, src_rank)) {
      dq_pending_.push_back(src_rank);
      return 1;
    } else {
      return 2;
    }
  }
  return -1;
}

int Permit::Pop(int src_rank) {
  auto search = std::find(dq_.begin(), dq_.end(), src_rank);
  if (search != dq_.end()) {
    dq_.erase(search);
  }
  auto search_pending = std::find(dq_pending_.begin(), dq_pending_.end(),
      src_rank);
  if (search_pending != dq_pending_.end()) {
    dq_pending_.erase(search_pending);
  }
  return 0;
}

void Permit::SwapPendingQueue() {
  checker_.clear();
  dq_.swap(dq_pending_);
}

void Permit::Next() {
  if (dq_.size() > 0) {
    *buf_ = dq_.front();
    //cache_ctl::clflush((char*) &buf_, sizeof(buf_));
    dq_.pop_front();
  } else {
    *buf_ = -1;
    //cache_ctl::clflush((char*) &buf_, sizeof(buf_));
  }
}

void Permit::SetValue(int value) {
  *buf_ = value;
  //cache_ctl::clflush((char*) &buf_, sizeof(buf_));
}

}  // namespace ptre
