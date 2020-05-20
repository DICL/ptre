#include "ptre/cm/push_permit.h"

#include <algorithm>

#include "ptre/lib/cache_ctl.h"

namespace ptre {

//void PermitScheduler::EnqueueRecvTask(int src_rank, int idx) {
//  auto&& p = permit_table_[idx];
//  p->Enqueue(src_rank);
//}

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

void Permit::SwapPendingQueue() {
  checker_.clear();
  dq_.swap(dq_pending_);
}

void Permit::Next() {
  if (dq_.size() > 0) {
    permit_ = dq_.front();
    cache_ctl::clflush((char*) &permit_, sizeof(permit_));
    dq_.pop_front();
  } else {
    permit_ = -1;
    cache_ctl::clflush((char*) &permit_, sizeof(permit_));
  }
}

void Permit::SetValue(int value) {
  permit_ = value;
  cache_ctl::clflush((char*) &permit_, sizeof(permit_));
}

}  // namespace ptre
