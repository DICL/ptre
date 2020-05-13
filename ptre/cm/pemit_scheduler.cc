#include "ptre/cm/permit_scheduler.h"
#include "ptre/lib/cache_ctl.h"

namespace ptre {

void PermitScheduler::EnqueueRecvTask(int src_rank, int idx) {
  auto&& p = permit_table_[idx];
  p->Enqueue(src_rank);
}

bool Contains(std::deque<int>& dq, int elem) {
  return std::find(dq.begin(), dq.end(), elem) != dq.end();
}

void Permit::Enqueue(int src_rank, int rcv_state) {
  if (rcv_state == 1 && !Contains(dq_, src_rank)) {
    dq_.push_back(src_rank);
  } else {
    if (!Contains(dq_pending_, src_rank)) {
      dq_pending_.push_back(src_rank);
    }
  }
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
}

}  // namespace ptre
