#ifndef PTRE_CM_PUSH_PERMIT_H_
#define PTRE_CM_PUSH_PERMIT_H_

#include <deque>
#include <map>

#include "ptre/core/allocator.h"

namespace ptre {

//class PermitScheduler {
// public:
//  void EnqueueRecvTask(int src_rank, int idx);
//};

class Permit {
 public:
  Permit();
  Permit(Allocator* a);
  int* data() { return buf_; }
  int value() { return *buf_; }
  int Enqueue(int src_rank, int rcv_state);
  void SwapPendingQueue();
  void Next();
  void SetValue(int value);

 private:
  int* buf_;
  std::deque<int> dq_;
  std::deque<int> dq_pending_;
  std::map<int, bool> checker_;
};

}  // namespace ptre

#endif  // PTRE_CM_PUSH_PERMIT_H_
