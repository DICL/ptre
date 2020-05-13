#ifndef PTRE_CM_PERMIT_SCHEDULER_H_
#define PTRE_CM_PERMIT_SCHEDULER_H_

#include <deque>

namespace ptre {

class PermitScheduler {
 public:
  void EnqueueRecvTask(int src_rank, int idx);
};

class Permit {
 public:
  int* data() { return &permit_; }
  int value() { return permit_; }

  void Enqueue(int src_rank);
  void Next();
  void SetValue(int value);

 private:
  int permit_;
  std::deque<int> dq_;
  std::deque<int> dq_pending_;
};

}  // namespace ptre

#endif  // PTRE_CM_PERMIT_SCHEDULER_H_
