#include "ptre/cm/push_permit_manager.h"

#include <algorithm>

namespace ptre {

PushPermitManager::PushPermitManager(int comm_size, int num_vars) {
  comm_size_ = comm_size;
  num_vars_ = num_vars;

  permits_ = (int*) malloc(sizeof(int) * num_vars);
  for (int i = 0; i < num_vars; i++) {
    permits_[i] = -1;
  }

  peer_q_mus_.resize(num_vars_);
  peer_qs_.resize(num_vars_);
#if 0
  for (int i = 0; i < comm_size; i++) {
    permit_cnts_.emplace_back(0, i);
  }
  std::make_heap(permit_cnts_.begin(), permit_cnts_.end(),
      std::greater<std::pair<int, int>>());
#endif
}

int* PushPermitManager::PermitArrayPtr() {
  return permits_;
}

void PushPermitManager::EnqueuePeer(int dst) {
  for (int i = 0; i < num_vars_; i++) {
    peer_qs_[i].push(dst);
  }
}

void PushPermitManager::ClearPeerQueue(int idx) {
  auto&& mu = peer_q_mus_[idx];
  auto&& q = peer_qs_[idx];
  mu.lock();
  while (!q.empty()) {
    q.pop();
  }
  mu.unlock();
}

//void PushPermitManager::ReceiveNotify(int dst) {
//
//}

void PushPermitManager::NextPeer(int idx) {
  auto&& mu = peer_q_mus_[idx];
  auto&& q = peer_qs_[idx];

  int peer = -1;
  mu.lock();
  if (q.size() != 0) {
    peer = q.front();
    q.pop();
  }
  mu.unlock();

  permits_mu_.lock();
  permits_[idx] = peer;
  permits_mu_.unlock();
#if 0
  std::pop_heap(permit_cnts_.begin(), permit_cnts_.end(),
      std::greater<std::pair<int, int>>());
  auto& min_cnt_rank = permit_cnts_.back();
  permits_[idx] = min_cnt_rank.second;
  min_cnt_rank.first++;
  std::push_heap(permit_cnts_.being(), permit_cnts_.end(),
      std::greater<std::pair<int, int>>());
#endif
}

}  // namespace ptre
