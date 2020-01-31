#include "peer_selector.h"
#include <unistd.h>

namespace ptre {

void PeerSelectorFactory::NewPeerSelector(int comm_size, int comm_rank,
    SelectionStrategy strategy,
    PeerSelectorInterface* &out_selector) {
  if (strategy == RANDOM) {
    out_selector = new RandomPeerSelector(comm_size, comm_rank);
  } else if (strategy == ROUND_ROBIN) {
    out_selector = new RoundRobinPeerSelector(comm_size, comm_rank);
  } else if (strategy == DHT_RANDOM) {
    out_selector = new DHTRandomPeerSelector(comm_size, comm_rank);
  } else if (strategy == DHT_ROUND_ROBIN) {
    out_selector = new DHTRoundRobinPeerSelector(comm_size, comm_rank);
  } else if (strategy == ADJACENT) {
    out_selector = new NextPeerSelector(comm_size, comm_rank);
  } else if (strategy == PRIORITY_DIFF) {
    out_selector = new DifferenceBasedPeerSelector(comm_size, comm_rank);
  } else {
    // Default: RANDOM
    out_selector = new RandomPeerSelector(comm_size, comm_rank);
  }
}

int RandomPeerSelector::get_peer() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(0, comm_size_ - 1);
  int ret = comm_rank_;
  while (ret == comm_rank_) {
    ret = distribution(gen);
  }
  return ret;
}

int RoundRobinPeerSelector::get_peer() {
  int ret = comm_rank_;
  while (ret == comm_rank_){
    ret = (prev_ + 1) % comm_size_;
    prev_ = ret;
  }
  return ret;
}

/// DHT Peer Selector
DHTRandomPeerSelector::DHTRandomPeerSelector(int comm_size, int comm_rank)
      : PeerSelectorInterface(comm_size, comm_rank) {
  max_power_ = log(comm_size) / log(2);
}

int DHTRandomPeerSelector::get_peer() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distribution(1, max_power_);
  int ret = comm_rank_;
  while (ret == comm_rank_) {
    int power = distribution(gen);
    int div = pow(2, power);
    ret = (comm_rank_ + comm_size_ / div) % comm_size_;
  }
  return ret;
}

DHTRoundRobinPeerSelector::DHTRoundRobinPeerSelector(int comm_size, int comm_rank)
      : PeerSelectorInterface(comm_size, comm_rank) {
  max_power_ = log(comm_size) / log(2);
  prev_ = 0;
}

int DHTRoundRobinPeerSelector::get_peer() {
  //std::uniform_int_distribution<int> distribution(1, max_power_);
  int ret = comm_rank_;
  while (ret == comm_rank_) {
    int power = prev_ + 1;
    int div = pow(2, power);
    ret = (comm_rank_ + comm_size_ / div) % comm_size_;
    prev_ = (prev_ + 1) % max_power_;
  }
  return ret;
}

int NextPeerSelector::get_peer() {
  int ret = (comm_rank_ + 1) % comm_size_;
  return ret;
}

DifferenceBasedPeerSelector::DifferenceBasedPeerSelector(
    int comm_size, int comm_rank) : PeerSelectorInterface(comm_size, comm_rank) {
  diff_list_.resize(comm_size_, 1.0);
  diff_list_[comm_rank_] = 0.0;
  cdf_.resize(comm_size_);
  update_cdf();
}

void DifferenceBasedPeerSelector::update_cdf() {
  cdf_[0] = diff_list_[0];
  for (int i = 1; i < comm_size_; i++) {
    cdf_[i] = cdf_[i - 1] + diff_list_[i];
  }
  float max = cdf_[comm_size_ - 1];
	for (int i = 0; i < comm_size_; i++) {
		cdf_[i] /= max;
	}
}

void DifferenceBasedPeerSelector::update(int rank, float diff) {
  diff_list_[rank] = diff;
  update_cdf();
}

int DifferenceBasedPeerSelector::get_peer() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  int ret = comm_rank_;
  while (ret == comm_rank_) {
    float r = distribution(gen);
    for (int i = 0; i < comm_size_; i++) {
      if (r < cdf_[i]) {
        ret = i;
        break;
      }
    }
  }
  return ret;
}

}  // namespace ptre
