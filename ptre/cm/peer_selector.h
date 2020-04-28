#ifndef PTRE_CM_PEER_SELECTOR_H_
#define PTRE_CM_PEER_SELECTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <random>
#include <cmath>

namespace ptre {

using std::string;

enum SelectionStrategy {
  RANDOM,          // 0
  ROUND_ROBIN,     // 1
  DHT_RANDOM,      // 2
  DHT_ROUND_ROBIN, // 3
  ADJACENT,        // 4
  MOVING_DHT_RR,   // 5
  PRIORITY_DIFF,   // 6
  DIVN_ROUND_ROBIN // 7
};

class PeerSelectorInterface {
 public:
  PeerSelectorInterface(int comm_size, int comm_rank)
      : comm_size_(comm_size), comm_rank_(comm_rank), select_cnt_(0) { }
  virtual int get_peer() = 0;
  void next() { }

 protected:
  uint64_t select_cnt_;
  int comm_size_;
  int comm_rank_;
};

class RandomPeerSelector : public PeerSelectorInterface {
 public:
  RandomPeerSelector(int comm_size, int comm_rank)
      : PeerSelectorInterface(comm_size, comm_rank) {}
  int get_peer() override;
};

class RoundRobinPeerSelector : public PeerSelectorInterface {
 public:
  RoundRobinPeerSelector(int comm_size, int comm_rank)
      : PeerSelectorInterface(comm_size, comm_rank), prev_(comm_rank) { }
  int get_peer() override;

 private:
  int prev_;
};

class DHTRandomPeerSelector : public PeerSelectorInterface {
 public:
  DHTRandomPeerSelector(int comm_size, int comm_rank);
  int get_peer() override;

 private:
  int max_power_;
};

class DHTRoundRobinPeerSelector : public PeerSelectorInterface {
 public:
  DHTRoundRobinPeerSelector(int comm_size, int comm_rank);
  int get_peer() override;

 protected:
  int max_power_;
  int prev_;
};

class NextPeerSelector : public PeerSelectorInterface {
 public:
  NextPeerSelector(int comm_size, int comm_rank)
      : PeerSelectorInterface(comm_size, comm_rank) { }
  int get_peer() override;
};

class MovingDHTRoundRobinSelector : public DHTRoundRobinPeerSelector {
 public:
  MovingDHTRoundRobinSelector(int size, int rank);
  void increase_delta();
  int get_peer() override;

 protected:
  int delta_;
};


class DifferenceBasedPeerSelector : public PeerSelectorInterface {
 public:
  DifferenceBasedPeerSelector(int comm_size, int comm_rank);
  void update(int rank, float diff);
  int get_peer() override;

 private:
  void update_cdf();
  std::vector<float> diff_list_;
  std::vector<float> cdf_;
};

class DivNRoundRobinPeerSelector : public PeerSelectorInterface {
 public:
  DivNRoundRobinPeerSelector(int comm_size, int comm_rank, int num_push);
  int get_peer() override;

 protected:
  std::vector<std::vector<int>> elems_;
  std::vector<int> indices_;
  int num_push_;
  int div_idx_;
};

class PeerSelectorFactory {
 public:
  static void NewPeerSelector(int comm_size, int comm_rank,
      SelectionStrategy strategy,
      PeerSelectorInterface* &out_selector,
      int num_push = 1);
};

}  // namespace ptre

#endif  // PTRE_CM_PEER_SELECTOR_H_
