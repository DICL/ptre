#ifndef PTRE_CM_PEER_SELECTOR_H_
#define PTRE_CM_PEER_SELECTOR_H_

#include <memory>
#include <string>
#include <vector>
#include <random>

namespace ptre {

using std::string;

enum SelectionStrategy {
  RANDOM,
  ROUND_ROBIN,
  PRIORITY_DIFF
};

class PeerSelectorInterface {
 public:
  PeerSelectorInterface(int comm_size, int comm_rank)
      : comm_size_(comm_size), comm_rank_(comm_rank) {}
  virtual int get_peer() = 0;

 protected:
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
      : PeerSelectorInterface(comm_size, comm_rank), prev_(comm_rank) {}
  int get_peer() override;

 private:
  int prev_;
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

class PeerSelectorFactory {
 public:
  static void NewPeerSelector(int comm_size, int comm_rank,
      SelectionStrategy strategy,
      PeerSelectorInterface* out_selector);
};

}  // namespace ptre

#endif  // PTRE_CM_PEER_SELECTOR_H_
