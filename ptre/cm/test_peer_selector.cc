#include "ptre/cm/peer_selector.h"
#include <iostream>

using ptre::PeerSelectorInterface;
using ptre::PeerSelectorFactory;
using ptre::SelectionStrategy;

int main() {
  int size = 45;
  int rank = 0;
  PeerSelectorInterface* selector_list[size];
  for (int rank = 0; rank < 45; rank++ ) {
    PeerSelectorInterface*& selector = selector_list[rank];
    PeerSelectorFactory::NewPeerSelector(size, rank,
                                         SelectionStrategy(4),
                                         selector);
  }
  for (int i = 0; i < 10; i++) {
    for (int rank = 0; rank < 45; rank++ ) {
      PeerSelectorInterface* selector = selector_list[rank];
      std::cout << selector->get_peer() << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
