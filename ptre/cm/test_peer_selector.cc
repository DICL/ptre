#include "ptre/cm/peer_selector.h"
#include <iostream>

using ptre::PeerSelectorInterface;
using ptre::PeerSelectorFactory;
using ptre::SelectionStrategy;

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cout << "Usage: " << argv[0] << " size rank strategy num_push" << endl;
    return 1;
  }
  int size = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int strategy = atoi(argv[3]);
  int num_push = 1;
  if (argc == 5) {
    num_push = atoi(argv[4]);
  }
  PeerSelectorInterface* selector_list[size];
  for (int i = 0; i < size; i++) {
    PeerSelectorInterface*& selector = selector_list[i];
    PeerSelectorFactory::NewPeerSelector(size, i,
                                         SelectionStrategy(strategy),
                                         selector,
                                         num_push);
  }
  int count[size] = {};
  PeerSelectorInterface* selector = selector_list[rank];
  for (int i = 0; i < 30; i++) {
    for (int j = 0; j < num_push; j++) {
      int peer = selector->get_peer();
      cout << peer << " ";
    }
    cout << endl;
    //((ptre::MovingDHTRoundRobinSelector*) selector)->increase_delta();
  }
  for (int i = 0; i < size; i++) {
    delete selector_list[i];
  }
  return 0;
}
