#ifndef PTRE_CORE_PTRE_H_
#define PTRE_CORE_PTRE_H_
//namespace ptre {
struct PtreGlobal {

  ConsensusManager consensus_manager;

  mutex mu;

  std::queue<PtreRequest> request_queue;

  // Background thread running PTRE communication.
  std::thread background_thread;

  int rank = 0;
  int size = 1;

  ~PtreGlobal() {
    if (background_thread.joinable()) {
      //shut_down = true;
      background_thread.join();
    }
  }

};
//}  // namespace ptre
#endif  // PTRE_CORE_PTRE_H_
