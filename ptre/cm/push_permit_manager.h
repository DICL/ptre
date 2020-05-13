#ifndef PTRE_CM_PUSH_PERMIT_MANAGER_H_
#define PTRE_CM_PUSH_PERMIT_MANAGER_H_

#include <string>
#include <vector>


namespace ptre {

using std::string;

class PushPermitManager {
 public:
  //void ReceiveNotify(int dst);
  int* PermitArrayPtr();
  void EnqueuePeer(int dst);
  void ClearPeerQueue(int idx);
  void NextPeer(int idx);

 private:
  int comm_size_;
  int num_vars_;
  std::map<string, int> name_to_idx_;
  std::vector<string> names_;
  //std::vector<std::pair<int, int>> permit_cnts_;
  std::vector<std::mutex> peer_q_mus_;
  std::vector<std::queue<int>> peer_qs_;

  std::mutex permits_mu_;
  int* permits_;
};

}  // namespace ptre

#endif  // PTRE_CM_PUSH_PERMIT_MANAGER_H_
