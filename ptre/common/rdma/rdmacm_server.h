#ifndef PTRE_COMMON_RDMA_RDMACM_SERVER_H_
#define PTRE_COMMON_RDMA_RDMACM_SERVER_H_

#include <string>
#include <unordered_map>

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

namespace ptre {
namespace common {

using std::string;

class RdmacmServer {
 public:
  RdmacmServer(const char* addr = NULL, int port = 50051);

 protected:
  void CmEventThreadLoop();

 protected:
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> shutdown_;

  std::thread cm_event_thread_;
  struct sockaddr_in sin_;
  struct rdma_event_channel* cm_channel_;
  struct rdma_cm_id* listen_id_;

  using KeyToCmId = std::unordered_map<string, struct rdma_cm_id*>;
  std::unordered_map<int, KeyToCmId> cm_id_table_;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_RDMA_RDMACM_SERVER_H_
