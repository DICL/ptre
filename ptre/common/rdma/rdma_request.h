#ifndef PTRE_COMMON_RDMA_RDMA_REQUEST_H_
#define PTRE_COMMON_RDMA_RDMA_REQUEST_H_

#include <condition_variable>
#include <mutex>
#include <infiniband/verbs.h>

namespace ptre {
namespace common {

class RdmaRequest {
 public:
  RdmaRequest();
  int Join();
  void Done();
  void DoneFailure();
  int status() { return status_; }
  void set_mr(struct ibv_mr* mr);
  struct ibv_mr* mr() { return mr_; }
  void set_imm_data(uint32_t imm_data);
  uint32_t imm_data() const { return imm_data_; }

 private:
  volatile int status_;
  struct ibv_mr* mr_;
  std::mutex mu_;
#ifndef RDMA_REQUEST_BUSY_WAIT
  std::condition_variable cv_;
#endif
  uint32_t imm_data_;
  //volatile bool done_;
};

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_RDMA_RDMA_REQUEST_H_
