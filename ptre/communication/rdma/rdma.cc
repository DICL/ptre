#include "ptre/communication/rdma/rdma.h"

#include <iostream>

namespace ptre {

int init_rdma_env(RdmaEnv& env) {
  int ret = 0;

  env.dev_list = ibv_get_device_list(nullptr);
  env.context = ibv_open_device(*env.dev_list);
  env.pd = ibv_alloc_pd(env.context);
  
  ret = ibv_query_port(env.context, IB_PORT, &env.port_attr);
  if (ret < 0) {
    std::cout << "ibv_query_port failed. ret=" << ret << std::endl;
    return ret;
  } else {
    std::cout << "ibv_query_port done." << std::endl;
  }

  ret = ibv_query_device(env.context, &env.dev_attr);
  if (ret < 0) {
    std::cout << "ibv_query_device failed. ret=" << ret << std::endl;
    return ret;
  } else {
    std::cout << "ibv_query_device done." << std::endl;
  }

  return ret;
}

/// ibv_post_send posts a linked list of WRs to a queue pair's (QP) send queue.
/// This operation is used to initiate all communication, including RDMA
/// operations. Processing of the WR list is stopped on the first error and a
/// pointer to the offending WR is returned in bad_wr.
///
/// The user should not alter or destroy AHs associated with WRs until the
/// request has been fully executed and a completion queue entry (CQE) has been
/// retrieved from the corresponding completion queue (CQ) to avoid unexpected
/// behaviour.
///
/// The buffers used by a WR can only be safely reused after the WR has been
/// fully executed and a WCE has been retrieved from the corresponding CQ.
/// However, if the IBV_SEND_INLINE flag was set, the buffer can be reused
/// immediately after the call returns.
int post_write(size_t buffer_size, uint64_t src_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               struct ibv_qp *qp) {
  int ret = 0;

  struct ibv_sge list;
  list.addr = src_addr;
  list.length = buffer_size;
  list.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  //wr.send_flags = IBV_SEND_SIGNALED;
  //wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(qp, &wr, &bad_wr);
  return ret;
}

}  // namespace ptre
