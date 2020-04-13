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
    //std::cout << "ibv_query_port done." << std::endl;
  }

  /// Local address
  {
    //union ibv_gid gid;
    int r = ibv_query_gid(env.context, IB_PORT, 0, &env.gid);
    if (r < 0) std::cout << "Failed to ibv_query_gid\n";
    //else std::cout << "GID: " << env.gid.global.subnet_prefix << ", " << env.gid.global.interface_id << std::endl;
  }

  ret = ibv_query_device(env.context, &env.dev_attr);
  if (ret < 0) {
    std::cout << "ibv_query_device failed. ret=" << ret << std::endl;
    return ret;
  } else {
    //std::cout << "ibv_query_device done." << std::endl;
  }

  return ret;
}

struct ibv_cq* ptre_rdma_create_cq(RdmaEnv* rdma_env, int comp_vector) {
  struct ibv_cq* cq = ibv_create_cq(rdma_env->context, QUEUE_DEPTH_DEFAULT * 2,
                             NULL, NULL, comp_vector);
  if (cq == NULL) {
    std::cout << "Failed to create CQ" << std::endl;
  }
  return cq;
}

struct ibv_qp* ptre_rdma_create_qp(RdmaEnv* rdma_env, struct ibv_cq* send_cq,
    struct ibv_cq* recv_cq) {
  /// Create QP
  struct ibv_qp* qp;
  {
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(ibv_qp_init_attr));
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.cap.max_send_wr = QUEUE_DEPTH_DEFAULT;
    qp_init_attr.cap.max_recv_wr = QUEUE_DEPTH_DEFAULT;
    qp_init_attr.cap.max_send_sge = 8;
    qp_init_attr.cap.max_recv_sge = 8;
    qp_init_attr.qp_type = IBV_QPT_RC;

    qp = ibv_create_qp(rdma_env->pd, &qp_init_attr);
    if (qp == NULL) {
      std::cout << "Failed to create QP" << std::endl;
    }
  }

  /// Init QP
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = IB_PORT;
    attr.qp_access_flags = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

    int attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    int r = ibv_modify_qp(qp, &attr, attr_mask);
    if (r < 0) {
      std::cout << "Failed to set QP to INIT" << std::endl;
    }
  }
  return qp;
}

int ptre_rdma_connect_qp(struct ibv_qp* qp, uint32_t dest_qp_num,
    uint64_t global_subnet_prefix, uint64_t global_interface_id,
    uint16_t dlid) {
  /// RTR
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;

    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = dest_qp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 64;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = global_subnet_prefix;
    attr.ah_attr.grh.dgid.global.interface_id = global_interface_id;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.dlid = dlid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = IB_PORT;

    int r = ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC |
                          IBV_QP_MIN_RNR_TIMER);
    if (r < 0) {
      std::cout << "Failed to ibv_modify_qp to RTR " << r << std::endl;
      return r;
    }
  }

  /// RTS
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = TIMEOUT_DEFAULT;
    attr.retry_cnt = RETRY_CNT_DEFAULT;
    attr.rnr_retry = 7; /* infinite */
    attr.max_rd_atomic = 64;

    int r = ibv_modify_qp(qp, &attr,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC);
    if (r < 0) {
      std::cout << "Failed to ibv_modify_qp to RTS " << r << std::endl;
      return r;
    }
  }
  return 0;
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
  wr.send_flags = IBV_SEND_SIGNALED;
  //wr.send_flags = IBV_SEND_INLINE;
  //wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret < 0) {
    std::cout << "Failed to ibv_post_send" << std::endl;
  }
  return ret;
}

int post_fetch_and_add(size_t buffer_size, uint64_t src_addr,
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
  wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr.send_flags = IBV_SEND_SIGNALED;
  //wr.send_flags = IBV_SEND_INLINE;
  //wr.imm_data = imm_data;
  wr.wr.atomic.remote_addr = remote_addr;
  wr.wr.atomic.compare_add = 1;
  wr.wr.atomic.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret < 0) {
    std::cout << "Failed to ibv_post_send" << std::endl;
  }
  return ret;
}

int post_atomic_cmp_and_swp(size_t buffer_size, uint64_t local_addr,
               uint32_t lkey,
               uint64_t remote_addr,
               uint32_t rkey,
               struct ibv_send_wr& wr,
               uint64_t wr_id,
               struct ibv_qp *qp, uint64_t compare_add, uint64_t swap) {
  int ret = 0;

  struct ibv_sge list;
  memset(&list, 0, sizeof(struct ibv_sge));
  list.addr = local_addr;
  list.length = buffer_size;
  list.lkey = lkey;

  wr.wr_id = wr_id;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  //wr.send_flags = IBV_SEND_INLINE;
  //wr.imm_data = imm_data;
  wr.wr.atomic.remote_addr = remote_addr;
  wr.wr.atomic.compare_add = compare_add;
  wr.wr.atomic.swap = swap;
  wr.wr.atomic.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret < 0) {
    std::cout << "Failed to ibv_post_send" << std::endl;
  }
  return ret;
}

int post_read(size_t buffer_size, uint64_t local_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               struct ibv_qp *qp) {
  int ret = 0;

  struct ibv_sge list;
  memset(&list, 0, sizeof(struct ibv_sge));
  list.addr = local_addr;
  list.length = buffer_size;
  list.lkey = lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = wr_id;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  //wr.send_flags = IBV_SEND_INLINE;
  //wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret < 0) {
    std::cout << "Failed to ibv_post_send" << std::endl;
  }
  return ret;
}

int post_atomic_add(size_t buffer_size, uint64_t src_addr,
               uint32_t lkey, uint64_t remote_addr,
               uint32_t rkey, uint64_t wr_id,
               struct ibv_qp *qp,
               uint64_t compare_add, uint64_t swap) {
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
  wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
  wr.send_flags = IBV_SEND_SIGNALED;
  //wr.send_flags = IBV_SEND_INLINE;
  //wr.imm_data = imm_data;
  wr.wr.atomic.remote_addr = remote_addr;
  wr.wr.atomic.compare_add = compare_add;
  wr.wr.atomic.swap = swap;
  wr.wr.atomic.rkey = rkey;

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(qp, &wr, &bad_wr);
  if (ret < 0) {
    std::cout << "Failed to ibv_post_send" << std::endl;
  }
  return ret;
}

RdmaTensorChannel::RdmaTensorChannel(const RdmaEnv* env,
                                     const RemoteTensorId& id)
    : env_(env), id_(id) {
  /// Create QP
  {
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(ibv_qp_init_attr));
    //qp_init_attr.send_cq = env_->cq;
    //qp_init_attr.recv_cq = env_->cq;
    qp_init_attr.cap.max_send_wr = QUEUE_DEPTH_DEFAULT;
    qp_init_attr.cap.max_recv_wr = QUEUE_DEPTH_DEFAULT;
    qp_init_attr.cap.max_send_sge = 8;
    qp_init_attr.cap.max_recv_sge = 8;
    qp_init_attr.qp_type = IBV_QPT_RC;

    qp_ = ibv_create_qp(env_->pd, &qp_init_attr);
    if (qp_ == nullptr) {
      std::cout << "Failed to create QP" << std::endl;
    }
  }

  /// Init QP
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = IB_PORT;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    int attr_mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    int r = ibv_modify_qp(qp_, &attr, attr_mask);
    if (r < 0) {
      std::cout << "Failed to set QP to INIT" << std::endl;
    }
  }
}

void RdmaTensorChannel::Connect(uint32_t dlid) {
  if (!connected_) {
    /// RTR
    {
      struct ibv_qp_attr attr;
      memset(&attr, 0, sizeof(ibv_qp_attr));
      attr.qp_state = IBV_QPS_RTR;

      attr.path_mtu = IBV_MTU_4096;
      attr.dest_qp_num = qp_->qp_num;
      attr.rq_psn = 0;
      attr.max_dest_rd_atomic = 64;
      attr.min_rnr_timer = 12;
      attr.ah_attr.is_global = 0;
      attr.ah_attr.dlid = dlid;
      attr.ah_attr.sl = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.port_num = IB_PORT;

      int r = ibv_modify_qp(qp_, &attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                            IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                            IBV_QP_MAX_DEST_RD_ATOMIC |
                            IBV_QP_MIN_RNR_TIMER);
      if (r < 0) {
        std::cout << "Failed to ibv_modify_qp to RTR " << r << std::endl;
      }
    }

    /// RTS
    {
      struct ibv_qp_attr attr;
      memset(&attr, 0, sizeof(ibv_qp_attr));
      attr.qp_state = IBV_QPS_RTS;
      attr.sq_psn = 0;
      attr.timeout = TIMEOUT_DEFAULT;
      attr.retry_cnt = RETRY_CNT_DEFAULT;
      attr.rnr_retry = 7; /* infinite */
      attr.max_rd_atomic = 64;

      int r = ibv_modify_qp(qp_, &attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                            IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC);
      if (r < 0) {
        std::cout << "Failed to ibv_modify_qp to RTS " << r << std::endl;
      }
    }
    connected_ = true;
  }
}

}  // namespace ptre
