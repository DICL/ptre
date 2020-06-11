#include "ptre/common/communication/rdma/rdma.h"

#include <iostream>

namespace ptre {
namespace common {

struct ibv_cq* ptre_rdma_create_cq(RdmaEnv* rdma_env, int comp_vector) {
  struct ibv_cq* cq = ibv_create_cq(rdma_env->context, QUEUE_DEPTH_DEFAULT * 2,
                             NULL, NULL, comp_vector);
  if (cq == NULL) {
    std::cout << "Failed to create CQ" << std::endl;
  } else {
    LOG(INFO) << "[DEBUG] cqe=" << cq->cqe;
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
    uint64_t global_subnet_prefix, uint64_t global_interface_id, uint16_t dlid,
    uint32_t my_psn, uint32_t remote_psn) {
  /// RTR
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;

    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = dest_qp_num;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 1;
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
    attr.sq_psn = my_psn;
    attr.timeout = TIMEOUT_DEFAULT;
    attr.retry_cnt = RETRY_CNT_DEFAULT;
    attr.rnr_retry = 7; /* infinite */
    attr.max_rd_atomic = 1;

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

int ptre_rdma_connect_qp_local(struct ibv_qp* qp, uint32_t dest_qp_num,
    uint16_t dlid,
    uint32_t my_psn, uint32_t remote_psn) {
  /// RTR
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;

    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = dest_qp_num;
    attr.rq_psn = remote_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 0;
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
    attr.sq_psn = my_psn;
    attr.timeout = TIMEOUT_DEFAULT;
    attr.retry_cnt = RETRY_CNT_DEFAULT;
    attr.rnr_retry = 7; /* infinite */
    attr.max_rd_atomic = 1;

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

int rdma_qp_reset_to_rts(struct ibv_qp* qp, uint32_t remote_qpn,
    uint16_t remote_lid, uint32_t remote_psn, uint32_t my_psn) {
  int ret;
  /// Reset QP
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RESET: " << std::strerror(ret) << "(code=" << ret << ")";
    //exit(1);
    return 1;
  }

  /// Init QP
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
      | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX
      | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to INIT: " << std::strerror(ret) << "(code=" << ret << ")";
    return 1;  //exit(1);
  }

  /// INIT -> RTR
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = remote_psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = remote_lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num  = 1;
  ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE
      | IBV_QP_AV
      | IBV_QP_PATH_MTU
      | IBV_QP_DEST_QPN
      | IBV_QP_RQ_PSN
      | IBV_QP_MAX_DEST_RD_ATOMIC
      | IBV_QP_MIN_RNR_TIMER);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RTR: " << std::strerror(ret) << "(code=" << ret << ")";
    return 1;
    //exit(1);
  }

  /// INIT -> RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = my_psn;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE
      | IBV_QP_TIMEOUT
      | IBV_QP_RETRY_CNT
      | IBV_QP_RNR_RETRY
      | IBV_QP_SQ_PSN
      | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RTS: " << std::strerror(ret) << "(code=" << ret << ")";
    return 1;
    //exit(1);
  }
  return 0;
}

int ptre_poll_cq(struct ibv_cq* cq, int num_comps,
                                struct ibv_wc* wcs, int caller_id) {
  int cnt = 0;
  while (cnt < num_comps) {
    struct ibv_wc& wc = wcs[cnt];
    int new_comps = ibv_poll_cq(cq, num_comps - cnt, &wc);
    if (new_comps > 0) {
      for (int i = 0; i < new_comps; i++) {
        struct ibv_wc& curr_wc = wcs[cnt + i];
        if (curr_wc.status) {
          //LOG(ERROR) << "Failed to post send: error_code=" << curr_wc.status << ": " << std::strerror(curr_wc.status) << ", caller=" << caller_id;
          //exit(1);
          //std::cerr << "Bad wc status " << curr_wc.status << endl;
        }
        RdmaWrId* wr_id = reinterpret_cast<RdmaWrId*>(curr_wc.wr_id);
        //std::cout << "WorkCompletion (RdmaWrIdType=" << wr_id->write_type
        //    << ")\n";
        delete wr_id;
      }
      cnt += new_comps;
    } else if (new_comps < 0) {
      LOG(INFO) << "[DEBUG] ibv_poll_cq failed.";
    }
  }
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
  if (ret) {
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
  if (ret) {
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
  if (ret) {
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
  if (ret) {
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
  if (ret) {
    std::cout << "Failed to ibv_post_send" << std::endl;
  }
  return ret;
}

void rdma_poll_cq(struct ibv_cq* cq, int num_comps,
                                struct ibv_wc* wcs) {
  int cnt = 0;
  while (cnt < num_comps) {
    struct ibv_wc& wc = wcs[cnt];
    int new_comps = ibv_poll_cq(cq, num_comps - cnt, &wc);
    if (new_comps > 0) {
      for (int i = 0; i < new_comps; i++) {
        struct ibv_wc& curr_wc = wcs[cnt + i];
        if (curr_wc.status) {
          std::cout << "Bad wc status " << curr_wc.status << std::endl;
        }
        char* wr_id = (char*) curr_wc.wr_id;
        delete wr_id;
      }
      cnt += new_comps;
    } else if (new_comps < 0) {
      std::cout << "Failed to poll CQ.\n";
    }
  }
}

void rdma_modify_qp_rts(struct ibv_qp* qp, uint32_t remote_qpn,
    uint32_t remote_psn, uint16_t remote_lid, uint32_t my_psn) {
  int ret;
  /// Reset QP
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (ret) {
    LOG(ERROR) << "Failed to modify QP to RESET: " << std::strerror(ret) << "(code=" << ret << ")";
    return;
    //exit(1);
  }

  /// Init QP
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
      | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX
      | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (ret) {
    std::cout << "Failed to modify QP to INIT.\n";
  }

  /// INIT -> RTR
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_qpn;
  attr.rq_psn = remote_psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = remote_lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num  = 1;
  ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE
      | IBV_QP_AV
      | IBV_QP_PATH_MTU
      | IBV_QP_DEST_QPN
      | IBV_QP_RQ_PSN
      | IBV_QP_MAX_DEST_RD_ATOMIC
      | IBV_QP_MIN_RNR_TIMER);
  if (ret) {
    std::cout << "Failed to modify QP to RTR errno=" << ret << ": " << std::strerror(ret) << std::endl;
  }

  /// INIT -> RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = my_psn;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE
      | IBV_QP_TIMEOUT
      | IBV_QP_RETRY_CNT
      | IBV_QP_RNR_RETRY
      | IBV_QP_SQ_PSN
      | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret) {
    std::cout << "Failed to modify QP to RTS errno=" << ret << ": " << std::strerror(ret) << std::endl;
  }
}

uint64_t rdma_cas(uint64_t compare, uint64_t swap, struct ibv_qp* qp,
    struct ibv_cq* cq, struct ibv_mr* read_buf_mr, uint64_t remote_addr,
    uint32_t rkey, struct ibv_pd* pd) {
  bool dereg_read_buf_mr = false;
  uint64_t* read_buf_ptr;
  int ret;
  // Init MR for read buffer if not provided.
  if (read_buf_mr == NULL) {
    dereg_read_buf_mr = true;
    read_buf_ptr = new uint64_t();
    void* read_buf_addr = (void*) read_buf_ptr;
    read_buf_mr = ibv_reg_mr(pd, read_buf_addr, sizeof(uint64_t),
        IBV_ACCESS_LOCAL_WRITE);
    if (!read_buf_mr) {
      LOG(ERROR) << "Failed to register MR for send_buf";
    }
  }
  // Init SGE
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uint64_t) read_buf_mr->addr;
  sge.length = read_buf_mr->length;
  sge.lkey = read_buf_mr->lkey;
  // Init WR
  struct ibv_send_wr wr;
  while (true) {
    /// Init WR
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t) new char();
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.atomic.remote_addr = (uint64_t) remote_addr;
    wr.wr.atomic.compare_add = compare;
    wr.wr.atomic.swap = swap;
    wr.wr.atomic.rkey = rkey;
    // Post send WR of CAS
    struct ibv_send_wr* bad_wr;
    ret = ibv_post_send(qp, &wr, &bad_wr);
    if (ret) {
      LOG(ERROR) << "Failed to ibv_post_send : " << std::strerror(ret);
    }
    // Poll
    struct ibv_wc wc;
    rdma_poll_cq(cq, 1, &wc);
    if (!wc.status) {
      break;
    }
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr init_attr;
    ret = ibv_query_qp(qp, &attr,
          IBV_QP_STATE
        | IBV_QP_AV
        | IBV_QP_DEST_QPN,
        &init_attr);
    if (ret) {
      LOG(ERROR) << "Failed to query QP state: " << std::strerror(ret);
      exit(1);
    }
    if (attr.qp_state != IBV_QPS_RTS) {
      uint32_t dest_qp_num = attr.dest_qp_num;
      uint16_t dlid = attr.ah_attr.dlid;
      rdma_modify_qp_rts(qp, dest_qp_num, 0, dlid, 0);
    }
  }
  // Retrieve the read value
  uint64_t read_val = *((uint64_t*) read_buf_mr->addr);
  if (dereg_read_buf_mr) {
    ret = ibv_dereg_mr(read_buf_mr);
    if (ret) {
       LOG(ERROR) << "Failed to degregister MR: " << std::strerror(ret);
    }
    delete read_buf_ptr;
  }
  return read_val;
}

}  // namespace common
}  // namespace ptre
