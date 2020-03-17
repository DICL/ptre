#ifndef PTRE_COMMUNICATION_RDMA_RDMA_TASK_H_
#define PTRE_COMMUNICATION_RDMA_RDMA_TASK_H_

#include "ptre/communication/rdma/rdma.h"

enum RdmaTaskType {
  CAS1,
  CAS2
};

struct RdmaTask {
  RdmaTaskType type;
  struct ibv_qp* qp;  // Not owned.
  struct ibv_send_wr* wr;  // Owned.
  struct ibv_send_wr* bad_wr;  // Owned.
  float* flat;

  ~RdmaTask() {
    delete[] wr->sg_list;
    delete wr;
    delete bad_wr;
  }
};

#endif  // PTRE_COMMUNICATION_RDMA_RDMA_TASK_H_
