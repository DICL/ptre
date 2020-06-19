#include "ptre/common/operations.h"
#include "ptre/common/rdma/rdma_mpi.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/ptre_global.h"
#include "/home/wkim/wild/mpi_tests/allreduce_test.h"

#include <string>
#include <cstdio>
#include <vector>
#include <chrono>

using namespace std;
using namespace ptre::common;

int main(int argc, char* argv[]) {
  string hostfile = argv[4];
  int comm_size = atoi(argv[6]);
  int comm_rank = atoi(argv[8]);

  ptre_init(comm_size, comm_rank, hostfile.c_str(), 0, 1);

  PtreGlobal& ptre_global = PtreGlobalState();
  RdmaContext ctx(ptre_global.rdma_mgr);

  int num_tensors = atoi(argv[1]);
  size_t total_bytes = atol(argv[2]);

  assert(total_bytes % sizeof(float) == 0);
  int num_elems = total_bytes / sizeof(float);
  assert(num_elems >= num_tensors);

  vector<pair<float*, int>> tensors;
  wkim_init_tensors(num_elems, num_tensors, comm_rank, &tensors);
  vector<pair<float*, int>> recvs;
  for (int i = 0; i < tensors.size(); i++) {
    recvs.emplace_back(new float[tensors[i].second], tensors[i].second);
  }

  chrono::system_clock::time_point begin = chrono::system_clock::now();
  for (int i = 0; i < tensors.size(); i++) {
    RdmaAllreduce((void*) tensors[i].first, recvs[i].first, tensors[i].second,
        DataType::DT_FLOAT, ReduceOp::REDUCE_SUM, &ctx);
  }
  chrono::system_clock::time_point end = chrono::system_clock::now();
  chrono::milliseconds dur =
      chrono::duration_cast<chrono::milliseconds>(end - begin);

  if (!wkim_validate_allreduce(comm_size, recvs)) {
    LOG(ERROR) << "Validation Failed!";
  } else {
    LOG(INFO) << (int) dur.count() << " msec";
  }

  PtreBarrier();
  ptre_finalize(0);

  return 0;
}
