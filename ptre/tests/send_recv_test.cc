#include "ptre/common/operations.h"
#include "ptre/common/rdma/rdma_mpi.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/ptre_global.h"

#include <string>
#include <unistd.h>

using std::string;
using namespace ptre::common;

const int kNumElements = 32;

int main(int argc, char* argv[]) {
  // TODO: Use a command line argument parsing library.
  string hostfile = argv[2];
  int comm_size = atoi(argv[4]);
  int comm_rank = atoi(argv[6]);
  ptre_init(comm_size, comm_rank, hostfile.c_str(), 0, 1);

  float arr[1024] = { };
  float arr2[1024] = { };
  float arr3[1024] = { };

  // Init arr
  if (comm_rank == 0) {
    for (int i = 0; i < kNumElements; i++) {
      arr[i] = comm_rank * 0.1;
      arr2[i] = (comm_rank + 1) * 0.1;
      arr3[i] = (comm_rank + 2) * 0.1;
    }
  }

  PtreGlobal& ptre_global = PtreGlobalState();

  LOG(INFO) << ptre_global.size << ", rank=" << ptre_global.rank;

  RdmaContext ctx(ptre_global.rdma_mgr);
  if (comm_rank == 0) {
    usleep(3 * 1000 * 1000);
    LOG(INFO) << "Post Send";
    RdmaSend((void*) arr, kNumElements, DataType::DT_FLOAT, 1, 0, &ctx);
    LOG(INFO) << "Send Done";
    LOG(INFO) << "Post Send";
    RdmaSend((void*) arr2, kNumElements, DataType::DT_FLOAT, 1, 0, &ctx);
    LOG(INFO) << "Send Done";
    LOG(INFO) << "Post Send";
    RdmaSend((void*) arr3, kNumElements, DataType::DT_FLOAT, 1, 0, &ctx);
    LOG(INFO) << "Send Done";
  } else {
    //RdmaRequest requests[2];
    RdmaRequest* requests[3];
    requests[0] = new RdmaRequest();
    requests[1] = new RdmaRequest();
    requests[2] = new RdmaRequest();
    LOG(INFO) << "Post Irecv";
    //RdmaRecv((void*) arr, kNumElements, DataType::DT_FLOAT, 0, 0, &ctx, NULL);
    RdmaIrecv((void*) arr, kNumElements, DataType::DT_FLOAT, 0, 0, &ctx,
        requests[0]);
    LOG(INFO) << "Post Irecv Done";
    LOG(INFO) << "Post Irecv";
    RdmaIrecv((void*) arr2, kNumElements, DataType::DT_FLOAT, 0, 0, &ctx,
        requests[1]);
    LOG(INFO) << "Post Irecv Done";
    RdmaWait(requests[0], NULL);
    LOG(INFO) << "Recv Done arr[0]=" << arr[0];
    LOG(INFO) << "Post Irecv";
    RdmaIrecv((void*) arr3, kNumElements, DataType::DT_FLOAT, 0, 0, &ctx,
        requests[2]);
    LOG(INFO) << "Post Irecv Done";
    RdmaWait(requests[1], NULL);
    LOG(INFO) << "Recv Done arr2[0]=" << arr2[0];
    RdmaWait(requests[2], NULL);
    LOG(INFO) << "Recv Done arr3[0]=" << arr3[0];
  }

  float sum = 0;
  float sum2 = 0;
  float sum3 = 0;
  for (int i = 0; i < kNumElements; i++) {
    sum += arr[i];
    sum2 += arr2[i];
    sum3 += arr3[i];
  }
  LOG(INFO) << "sum=" << sum;
  LOG(INFO) << "sum2=" << sum2;
  LOG(INFO) << "sum3=" << sum3;
  return 0;
}
