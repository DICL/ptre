#include "ptre/common/operations.h"

using namespace std;
using namespace ptre::common;

int main(int argc, char* argv[]) {
  ptre_init(atoi(argv[argc - 3]), atoi(argv[argc - 1]), argv[argc - 5], 0, 1);

  int comm_size = ptre_size();
  int comm_rank = ptre_rank();
  PtreGlobal& ptre_global = PtreGlobalState();
  RdmaContext ctx(ptre_global.rdma_mgr);

  int ret;
  const size_t kSize = atol(argv[1]);
  const int kCount = kSize / sizeof(float);
  float* arr = (float*) malloc(kSize);
  RemoteAddr ra;

  for (int i = 0; i < kCount; i++) {
    arr[i] = 0.1 * (comm_rank + 1);
  }


  if (comm_rank == 0) {
    ctx.RegisterRecvBuffer((void*) arr, kSize);
    struct ibv_mr* mr = ctx.recv_mr((void*) arr);
    if (mr == NULL) {
      cout << "recv mr is NULL!\n";
      return 1;
    }
    ra.remote_addr = (uint64_t) mr->addr;
    ra.rkey = mr->rkey;
    cout << "Send remote_addr=" << ra.remote_addr << ", rkey=" << ra.rkey << endl;
    RdmaSend((void*) &ra, sizeof(RemoteAddr), DataType::DT_BOOL, 1, 0, &ctx);

    PtreBarrier();

    uint32_t imm_data;
    ret = RdmaRecvWithImm(NULL, &imm_data, 0, DataType::DT_BOOL, 1, 0,
        &ctx, NULL);
    cout << "Recv arr[0]=" << arr[0]
      << ", arr[kCount - 1]=" << arr[kCount - 1]
      << ", imm_data=" << imm_data << endl;
  } else {
#if 0
cout << "00\n";
    ctx->RegisterSendBuffer((void*) arr, kSize);
cout << "01\n";
#endif
    RdmaRecv((void*) &ra, sizeof(RemoteAddr), DataType::DT_BOOL, 0, 0, &ctx,
        NULL);
    cout << "Recv remote_addr=" << ra.remote_addr << ", rkey=" << ra.rkey << endl;

    PtreBarrier();

    uint32_t imm_data = 1234;
    ret = RdmaWriteWithImm((void*) arr, imm_data, ra, kCount,
        DataType::DT_FLOAT, 0, 0, &ctx);
    cout << "Write arr[0]=" << arr[0]
      << ", arr[kCount - 1]=" << arr[kCount - 1]
      << ", imm_data=" << imm_data << endl;
  }
  PtreBarrier();
  ptre_finalize(0);
  return 0;
}
