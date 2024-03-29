#ifndef PTRE_COMMON_OPERATIONS_H_
#define PTRE_COMMON_OPERATIONS_H_

#include <string>

#include "ptre/common/cm/ready_tensor.h"
#include "ptre/common/common.h"
//#include "ptre/common/communication/rdma/rdma_task.h"
#include "ptre/common/message.h"
#include "ptre/common/ptre_global.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/rdma/rdma_mpi.h"

#define NUM_POLLING_THREADS 1
#define NUM_AGG_THREADS 1
#define NUM_AGG_EIGEN_THREADS 16
#define AGG_EIGEN_POOLSIZE 16
#define NUM_REDUCE_EIGEN_THREADS 32
//#define NUM_PUSH_THREADS 4
//#define NUM_RECV_THREADS 1

namespace ptre {
namespace common {

using ::std::string;

void load_grpc_hosts(const string& grpc_hosts_file);
void InitComm(int size, int rank, const string& grpc_hosts_file);

void ShutdownGrpcServer();

PtreGlobal& PtreGlobalState();

void PtreSend(int dst_rank, char* buf, size_t len, const string& name);

void PtreRecv(int src_rank, char* buf, size_t len, const string& name);

void PtreBroadcast(char* buf, size_t len, int root_rank, const string& name);

void PtreBarrier();

#if 1
void PtreFlushSimpleHtod();
#endif

void CreatePullJob(int step, int num_pull);

void ClearPullJobs();

void EnqueuePullTasks(const string& var_name, int num_pull);
void StopPullTasks(const string& var_name);

//void EnqueueAggregation(PullTask* task);
//void ProcessPullTaskCQ(PullTask* task);
int ProcessCQ(int dst, struct ibv_wc* wcs);
#if 0
void PollingThreadLoop(int tid);
#else
void PollingThreadLoop();
#endif
void ConcurrentAggregationThreadLoop();

void RdmaSetRemoteAddress(int dst, BufType buf_type, const string& var_name);
void RegisterTrainableVariables(OpContext* context,
                                const std::vector<string>& names_);

extern "C" {

int ptre_init(int size, int rank, const char* grpc_hosts_file,
              int selection_strategy, int num_push);

void ptre_finalize(unsigned int wait_time);

int ptre_size();

int ptre_rank();

void ptre_set_local_step(int local_step);

void ptre_create_pull_job();

void ptre_barrier();

void ptre_print_counter_summary_epoch();

void ptre_print_counter_summary();

void ptre_call_generic(const char* func_name);

}

// --------------------------------------------------------------------------

ReadyTensor* GetReadyTensor(const string& name);

// --------------------------------------------------------------------------

//Status EnqueueGetRemoteVariable(OpContext* ctx, const string& var_name,
//                                Tensor* output, Tensor* num_agg,
//                                StatusCallback callback);

Status EnqueueTensorAsyncComm(OpContext* context,
                              const string var_name,
                              std::shared_ptr<Tensor> tensor,
                              StatusCallback callback,
                              CommOp comm_op);

Status EnqueueTensorAwaitComm(OpContext* context,
                              const string var_name,
                              std::shared_ptr<Tensor> tensor,
                              StatusCallback callback);

Status EnqueueTensorModelaverage(OpContext* ctx, Tensor& tensor, Tensor& output,
                                 const string& node_name,
                                 StatusCallback callback,
                                 ModelaverageOp modelaverage_op);

Status EnqueueTensorPull(const string name);

Status EnqueueTensorAllreduce(OpContext* ctx, Tensor& tensor, Tensor& output,
                              const string node_name, StatusCallback callback,
                              ReduceOp reduce_op);

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_OPERATIONS_H_
