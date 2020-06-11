#ifndef PTRE_COMMON_OPERATIONS_H_
#define PTRE_COMMON_OPERATIONS_H_

#include <string>

#include "ptre/common/common.h"
#include "ptre/common/communication/rdma/rdma_task.h"

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

void RunGrpcServer();

void ShutdownGrpcServer();

void PtreSend(int dst_rank, char* buf, size_t len, const string& name);

void PtreRecv(int src_rank, char* buf, size_t len, const string& name);

void PtreBroadcast(char* buf, size_t len, int root_rank, const string& name);

void PtreBarrier();

void CreatePullJob(int step, int num_pull);

void ClearPullJobs();

void EnqueuePullTasks(const string& var_name, int num_pull);
void StopPullTasks(const string& var_name);

void EnqueueAggregation(PullTask* task);
void ProcessPullTaskCQ(PullTask* task);
int ProcessCQ(int dst, struct ibv_wc* wcs);
void PollingThreadLoop(int tid);
void ConcurrentAggregationThreadLoop();

void RdmaSetRemoteAddress(int dst, BufType buf_type, const string& var_name);
void RegisterTrainableVariables(OpContext* context,
                                const std::vector<string>& names_);

extern "C" {

int ptre_init(int size, int rank, char* grpc_hosts_file, int selection_strategy,
              int num_push);

void ptre_finalize(unsigned int wait_time);

int ptre_size();

int ptre_rank();

void ptre_set_local_step(int local_step);

void ptre_create_pull_job();

void ptre_barrier();

void ptre_print_counter_summary_epoch();

void ptre_print_counter_summary();

}

Status EnqueueGetRemoteVariable(OpContext* ctx, const string& var_name,
                                Tensor* output, Tensor* num_agg,
                                StatusCallback callback);

}  // namespace common
}  // namespace ptre

#endif  // PTRE_COMMON_OPERATIONS_H_
