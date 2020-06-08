#ifndef PTRE_COMMUNICATION_RDMA_PULL_JOB_H_
#define PTRE_COMMUNICATION_RDMA_PULL_JOB_H_

#include <map>
#include <string>
#include <vector>

#include "ptre/communication/rdma/rdma_task.h"

namespace ptre {

using std::string;

class PullJob {
 public:
  PullJob(int step, const std::map<int, std::vector<string>>& init_attr);
  void GetDstPeers(const string& var_name, std::vector<int>* out_dsts);
  void SetTask(int dst, const string& var_name, PullTask* task);
  void DeleteTask(int dst, const string& var_name);
  void DeleteTask(PullTask* task);
  void StopTasks(const string& var_name);
  int NumLiveTasks();

 private:
  int step_;
  int num_dsts_;
  std::map<int, std::map<string, PullTask*>> task_tables_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_RDMA_PULL_JOB_H_
