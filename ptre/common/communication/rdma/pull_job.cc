#include "ptre/common/communication/rdma/pull_job.h"

namespace ptre {
namespace common {

PullJob::PullJob(int step,
                 const std::map<int, std::vector<string>>& init_attr) {
  step_ = step;
  num_dsts_ = init_attr.size();
  for (auto it : init_attr) {
    int dst = it.first;
    for (auto&& var_name : it.second) {
      task_tables_[dst][var_name] = nullptr;
    }
  }
}

void PullJob::GetDstPeers(const string& var_name, std::vector<int>* out_dsts) {
  out_dsts->clear();
  for (auto it : task_tables_) {
    auto&& table = it.second;
    auto search = table.find(var_name);
    if (search != table.end()) {
      out_dsts->push_back(it.first);
    }
  }
}

void PullJob::SetTask(int dst, const string& var_name, PullTask* task) {
  std::lock_guard<std::mutex> guard(mu_);
  PullTask* old_task = task_tables_[dst][var_name];
  if (old_task != nullptr) {
    if (old_task->state() == PullTask::STATE_STOPPED) {
      task->SetState(PullTask::STATE_STOPPED);
    }
    task_tables_[dst][var_name] = task;
    delete old_task;
  } else {
    task_tables_[dst][var_name] = task;
  }
}

void PullJob::DeleteTask(int dst, const string& var_name) {
  std::lock_guard<std::mutex> guard(mu_);
  auto task = task_tables_[dst][var_name];
  if (task != nullptr) {
    delete task;
    task_tables_[dst][var_name] = nullptr;
  }
}

void PullJob::DeleteTask(PullTask* task) {
  std::lock_guard<std::mutex> guard(mu_);
  for (auto& it : task_tables_) {
    for (auto& task_it : it.second) {
      if (task_it.second == task) {
        delete task;
        task_it.second = nullptr;
      }
    }
  }
}

void PullJob::StopTasks(const string& var_name) {
  std::lock_guard<std::mutex> guard(mu_);
  for (auto it : task_tables_) {
    auto task = it.second[var_name];
    if (task != nullptr) {
      task->SetState(PullTask::STATE_STOPPED);
    }
  }
}

int PullJob::NumLiveTasks() {
  std::lock_guard<std::mutex> guard(mu_);
  int live_cnt = 0;
  for (auto it : task_tables_) {
    for (auto task_it : it.second) {
      if (task_it.second != nullptr) {
        live_cnt++;
      }
    }
  }
  return live_cnt;
}

}  // namespace common
}  // namespace ptre
