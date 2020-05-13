#include "ptre/kernels/job_def.h"

namespace ptre {

PushTask::PushTask(int dst, const string& var_name) {
  dst_ = dst;
  var_name_ = var_name;
}

PushTask::PushTask(const string& var_name) : PushTask(-1, var_name) { }

void PushTask::set_dst(int dst) {
  dst_ = dst;
}

const int& PushTask::dst() {
  return dst_;
}

const string& PushTask::var_name() {
  return var_name_;
}

PushJob::PushJob(PushRequest* request, int dst) {
  dst_ = dst;
  request_ = request;
}

std::queue<std::shared_ptr<PushTask>>& PushJob::q() {
  return q_;
}

PushRequest::PushRequest(int num_push, int step, int comm_size) {
  num_push_ = num_push;
  step_ = step;
  comm_size_ = comm_size;
  checker_.resize(comm_size_);
  q_.push
}

std::queue<std::shared_ptr<PushJob>>& PushRequest::q() {
  return q_;
}

const bool& PushRequest::checker(int dst) {
  return checker_[dst];
}

void PushRequest::check(int dst) {
  checker_[dst] = true;
}

}  // namespace ptre
