#include "ptre/kernels/job_def.h"

namespace ptre {

PushTask::PushTask(PushJob* job, int dst, const string& var_name) {
  job_ = job;
  dst_ = dst;
  var_name_ = var_name;
}

PushTask::PushTask(PushJob* job, const string& var_name)
  : PushTask(job, -1, var_name) { }

void PushTask::set_dst(int dst) {
  dst_ = dst;
}

int PushTask::dst() {
  return dst_;
}

const string& PushTask::var_name() {
  return var_name_;
}

PushJob* PushTask::job() {
  return job_;
}

void PushJob::set_dst(int dst) {
  dst_ = dst;
}

int PushJob::dst() {
  return dst_;
}

PushJob::PushJob(PushRequest* request, int dst) {
  dst_ = dst;
  request_ = request;
}

std::queue<std::shared_ptr<PushTask>>& PushJob::q() {
  return q_;
}

PushRequest* PushJob::request() {
  return request_;
}

PushRequest::PushRequest(int num_push, int step, int comm_size) {
  num_push_ = num_push;
  step_ = step;
  comm_size_ = comm_size;
  checker_.resize(comm_size_);
  for (int i = 0; i < num_push; i++) {
    std::shared_ptr<PushJob> job(new PushJob(this, -1));
    q_.push(job);
  }
}

std::queue<std::shared_ptr<PushJob>>& PushRequest::q() {
  return q_;
}

bool PushRequest::checker(int dst) {
  return checker_[dst];
}

void PushRequest::check(int dst) {
  checker_[dst] = true;
}

int PushRequest::step() {
  return step_;
}

}  // namespace ptre
