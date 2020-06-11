#include "ptre/tensorflow/kernels/job_def.h"

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

PushJob::PushJob(PushRequest* request, int dst,
    const std::vector<string>& var_names) {
  dst_ = dst;
  request_ = request;
  for (auto&& name : var_names) {
    auto task = std::make_shared<PushTask>(this, -1, name);
    q_.push(task);
  }
}

std::queue<std::shared_ptr<PushTask>>& PushJob::q() {
  return q_;
}

PushRequest* PushJob::request() {
  return request_;
}

PushRequest::PushRequest(int num_push, int step, int comm_size,
    const std::vector<string>& var_names) {
  num_push_ = num_push;
  step_ = step;
  comm_size_ = comm_size;
  checker_.resize(comm_size_);
  for (int i = 0; i < num_push; i++) {
    std::shared_ptr<PushJob> job(new PushJob(this, -1, var_names));
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

PushTaskV2::PushTaskV2(int dst, const string& var_name) {
  dst_ = dst;
  var_name_ = var_name;
  attempt_done_ = false;
}

int PushTaskV2::dst() {
  return dst_;
}

const string& PushTaskV2::var_name() {
  return var_name_;
}

void PushTaskV2::SetAttemptDone() {
  attempt_done_ = true;
}

bool PushTaskV2::IsAttemptDone() {
  return attempt_done_;
}

}
