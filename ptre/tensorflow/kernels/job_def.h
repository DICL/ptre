#ifndef PTRE_TENSORFLOW_KERNELS_JOB_DEF_H_
#define PTRE_TENSORFLOW_KERNELS_JOB_DEF_H_

#include <queue>
#include <memory>
#include <string>

namespace ptre {

using std::string;

class PushRequest;
class PushJob;

class PushTask {
 public:
  PushTask(PushJob* job, int dst, const string& var_name);
  PushTask(PushJob* job, const string& var_name);
  // Member Access Functions
  void set_dst(int dst);
  int dst();
  const string& var_name();
  PushJob* job();

 private:
  int dst_;
  PushJob* job_;
  string var_name_;
};

class PushJob {
 public:
  PushJob(PushRequest* request, int dst, const std::vector<string>& var_names);
  std::queue<std::shared_ptr<PushTask>>& q();
  PushRequest* request();
  void set_dst(int dst);
  int dst();

 private:
  int dst_;
  PushRequest* request_;
  std::queue<std::shared_ptr<PushTask>> q_;
};

class PushRequest {
 public:
  PushRequest(int num_push, int step, int comm_size,
      const std::vector<string>& var_names);
  std::queue<std::shared_ptr<PushJob>>& q();
  bool checker(int dst);
  void check(int dst);
  int step();

 private:
  int num_push_;
  int step_;
  int comm_size_;
  std::queue<std::shared_ptr<PushJob>> q_;
  std::vector<bool> checker_;
};

class PushTaskV2 {
 public:
  PushTaskV2(int dst, const string& var_name);
  int dst();
  const string& var_name();
  void SetAttemptDone();
  bool IsAttemptDone();

 private:
  int dst_;
  string var_name_;
  bool attempt_done_;
};

}

#endif  // PTRE_TENSORFLOW_KERNELS_JOB_DEF_H_
