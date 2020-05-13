#ifndef PTRE_KERNELS_JOB_DEF_H_
#define PTRE_KERNELS_JOB_DEF_H_

#include <queue>
#include <string>

namespace ptre {

using std::string;

class PushTask {
 public:
  PushTask(int dst, const string& var_name);
  PushTask(const string& var_name);
  // Member Access Functions
  void set_dst(int dst);
  const int& dst();
  const string& var_name();

 private:
  int dst_;
  string var_name_;
};

class PushJob {
 private:
  int dst;
  std::weak_ptr<PushRequest> request_;
  std::queue<std::shared_ptr<PushTask>> q_;
};

class PushRequest {
 public:
  PushRequest(int num_push, int step, int comm_size);
  std::queue<std::shared_ptr<PushJob>>& q();
  const bool& checker(int dst);
  void check(int dst);

 private:
  int num_push_;
  int step_;
  int comm_size_;
  std::queue<std::shared_ptr<PushJob>> q_;
  std::vector<bool> checker_;
};

}

#endif  // PTRE_KERNELS_JOB_DEF_H_
