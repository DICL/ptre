#ifndef PTRE_COMMUNICATION_PUSH_VARIABLE_H_
#define PTRE_COMMUNICATION_PUSH_VARIABLE_H_

#include <mutex>

namespace ptre {

class PushVariable {
 public:
  PushVariable(size_t length);

  void StartPush();
  void StopPush();
  int GetState();
  void* data();
  size_t length();

 private:
  std::mutex mu_;
  // Send Buffer
  void* buf_;
  size_t length_;
  // State
  int state_;
};

}  // namespace ptre

#endif  // PTRE_COMMUNICATION_PUSH_VARIABLE_H_
