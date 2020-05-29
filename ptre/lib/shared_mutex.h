#ifndef PTRE_LIB_SHARED_MUTEX_H_
#define PTRE_LIB_SHARED_MUTEX_H_

#include <shared_mutex>

namespace ptre {

class SharedMutex {
 public:
  void lock();
  void unlock();
  void lock_shared();
  void unlock_shared();

 private:
  std::shared_mutex mu_;
};

}  // namespace ptre

#endif  // PTRE_LIB_SHARED_MUTEX_H_
