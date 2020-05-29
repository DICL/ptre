#include "ptre/lib/shared_mutex.h"

namespace ptre {

void SharedMutex::lock() {
  mu_.lock();
}

void SharedMutex::unlock() {
  mu_.unlock();
}

void SharedMutex::lock_shared() {
  mu_.lock_shared();
}

void SharedMutex::unlock_shared() {
  mu_.unlock_shared();
}

}  // namespace ptre
