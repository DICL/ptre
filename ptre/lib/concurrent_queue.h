#ifndef PTRE_LIB_CONCURRENT_QUEUE_H_
#define PTRE_LIB_CONCURRENT_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>

namespace ptre {

template <typename T>
class ConcurrentQueue {
 public:
  void push(const T& value);
  void push(T&& value);
  void pop();
  void wait_and_pop(T& p);

 protected:
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<T> q_;
};

template <typename T>
void ConcurrentQueue<T>::push(const T& value) {
  mu_.lock();
  const bool was_empty = q_.empty();
  q_.push(value);
  mu_.unlock();
  if (was_empty) {
    cv_.notify_one();
  }
}

template <typename T>
void ConcurrentQueue<T>::push(T&& value) {
  mu_.lock();
  const bool was_empty = q_.empty();
  q_.push(std::move(value));
  mu_.unlock();
  if (was_empty) {
    cv_.notify_one();
  }
}

template <typename T>
void ConcurrentQueue<T>::pop() {
  std::lock_guard<std::mutex> lk(mu_);
  if (!q_.empty()) {
    q_.pop();
  }
}

template <typename T>
void ConcurrentQueue<T>::wait_and_pop(T& p) {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return !q_.empty(); });
  //p = std::move(q_.front());
  p = q_.front();
  q_.pop();
  lk.unlock();
}

}  // namespace ptre

#endif  // PTRE_LIB_CONCURRENT_QUEUE_H_
