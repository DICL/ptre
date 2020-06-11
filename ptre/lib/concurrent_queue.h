#ifndef PTRE_LIB_CONCURRENT_QUEUE_H_
#define PTRE_LIB_CONCURRENT_QUEUE_H_

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <queue>
#include <deque>

namespace ptre {

template <typename T>
class ConcurrentQueue {
 public:
  void push(const T& value);
  void push(T&& value);
  void pop();
  void wait_and_pop(T& p);
  bool wait_for_and_pop(int sec, T& p);

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

template <typename T>
bool ConcurrentQueue<T>::wait_for_and_pop(int sec, T& p) {
  std::unique_lock<std::mutex> lk(mu_);
  bool ret;
  if (cv_.wait_for(lk, std::chrono::seconds(sec),
                   [&] { return !q_.empty(); })) {
    //p = std::move(q_.front());
    p = q_.front();
    q_.pop();
    ret = true;
  } else {
    ret = false;
  }
  lk.unlock();
  return ret;
}

template <typename T>
class ConcurrentUniqueQueue {
 public:
  void push(const T& value);
  void push(T&& value);
  void pop();
  void wait_and_pop(T& p);

 protected:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> dq_;
};

template <typename T>
void ConcurrentUniqueQueue<T>::push(const T& value) {
  mu_.lock();
  const bool was_empty = dq_.empty();
  auto search = std::find(dq_.begin(), dq_.end(), value);
  if (search == dq_.end()) {
    dq_.push_back(value);
  }
  mu_.unlock();
  if (was_empty) {
    cv_.notify_one();
  }
}

template <typename T>
void ConcurrentUniqueQueue<T>::push(T&& value) {
  mu_.lock();
  const bool was_empty = dq_.empty();
  auto search = std::find(dq_.begin(), dq_.end(), value);
  if (search == dq_.end()) {
    dq_.push_back(std::move(value));
  }
  mu_.unlock();
  if (was_empty) {
    cv_.notify_one();
  }
}

template <typename T>
void ConcurrentUniqueQueue<T>::pop() {
  std::lock_guard<std::mutex> lk(mu_);
  if (!dq_.empty()) {
    dq_.pop_front();
  }
}

template <typename T>
void ConcurrentUniqueQueue<T>::wait_and_pop(T& p) {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return !dq_.empty(); });
  //p = std::move(dq_.front());
  p = dq_.front();
  dq_.pop_front();
  lk.unlock();
}

}  // namespace ptre

#endif  // PTRE_LIB_CONCURRENT_QUEUE_H_
