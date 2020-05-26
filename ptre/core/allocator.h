#ifndef PTRE_CORE_ALLOCATOR_H_
#define PTRE_CORE_ALLOCATOR_H_

#include <mutex>
#include <vector>

namespace ptre {

class Allocator {
 public:
  Allocator(size_t n);
  Allocator(const std::vector<size_t>& sizes);
  void* Allocate(size_t n);

 private:
  std::mutex mu_;
  void* base_buf_;
  size_t head_;
  size_t capacity_;
};

}  // namespace ptre

#endif  // PTRE_CORE_ALLOCATOR_H_
