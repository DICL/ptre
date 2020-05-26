#include "ptre/core/allocator.h"

#include "ptre/lib/cache_ctl.h"

#include "tensorflow/core/platform/logging.h"

namespace ptre {

inline size_t padded_size(size_t n) {
  size_t ret = n;
  size_t r = n % CACHE_LINE_SIZE;
  if (r != 0) {
    ret = ret - r + CACHE_LINE_SIZE;
  }
  return ret;
}

Allocator::Allocator(size_t n) {
  capacity_ = padded_size(n);
  base_buf_ = aligned_alloc(CACHE_LINE_SIZE, capacity_);
  head_ = 0;
}

Allocator::Allocator(const std::vector<size_t>& sizes) {
  size_t required = 0;
  for (int i = 0; i < sizes.size(); i++) {
    required += padded_size(sizes[i]);
  }
  capacity_ = padded_size(required);
  base_buf_ = aligned_alloc(CACHE_LINE_SIZE, capacity_);
  head_ = 0;
}

void* Allocator::Allocate(size_t n) {
  std::lock_guard<std::mutex> guard(mu_);
  size_t alloc_size = padded_size(n);
  if (head_ + alloc_size <= capacity_) {
    void* ret = (void*) ((size_t) base_buf_ + head_);
    head_ += alloc_size;
    return ret;
  } else {
    LOG(ERROR) << "Failed to allocate size=" << n << ", alloc_size="
        << alloc_size<< ", available=" << capacity_ - head_;
    return NULL;
  }
}


}  // namespace ptre
