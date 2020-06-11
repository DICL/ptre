#include "ptre/common/communication/pull_variable.h"
#include "ptre/lib/cache_ctl.h"

namespace ptre {

PullVariable::PullVariable(const Tensor& var, const string& name,
                           Allocator* a) {
  key_ = (struct PullKey*) a->Allocate(sizeof(struct PullKey));
  memset(key_, 0, sizeof(struct PullKey));

  length_ = var.AllocatedBytes();
  for (int i = 0; i < 2; i++) {
    data_ptrs_[i] = a->Allocate(length_);
  }
  name_ = name;
}

void PullVariable::Switch() {
  key_->curr = !key_->curr;
  cache_ctl::clflush((char*) key_, sizeof(struct PullKey));
}

void PullVariable::SetNextKey(uint64_t key) {
  key_->keys[!key_->curr] = key;
  cache_ctl::clflush((char*) key_, sizeof(struct PullKey));
}

}  // namespace ptre
