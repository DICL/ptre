#include "ptre/common/buffer_table.h"

#include "ptre/common/logging.h"

namespace ptre {
namespace common {

//std::size_t BufKeyHash::operator()(const BufKey& key) const {
//  return std::hash<BufType>()(key.first) ^ std::hash<string>()(key.second);
//}

void* BufferTable::Get(const BufType type, const string& name) {
  std::lock_guard<std::mutex> guard(mu_);
  auto iter = table_.find(type);
  if (iter == table_.end()) {
    return NULL;
  }
  auto& inner = iter->second;
  auto iter2 = inner.find(name);
  if (iter2 == inner.end()) {
    return NULL;
  }
  return iter2->second.first;
}

size_t BufferTable::GetSize(const BufType type, const string& name) {
  std::lock_guard<std::mutex> guard(mu_);
  return table_[type][name].second;
}

void* BufferTable::Set(const BufType type, const string& name, void* buf,
                       const size_t size) {
  std::lock_guard<std::mutex> guard(mu_);
  table_[type][name] = PtrAndSize { buf, size };
  return buf;
}

void* BufferTable::GetOrAllocate(const BufType type, const string& name) {
  std::lock_guard<std::mutex> guard(mu_);
  auto it = table_.find(type);
  if (it != table_.end()) {
    auto it2 = it->second.find(name);
    if (it2 != it->second.end()) {
      return it2->second.first;
    }
  }
  if (type == BUF_TYPE_RECVBUF_STATE_WRITE) {
    void* new_buf;
    try {
    new_buf = (void*) new int(0);
    } catch (const std::bad_alloc& e) {
      LOG(ERROR) << "Allocation failed: " << e.what();
      exit(1);
    }
    table_[type][name] = PtrAndSize { new_buf, sizeof(int) };
    return new_buf;
  }
  return NULL;
}

void* BufferTable::WaitAndGet(const BufType type, const string& name) {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] {
        auto it1 = table_.find(type);
        if (it1 != table_.end()) {
          auto it2 = it1->second.find(name);
          return it2 != it1->second.end();
        }
        return false;
      });
  void* ret = table_[type][name].first;
  lk.unlock();
  return ret;
}

}
}
