#ifndef PTRE_COMMON_BUFFER_TABLE_H_
#define PTRE_COMMON_BUFFER_TABLE_H_

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ptre/protobuf/rdma_service.pb.h"

namespace ptre {
namespace common {

using std::string;

//using BufKey = std::pair<BufType, string>;
//struct BufKeyHash {
//  std::size_t operator()(const BufKey& key) const;
//};

class BufferTable {
 public:
  void* Get(const BufType type, const string& name);

  // Must be called after it is ensured the requesting buf exists.
  size_t GetSize(const BufType type, const string& name);

  void* Set(const BufType type, const string& name, void* buf,
            const size_t size);

  void* GetOrAllocate(const BufType type, const string& name);

  void* WaitAndGet(const BufType type, const string& name);

 private:
  std::mutex mu_;
  std::condition_variable cv_;

  using PtrAndSize = std::pair<void*, size_t>;
  std::unordered_map<BufType, std::unordered_map<string, PtrAndSize>> table_;
};

}
}
#endif  // PTRE_COMMON_BUFFER_TABLE_H_
