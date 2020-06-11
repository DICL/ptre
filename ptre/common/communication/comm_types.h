#ifndef PTRE_COMMON_COMMUNICATION_COMM_TYPES_H_
#define PTRE_COMMON_COMMUNICATION_COMM_TYPES_H_

namespace ptre {

struct PullKey {
  bool curr;
  uint64_t keys[2];
};

}  // namespace ptre

#endif  // PTRE_COMMON_COMMUNICATION_COMM_TYPES_H_
