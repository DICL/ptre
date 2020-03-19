#ifndef PTRE_LIB_TYPES_H_
#define PTRE_LIB_TYPES_H_

#include <string>

namespace ptre {

using std::string;

struct NameBuffer {
  string name;  // Tensor name
  void* buf;
  size_t length;
};

}  // namespace ptre
#endif  // PTRE_LIB_TYPES_H_
