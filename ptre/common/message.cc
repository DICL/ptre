#include "ptre/common/message.h"

namespace ptre {
namespace common {

std::size_t DataType_Size(DataType value) {
  switch (value) {
    case PTRE_FLOAT32: {
      return sizeof(float);
    }
    default: {
      assert(0);
    }
  }
}


}  // namespace common
}  // namespace ptre
