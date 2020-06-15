#ifndef PTRE_COMMON_MESSAGE_H_
#define PTRE_COMMON_MESSAGE_H_

namespace ptre {
namespace common {

class Request {
};

enum DataType {
  PTRE_UINT8 = 0,
  PTRE_INT8 = 1,
  PTRE_UINT16 = 2,
  PTRE_INT16 = 3,
  PTRE_INT32 = 4,
  PTRE_INT64 = 5,
  PTRE_FLOAT16 = 6,
  PTRE_FLOAT32 = 7,
  PTRE_FLOAT64 = 8,
  PTRE_BOOL = 9,
};

std::size_t DataType_Size(DataType value);


}  // namespace common
}  // namespace ptre


#endif  // PTRE_COMMON_MESSAGE_H_
