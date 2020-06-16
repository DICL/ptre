#ifndef PTRE_COMMON_MESSAGE_H_
#define PTRE_COMMON_MESSAGE_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/types.pb.h"

#define COMM_IN_PLACE ((void*) 1)

namespace ptre {
namespace common {

using std::string;
using ::tensorflow::DataType;

#if 0
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
#endif

std::size_t DataType_Size(DataType value);

class Request {
 public:
  enum RequestType {
    ALLREDUCE = 0
  };

  void set_request_rank(int comm_rank);

  void set_request_type(RequestType val);

  RequestType request_type() const { return request_type_; }

  void set_tensor_name(const string& name);

  const string& tensor_name() const { return tensor_name_; }

  void set_tensor_type(DataType dtype);

  DataType tensor_type() const { return tensor_type_; }

 private:
  int request_rank_;
  RequestType request_type_;
  DataType tensor_type_;
  string tensor_name_;
};

class Response {
 public:
  enum ResponseType {
    ALLREDUCE = 0
  };

  void set_response_type(ResponseType value);

  ResponseType response_type() const { return response_type_; }

  void set_tensor_type(DataType value);

  DataType tensor_type() const { return tensor_type_; }

  //const string tensor_names_string() const;

  //void set_tensor_names(const std::vector<string>& value);

  void add_tensor_name(const string& value);

  void add_tensor_name(string&& value);

  // Empty if the type is DONE or SHUTDOWN.
  const std::vector<string>& tensor_names() const { return tensor_names_; }

 private:
  ResponseType response_type_;
  DataType tensor_type_;
  std::vector<string> tensor_names_;
};


}  // namespace common
}  // namespace ptre


#endif  // PTRE_COMMON_MESSAGE_H_
