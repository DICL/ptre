#include "ptre/common/message.h"

#include <cassert>

namespace ptre {
namespace common {

std::size_t DataType_Size(DataType value) {
  switch (value) {
    case DataType::DT_FLOAT: {
      return sizeof(float);
    }
    default: {
      assert(0);
    }
  }
}

void Request::set_request_rank(int comm_rank) {
  request_rank_ = comm_rank;
}

void Request::set_request_type(RequestType val) {
  request_type_ = val;
}

void Request::set_tensor_name(const string& name) {
  tensor_name_ = name;
}

void Request::set_tensor_type(DataType dtype) {
  tensor_type_ = dtype;
}

void Response::set_response_type(ResponseType value) {
  response_type_ = value;
}

void Response::set_tensor_type(DataType value) {
  tensor_type_ = value;
}

void Response::add_tensor_name(const string& value) {
  tensor_names_.push_back(value);
}

void Response::add_tensor_name(string&& value) {
  tensor_names_.push_back(std::move(value));
}

}  // namespace common
}  // namespace ptre
