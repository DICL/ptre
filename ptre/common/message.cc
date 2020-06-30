#include "ptre/common/message.h"

#include <cassert>

namespace ptre {
namespace common {

std::size_t DataType_Size(DataType value) {
  switch (value) {
    case DataType::DT_FLOAT: {
      return sizeof(float);
    }
    case DataType::DT_INT32: {
      return sizeof(int);
    }
    case DataType::DT_BOOL: {
      return sizeof(bool);
    }
    default: {
      return 0;
    }
  }
}

/*
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
*/

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

bool Response::FromProto(ResponseProto&& proto) {
  set_response_type(proto.response_type());
  std::vector<string> tensor_names;
  for (auto& name : *proto.mutable_tensor_names()) {
    tensor_names.push_back(std::move(name));
  }
  tensor_names_ = std::move(tensor_names);
  set_tensor_type(DataType(proto.tensor_type()));
  std::vector<int> devices;
  for (auto& d : *proto.mutable_devices()) {
    devices.push_back(std::move(d));
  }
  devices_ = std::move(devices);
  std::vector<size_t> tensor_sizes;
  for (auto& size : *proto.mutable_tensor_sizes()) {
    tensor_sizes.push_back(std::move(size));
  }
  tensor_sizes_ = std::move(tensor_sizes);
  return true;
}

void Response::AsProto(ResponseProto* proto) {
  proto->Clear();
  proto->set_response_type(response_type_);
  proto->set_tensor_type(tensor_type_);
  for (auto& tensor_name : tensor_names_) {
    proto->add_tensor_names(tensor_name);
  }
  for (auto& d : devices_) {
    proto->add_devices(d);
  }
  for (auto& tensor_size : tensor_sizes_) {
    proto->add_tensor_sizes(tensor_size);
  }
}

void ResponseList::add_response(Response&& response) {
  responses_.push_back(std::move(response));
}

bool ResponseList::FromProto(ResponseListProto&& proto) {
  std::vector<Response> responses;
  for (auto& res_proto : *proto.mutable_responses()) {
    responses.emplace_back();
    responses.back().FromProto(std::move(res_proto));
  }
  responses_ = std::move(responses);
  return true;
}

void ResponseList::AsProto(ResponseListProto* proto) {
  proto->Clear();
  for (auto& res : responses_) {
    ResponseProto res_proto;
    res.AsProto(&res_proto);
    *proto->add_responses() = std::move(res_proto);
  }
}


}  // namespace common
}  // namespace ptre
