#include "ptre/core/ptre_server.h"

#include <string>

#include "tensorflow/ptre/rpc/ptre_service_impl.h"

namespace tensorflow {

PtreServer::PtreServer(int rank) : rank_(rank) {}

PtreServer::~PtreServer() {
  delete ptre_service_;
}

void PtreServer::Init() {
}
void PtreServer::Start() {
}
void PtreServer::Stop() {
}
void PtreServer::Join() {
}

bool PtreServer::CheckIncoming() {
  //TODO: imp.
  //auto ret = cm_->CheckIncoming();
  //return ret;
  return true;
}

void PtreServer::InitTrainableVariables(const std::vector<std::string>& names,
                                        const std::vector<Tensor*>& tvars,
                                        const std::vector<Tensor*>& cvars,
                                        int nvars) {
  LOG(INFO) << "PtreServer got " << nvars << " tensors:" << std::endl;
  LOG(INFO) << "Adding tensors to remote store" << std::endl;
  for (int i = 0; i < nvars; i++) {
    remote_store_.AddVariable(names[i], cvars[i]);
    trainer_store_.AddVariable(names[i], tvars[i]);
  }
}

Tensor* PtreServer::CmTensor(const std::string& name) {
  return remote_store_.tensor(name);
}

void PtreServer::LogDebugString(const std::string& name, int max_entries) {
  LOG(INFO) << "tvar: " << name << std::endl
            << trainer_store_.DebugString(name, max_entries) << std::endl;
  LOG(INFO) << "cvar: " << name << std::endl
            << remote_store_.DebugString(name, max_entries) << std::endl;
}

/*
void Tensor::CopyFromInternal(const Tensor& other, const TensorShape& shape) {
  CHECK_EQ(shape.num_elements(), other.NumElements());
  // Data type will be overwritten if this == &other, since dtype is part of
  // shape.
  DataType other_dtype = other.dtype();
  shape_ = shape;
  set_dtype(other_dtype);
  if (buf_ != other.buf_) {
    UnrefIfNonNull(buf_);
    buf_ = other.buf_;
    RefIfNonNull(buf_);
  }
}
*/
/*
Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape)
    : shape_(shape), buf_(nullptr) {
  set_dtype(type);
  CHECK_NOTNULL(a);
  if (shape_.num_elements() > 0 || a->AllocatesOpaqueHandle()) {
    CASES(type, buf_ = new Buffer<T>(a, shape.num_elements()));
  }
  if (buf_ != nullptr && buf_->data() != nullptr && LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation("Unknown", LogMemory::UNKNOWN_STEP_ID,
                                      *this);
  }
}
*/

const std::string PtreServer::target() const {
  return "grpc://localhost:50051";
}

void PtreServer::GrpcStart() {
  std::string server_address("0.0.0.0:50051");

}

void NewPtreServer(int rank, std::unique_ptr<PtreServer>* out_server) {
  std::unique_ptr<PtreServer> ret(new PtreServer(rank));
  *out_server = std::move(ret);
}

}  // namespace tensorflow
