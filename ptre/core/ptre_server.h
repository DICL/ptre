#ifndef TENSORFLOW_PTRE_CORE_PTRE_SERVER_H_
#define TENSORFLOW_PTRE_CORE_PTRE_SERVER_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "ptre/rpc/ptre_service_impl.h"
#include "ptre/common/cm/remote_store.h"

namespace tensorflow {

class PtreServer {
 public:
  PtreServer(int rank);
  ~PtreServer();

  void Init();
  void Start();
  void Stop();
  void Join();

  bool CheckIncoming();
  void InitTrainableVariables(const std::vector<std::string>& names,
                              //const std::vector<Tensor*>& var_tensors,
                              const std::vector<Tensor*>& tvars,
                              const std::vector<Tensor*>& cvars,
                              //TF_Tensor* const* vars,
                              //const std::vector<DataType>& dtypes,
                              //const std::vector<TensorShape>& shapes,
                              int nvars);
  void LogDebugString(const std::string& name, int max_entries);
  Tensor* CmTensor(const std::string& name);
  //void CM_AverageVariable(const Tensor* other);
  //void NewPtreServer(int rank, std::unique_ptr<PtreServer>* out_server);
  const std::string target() const;

 private:
  void GrpcStart();
  RemoteStore remote_store_;
  RemoteStore trainer_store_;
  mutex mu_;
  enum State { DISCONNECTED, CONNECTED };
  PtreServiceImpl* ptre_service_ = nullptr;
  int rank_;
};

void NewPtreServer(int rank, std::unique_ptr<PtreServer>* out_server);

}

#endif  // TENSORFLOW_PTRE_CORE_PTRE_SERVER_H_
