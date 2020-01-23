//#include "ptre/core/ptre_global.h"

#define EIGEN_USE_THREADS

#include <unistd.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>

#include "ptre/kernels/ptre_ops.h"
//#include "ptre/kernels/ptre_op_helpers.h"
//#include "ptre/ptre_global.h"
#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/rdma/rdma_manager.h"
//#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

using std::string;
using ::ptre::ConsensusManager;
using ::ptre::RdmaManager;
using ::ptre::RdmaServiceImpl;
using ::ptre::GrpcClient;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

struct PtreGlobal {
  PtreGlobal() {
    std::cout << (void*) this << std::endl;
  }
  ConsensusManager cm;
  RdmaManager* rdma_manager;
  //std::vector<Tensor*> remote_tensors;
  std::mutex mu;
  //std::queue<PtreRequest> request_queue;
  std::queue<int> q;

  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::thread background_thread;

  //bool new_incoming;

  int rank;
  int size;

  std::vector<std::string> grpc_hosts;

  bool is_push_step = false;
  /// 0: None
  /// 1: New
  /// 2: Used
  int incoming_state = 0;

  ~PtreGlobal() {
    if (background_thread.joinable()) {
      //shut_down = true;
      background_thread.join();
    }
    if (grpc_server_thread.joinable()) {
      grpc_server_thread.join();
    }
  }
};

static PtreGlobal ptre_global;

void RunGrpcServer() {
  RdmaServiceImpl service;
  service.SetRdmaManager(ptre_global.rdma_manager);
  std::string server_address("0.0.0.0:50051");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  std::cout << "Grpc server listening on " << server_address << std::endl;
  server->Wait();
}

void InitRemoteMR() {
}

void ProcessRequestQueue() {
  //std::lock_guard<std::mutex> l(ptre_global.mu);
  auto& q = ptre_global.q;
  if (!ptre_global.q.empty()) {
    int dst_rank = q.front();
    q.pop();
    //std::cout << std::endl << "PushTensors to target=" << dst_rank << std::endl;
    ptre_global.cm.PushTensors(dst_rank);
  }
}

void BackgroundThreadLoop() {
  /// TODO: 1. Init MR
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    /// TODO: Fetch a push task from a task queue.
    ProcessRequestQueue();
  }
}


void InitPtreOnce() {
  //while(true) {
  //  if (ptre_global.rdma_manager->IsMRInitialized()) {
  //    break;
  //  }
  //}

}

void load_grpc_hosts(const string& grpc_hosts_file) {
  std::string in_line;
  //std::ifstream in("/home/wkim/experiments/grpc_hosts");
  std::ifstream in(grpc_hosts_file);
  while (std::getline(in, in_line)) {
    if (in_line[0] == '#') continue;
    ptre_global.grpc_hosts.emplace_back(in_line);
  }
  in.close();
  for (int i = 0; i < ptre_global.size; i++) {
    std::cout << ptre_global.grpc_hosts[i] << std::endl;
  }
}

void InitComm(int size, int rank, const string& grpc_hosts_file) {
  /// First Initialization step
  ptre_global.size = size;
  ptre_global.rank = rank;
  ptre_global.cm.set_size(size);
  ptre_global.cm.set_rank(rank);
  // InitPeerSelector

  load_grpc_hosts(grpc_hosts_file);

  std::cout << "Initializing RdmaManager" << std::endl;
  ptre_global.rdma_manager = new RdmaManager(size, rank);
  ptre_global.cm.SetRdmaManager(ptre_global.rdma_manager);
  ptre_global.background_thread = std::thread(BackgroundThreadLoop);
}

void InitGrpcService() {
  std::cout << "Running grpc server" << std::endl;
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);
}

}  // namespace


//namespace functor {
//template <typename T>
//struct ApplyModelAveraging<CPUDevice, T> {
//  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
//                  typename TTypes<T>::ConstFlat remote) {
//    var.device(d) = 0.5 * (var + remote);
//  }
//};
//}  // namespace functor

REGISTER_OP("InitComm")
    .Attr("size: int")
    .Attr("rank: int")
    .Attr("grpc_hosts_file: string = '/home/wkim/experiments/grpc_hosts'");
class InitCommOp : public OpKernel {
 public:
  explicit InitCommOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr<int>("size", &size_);
    context->GetAttr<int>("rank", &rank_);
    context->GetAttr<string>("grpc_hosts_file", &grpc_hosts_file_);
  }
  void Compute(OpKernelContext* context) override {
    InitComm(size_, rank_, grpc_hosts_file_);
  }
 private:
  int size_;
  int rank_;
  string grpc_hosts_file_;
};
REGISTER_KERNEL_BUILDER(Name("InitComm").Device(DEVICE_CPU),
                        InitCommOp);

REGISTER_OP("InitGlobalConsensus")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class InitGlobalConsensusOp : public OpKernel {
 public:
  explicit InitGlobalConsensusOp(OpKernelConstruction* context)
      : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.cm.InitBufTensor(names_[i], context->input(i));
    }
    ptre_global.cm.InitBufParam();
    //std::vector<const Tensor*> inputs;
    //for (int i = 0; i < num_inputs; i++) {
    //  const Tensor &input = context->input(i);
    //  inputs.push_back(&input);
    //}
    //ptre_global.cm.InitGlobalConsensus(inputs);
    //auto recv_tensors = ptre_global.cm.GetGlobalConsensusList();
    //auto send_tensors = ptre_global.cm.GetSendTensorsList();
    //for (int i = 0; i < ptre_global.size; i++) {
    //  if (i == ptre_global.rank) {
    //    continue;
    //  }
    //  for (int j = 0; j < num_inputs; j++) {
    //    ptre_global.rdma_manager->InitTensorMR(i, names_[j], recv_tensors[j],
    //                                           send_tensors[j]);
    //  }
    //}
    //ptre_global.rdma_manager->MarkMRInitialized();
    //usleep((ptre_global.size - ptre_global.rank) * 1000000);
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("InitGlobalConsensus").Device(DEVICE_CPU),
                        InitGlobalConsensusOp);

REGISTER_OP("InitRemoteMr")
    .Attr("names: list(string)");
class InitRemoteMrOp : public OpKernel {
 public:
  explicit InitRemoteMrOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    bool peer_flag[ptre_global.size] = {};
    peer_flag[ptre_global.rank] = true;
    int done_flag = 0;
    while (!done_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      done_flag = 1;
      for (int i = 0; i < ptre_global.size; i++) {
        if (peer_flag[i]) {
          continue;
        }
        GrpcClient grpc_client(ptre_global.rank, i, ptre_global.grpc_hosts[i]);
        grpc_client.SetRdmaManager(ptre_global.rdma_manager);
        if (!ptre_global.rdma_manager->IsDlidSet(i)) {
          int ret = grpc_client.GetRemoteEnv();
          if (ret < 0) {
            done_flag = 0;
            continue;
          }
        }
        int client_status = 0;
        for (int j = 0; j < names_.size(); j++) {
          if (ptre_global.rdma_manager->IsRemoteMRSet(i, names_[j])) {
            continue;
          }
          int ret = grpc_client.GetRemoteAddress(names_[j]);
          if (ret < 0) {
            client_status = -1;
            break;
          }
        }
        if (client_status < 0) {
          done_flag = 0;
          continue;
        }
        if (!ptre_global.rdma_manager->IsRemoteParamMRSet(i)) {
          int ret = grpc_client.GetRemoteParamAddress();
          if (ret < 0) {
            done_flag = 0;
            continue;
          }
        }
        peer_flag[i] = true;
      }
    }
    std::cout << "Init RemoteMR done." << std::endl;
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("InitRemoteMr").Device(DEVICE_CPU),
                        InitRemoteMrOp);

REGISTER_OP("ConnectQps");
class ConnectQpsOp : public OpKernel {
 public:
  explicit ConnectQpsOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    int done_flag = 0;
    while (!done_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      done_flag = 1;
      for (int i = 0; i < ptre_global.size; i++) {
        if (i == ptre_global.rank) {
          continue;
        }
        int r = ptre_global.rdma_manager->ConnectQP(i);
        if (r < 0) {
          done_flag = 0;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ConnectQps").Device(DEVICE_CPU),
                        ConnectQpsOp);

REGISTER_OP("PushModel")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Attr("names: list(string)")
    .Input("vars: NumTensors * T");
class PushModelOp : public OpKernel {
 public:
  explicit PushModelOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    //std::cout << (void*) context->input(0).tensor_data().begin() << std::endl;
    int num_inputs = context->num_inputs();
    //std::cout << "PushTensor: " << names_[0] << std::endl;
    //std::lock_guard<std::mutex> l(ptre_global.mu);
    //while (!ptre_global.q.empty()) {
    //  ptre_global.q.pop();
    //}
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.cm.CopyTensorSend(names_[i], context->input(i));
    }
    int target_rank = ptre_global.cm.GetRandomTarget();
    //int target_rank = ptre_global.cm.GetIncNeighbor();
    ptre_global.q.push(target_rank);
    //ptre_global.cm.SetPushReady();
    //std::vector<const Tensor*> inputs;
    //for (int i = 0; i < num_inputs; i++) {
    //  const Tensor &input = context->input(i);
    //  inputs.push_back(&input);
    //}
    //ptre_global.cm.EnqueuePushList(inputs);
  }
 private:
  std::vector<std::string> names_;
};
REGISTER_KERNEL_BUILDER(Name("PushModel").Device(DEVICE_CPU), PushModelOp);

REGISTER_OP("GetRemoteVariable")
    .Attr("index: int = -1")
    .Attr("var_name: string = ''")
    .Output("var: float32");
class GetRemoteVariableOp : public OpKernel {
 public:
  explicit GetRemoteVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("index", &index_));
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* context) override {
    //Tensor other(ptre_global.cm.global_consensus(index_));
    usleep(1);
    Tensor other;
    if (index_ >= 0) {
      other = ptre_global.cm.global_consensus(index_);
    //}
    } else {
      //std::cout << "get_remote_variable with name: " << var_name_ << std::endl;
      other = ptre_global.cm.global_consensus(var_name_);
    }
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, other.shape(), &output));
    //std::copy(other.tensor_data().begin(), other.tensor_data().end(),
    //          const_cast<char*>(output->tensor_data().begin()));
    auto output_flat = output->flat<float>();
    output_flat = other.flat<float>();
  }

 private:
  int index_;
  std::string var_name_;
};
REGISTER_KERNEL_BUILDER(Name("GetRemoteVariable").Device(DEVICE_CPU),
                        GetRemoteVariableOp);

REGISTER_OP("GetRemoteVariables")
    .Output("out: float32");
class GetRemoteVariablesOp : public OpKernel {
 public:
  explicit GetRemoteVariablesOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    int num_vars = ptre_global.cm.num_vars();
    for (int i = 0; i < num_vars; i++) {
      Tensor other(ptre_global.cm.global_consensus(i));
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(i, other.shape(), &output));
      auto output_flat = output->flat<float>();
      output_flat = other.flat<float>();
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GetRemoteVariables").Device(DEVICE_CPU),
                        GetRemoteVariablesOp);

REGISTER_OP("GetSendTensor")
    .Attr("index: int")
    .Output("var: float32");
class GetSendTensorOp : public OpKernel {
 public:
  explicit GetSendTensorOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("index", &index_));
  }
  void Compute(OpKernelContext* context) override {
    Tensor* other(ptre_global.cm.send_tensor(index_));
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, other->shape(), &output));
    auto output_flat = output->flat<float>();
    output_flat = other->flat<float>();
  }

 private:
  int index_;
};
REGISTER_KERNEL_BUILDER(Name("GetSendTensor").Device(DEVICE_CPU),
                        GetSendTensorOp);

REGISTER_OP("AverageModelWithRemote")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T")
    .Output("outputs: NumTensors * T");
class AverageModelWithRemoteOp : public OpKernel {
 public:
  explicit AverageModelWithRemoteOp(OpKernelConstruction* context)
      : OpKernel(context) {
    //OP_REQUIRES_OK(context, context->GetAttr("var_sizes", &var_sizes_));
  }
  void Compute(OpKernelContext* context) override {
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      const Tensor &input = context->input(i);
      //Tensor input = context->mutable_input(i, true);
      Tensor* output;
      OP_REQUIRES_OK(context, context->allocate_output(i, input.shape(), &output));
      //OP_REQUIRES_OK(context,
      //               context->forward_input_or_allocate_output(
      //                   {i}, i, input.shape(), &output));
      // Load remote tensor from CM
      Tensor other(ptre_global.cm.global_consensus(i));
      //Tensor other(input.dtype(), input.shape());
      //auto other_flat = other.flat<float>();
      //for (int j = 0; j < var_sizes_[i]; j++) {
      //  other_flat(j) = 1.0;
      //}

      // Averaging
      //auto input_flat = input.flat<float>();
      //input_flat = 0.5 * (input_flat + other_flat);
      auto output_flat = output->flat<float>();
      output_flat = 0.5 * (input.flat<float>() + other.flat<float>());
      //std::copy(output->tensor_data().begin(), output->tensor_data().end(),
      //          const_cast<char*>(input.tensor_data().begin()));
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("AverageModelWithRemote").Device(DEVICE_CPU),
                        AverageModelWithRemoteOp);

//REGISTER_OP("PushModel")
//    .Attr("T: {float32}")
//    .Attr("NumTensors: int")
//    .Input("var: NumTensors * T");
//
//class PushModel : public OpKernel {
// public:
//  void Compute(OpKernelContext *context) override {
//    //ptre_global->PushModel
//  }
//};

//REGISTER_OP("Incoming")
//    .
REGISTER_OP("DummyListInput")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T");
class DummyListInputOp : public AsyncOpKernel {
 public:
  explicit DummyListInputOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    //std::cout << (void*) context->input(0).tensor_data().begin() << std::endl;
    int num_inputs = context->num_inputs();
    //std::lock_guard<std::mutex> l(ptre_global.mu);
    //while (!ptre_global.q.empty()) {
    //  ptre_global.q.pop();
    //}
    //for (int i = 0; i < num_inputs; i++) {
    //  ptre_global.cm.CopyTensorSend(names_[i], context->input(i));
    //}
    int target_rank = ptre_global.cm.GetRandomTarget();
    //ptre_global.q.push(target_rank);
    done();
  }
};
REGISTER_KERNEL_BUILDER(Name("DummyListInput").Device(DEVICE_CPU), DummyListInputOp);

REGISTER_OP("DummySingleInput")
    .Input("var: float32");
class DummySingleInputOp : public AsyncOpKernel {
 public:
  explicit DummySingleInputOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    //std::cout << (void*) context->input(0).tensor_data().begin() << std::endl;
    //int num_inputs = context->num_inputs();
    //std::lock_guard<std::mutex> l(ptre_global.mu);
    //while (!ptre_global.q.empty()) {
    //  ptre_global.q.pop();
    //}
    //for (int i = 0; i < num_inputs; i++) {
    //  ptre_global.cm.CopyTensorSend(names_[i], context->input(i));
    //}
    int target_rank = ptre_global.cm.GetRandomTarget();
    //ptre_global.q.push(target_rank);
    done();
  }
};
REGISTER_KERNEL_BUILDER(Name("DummySingleInput").Device(DEVICE_CPU), DummySingleInputOp);


REGISTER_OP("ApplyModelAveraging")
    .Input("var: Ref(T)")
    .Input("remote: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype");
template <typename T>
class ApplyModelAveragingOp : public OpKernel {
 public:
  explicit ApplyModelAveragingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, false);
    const Tensor& remote = ctx->input(1);
    //const Device& device = ctx->template eigen_device<Device>();
//  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
//                  typename TTypes<T>::ConstFlat remote) {
//    var.device(d) = 0.5 * (var + remote);
//  }
    //functor::ApplyModelAveraging<Device, T>()(
    //    device, var.flat<T>(), remote.flat<T>());
    var.flat<T>() = 0.5 * (var.flat<T>() + remote.flat<T>());

    ctx->forward_ref_input_to_ref_output(0, 0);
  }
};
REGISTER_KERNEL_BUILDER(
    Name("ApplyModelAveraging").Device(DEVICE_CPU).TypeConstraint<float>("float"),
    ApplyModelAveragingOp<float>);

REGISTER_OP("IsNewIncoming")
    .Output("out: bool");
class IsNewIncomingOp : public OpKernel {
 public:
  explicit IsNewIncomingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) {
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    usleep(1);
    //volatile bool* ret = ptre_global.cm.is_new_incoming_ptr();
    bool* ret = ptre_global.cm.is_new_incoming_ptr();
    //std::cout << "IsNewIncomingOp: *is_new_incoming_ptr = " << *ret << std::endl;
    std::copy(ret, ret + sizeof(bool), const_cast<char*>(output->tensor_data().begin()));
    //std::cout << "IsNewIncomingOp: *((bool*) output->tensor_data().begin()) = " << *((bool*) output->tensor_data().begin()) << std::endl;
  }
};
REGISTER_KERNEL_BUILDER(
    Name("IsNewIncoming").Device(DEVICE_CPU), IsNewIncomingOp);

REGISTER_OP("MarkNoNew");
class MarkNoNewOp : public OpKernel {
 public:
  explicit MarkNoNewOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) {
    ptre_global.cm.MarkNoNew();
    //volatile bool* ret = ptre_global.cm.is_new_incoming_ptr();
    bool* ret = ptre_global.cm.is_new_incoming_ptr();
    //std::cout << "MarkNoNewOp: *is_new_incoming_ptr = " << *ret << std::endl;
  }
};
REGISTER_KERNEL_BUILDER(
    Name("MarkNoNew").Device(DEVICE_CPU), MarkNoNewOp);

namespace functor {
template <>
struct Modelaverage<CPUDevice> {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::ConstFlat other) {
    var.device(d) = var.constant(float(0.5)) * (var + other);
  }
};

template <>
struct CopyTensorToSendBuf<CPUDevice> {
  void operator()(const CPUDevice& d,
                  typename TTypes<float>::Flat src,
                  typename TTypes<float>::Flat dst) {
    auto bytes = sizeof(float) * src.size();
    memcpy(dst.data(), src.data(), bytes);
  }
};

}  // namespace functor

//REGISTER_OP("ResourceGetRemote")
//  .Input("var: resource")
//  .Attr("var_name: string")
//  .Output("remote: resource")
//template <typename Device>
//class GetRemoteOp : public OpKernel {
// public:
//  explicit GetRemoteOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
//    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
//  }
//  void Compute(OpKernelContext* ctx) {
//  }
// private:
//  string var_name_;
//};

static Status ModelaverageShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                  // var
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceModelaverage")
  .Input("var: resource")
  .Attr("var_name: string")
  .SetShapeFn(ModelaverageShapeFn);
template <typename Device>
class ModelaverageOp : public OpKernel {
 public:
  explicit ModelaverageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* ctx) {
    bool* ret = ptre_global.cm.is_new_incoming_ptr();
    if (!*ret) {
      return;
    }
    Tensor var;
    core::RefCountPtr<Var> ref;
    //TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &ref));
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);

    var = *ref->tensor();

    const Device& d = ctx->template eigen_device<Device>();
    const Tensor other(ptre_global.cm.global_consensus(var_name_));
    functor::Modelaverage<Device>()(d, var.flat<float>(), other.flat<float>());
    ptre_global.incoming_state = 2;  // Used
  }

 private:
  string var_name_;
};
REGISTER_KERNEL_BUILDER(Name("ResourceModelaverage")
                            .Device(DEVICE_CPU)
                            .HostMemory("var"),
                        ModelaverageOp<CPUDevice>);
#ifdef GOOGLE_CUDA
namespace functor {
template <>
void Modelaverage<GPUDevice>::operator()(const GPUDevice& d,
    typename TTypes<float>::Flat var,
    //typename TTypes<float>::Flat other);
    typename TTypes<float>::ConstFlat other);
extern template struct Modelaverage<GPUDevice>;
}  // namespace functor
REGISTER_KERNEL_BUILDER(Name("ResourceModelaverage")
                            .Device(DEVICE_GPU)
                            .HostMemory("var"),
                        ModelaverageOp<GPUDevice>);
#endif  // GOOGLE_CUDA


REGISTER_OP("ResourcePushTensor")
  .Input("var: resource")
  .Attr("var_name: string");
template <typename Device>
class PushTensorOp : public OpKernel {
 public:
  explicit PushTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* ctx) {
    if (!ptre_global.is_push_step) {
      return;
    }
    Tensor var;
    core::RefCountPtr<Var> ref;
    //TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &ref));
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    var = *ref->tensor();
    const Device& d = ctx->template eigen_device<Device>();
    Tensor* send_tensor = ptre_global.cm.send_tensor(var_name_);
    functor::CopyTensorToSendBuf<Device>()(d, var.flat<float>(),
        send_tensor->flat<float>());
  }
 private:
  string var_name_;
};
REGISTER_KERNEL_BUILDER(Name("ResourcePushTensor")
                            .Device(DEVICE_CPU)
                            .HostMemory("var"),
                        PushTensorOp<CPUDevice>);
#ifdef GOOGLE_CUDA
namespace functor {
template <>
void CopyTensorToSendBuf<GPUDevice>::operator()(const GPUDevice& d,
                  typename TTypes<float>::Flat src,
                  typename TTypes<float>::Flat dst);
extern template struct CopyTensorToSendBuf<GPUDevice>;
}  // namespace functor
REGISTER_KERNEL_BUILDER(Name("ResourcePushTensor")
                            .Device(DEVICE_GPU)
                            .HostMemory("var"),
                        PushTensorOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow


extern "C" {

int ptre_init(int size, int rank, char* grpc_hosts_file,
              int selection_strategy) {
  tensorflow::InitComm(size, rank, grpc_hosts_file);
  tensorflow::ptre_global.cm.InitPeerSelector(selection_strategy);
}

int ptre_init_rdma_grpc_service() {
  tensorflow::InitGrpcService();
}

int ptre_size() {
  return tensorflow::ptre_global.size;
}

int ptre_rank() {
  return tensorflow::ptre_global.rank;
}

bool ptre_is_new_incoming() {
  bool* ret = tensorflow::ptre_global.cm.is_new_incoming_ptr();
  return *ret;
}

void ptre_mark_no_new() {
  bool* ret = tensorflow::ptre_global.cm.is_new_incoming_ptr();
  if (*ret == true && tensorflow::ptre_global.incoming_state == 2) {
    *ret = false;
    tensorflow::ptre_global.incoming_state = 0;
  }
}

void ptre_enqueue_push() {
  //int target_rank = tensorflow::ptre_global.cm.GetRandomTarget();
  int target_rank = tensorflow::ptre_global.cm.get_peer();
  tensorflow::ptre_global.q.push(target_rank);
}

void ptre_set_push() {
  tensorflow::ptre_global.is_push_step = true;
}

void ptre_unset_push() {
  tensorflow::ptre_global.is_push_step = false;
}

}

