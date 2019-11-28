//#include "ptre/core/ptre_global.h"

#include <unistd.h>
#include <thread>
#include <iostream>
#include <vector>
#include <queue>

#include "ptre/kernels/ptre_op_helpers.h"
#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/rdma/rdma_manager.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using ::ptre::ConsensusManager;
using ::ptre::RdmaManager;
using ::ptre::RdmaServiceImpl;
using ::ptre::GrpcClient;

namespace {
struct PtreGlobal {
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
  std::lock_guard<std::mutex> l(ptre_global.mu);
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
  while(true) {
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

void InitComm(int size, int rank) {
  /// First Initialization step
  ptre_global.size = size;
  ptre_global.rank = rank;
  ptre_global.cm.set_size(size);
  ptre_global.cm.set_rank(rank);
  std::cout << "Initializing RdmaManager" << std::endl;
  ptre_global.rdma_manager = new RdmaManager(size, rank);
  ptre_global.cm.SetRdmaManager(ptre_global.rdma_manager);
  std::cout << "Running grpc server" << std::endl;
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);
  ptre_global.background_thread = std::thread(BackgroundThreadLoop);
}
}  // namespace

namespace functor {
template <typename T>
struct ApplyModelAveraging<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstFlat remote) {
    var.device(d) = 0.5 * (var + remote);
  }
};
}  // namespace functor

REGISTER_OP("InitComm")
    .Attr("size: int")
    .Attr("rank: int");
class InitCommOp : public OpKernel {
 public:
  explicit InitCommOp(OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr<int>("size", &size_);
    context->GetAttr<int>("rank", &rank_);
  }
  void Compute(OpKernelContext* context) override {
    InitComm(size_, rank_);
  }
 private:
  int size_;
  int rank_;
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
    int done_flag = 0;
    while (!done_flag) {
      std::this_thread::sleep_for(std::chrono::milliseconds(3000));
      done_flag = 1;
      for (int i = 0; i < ptre_global.size; i++) {
        if (i == ptre_global.rank) {
          continue;
        }
        GrpcClient grpc_client(ptre_global.rank, i);
        grpc_client.SetRdmaManager(ptre_global.rdma_manager);
        if (!ptre_global.rdma_manager->IsDlidSet(i)) {
          int ret = grpc_client.GetRemoteEnv();
          if (ret < 0) {
            done_flag = 0;
          }
        }
        for (int j = 0; j < names_.size(); j++) {
          if (ptre_global.rdma_manager->IsRemoteMRSet(i, names_[j])) {
            continue;
          }
          int ret = grpc_client.GetRemoteAddress(names_[j]);
          if (ret < 0) {
            done_flag = 0;
          }
        }
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
    std::lock_guard<std::mutex> l(ptre_global.mu);
    while (!ptre_global.q.empty()) {
      ptre_global.q.pop();
    }
    for (int i = 0; i < num_inputs; i++) {
      ptre_global.cm.CopyTensorSend(names_[i], context->input(i));
    }
    int target_rank = ptre_global.cm.GetRandomTarget();
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
    .Attr("index: int")
    .Output("var: float32");
class GetRemoteVariableOp : public OpKernel {
 public:
  explicit GetRemoteVariableOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("index", &index_));
  }
  void Compute(OpKernelContext* context) override {
    Tensor other(ptre_global.cm.global_consensus(index_));
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, other.shape(), &output));
    //usleep(1);
    //std::copy(other.tensor_data().begin(), other.tensor_data().end(),
    //          const_cast<char*>(output->tensor_data().begin()));
    auto output_flat = output->flat<float>();
    output_flat = other.flat<float>();
  }

 private:
  int index_;
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
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
template <typename Device, typename T>
class ApplyModelAveragingOp : public OpKernel {
 public:
  explicit ApplyModelAveragingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    const Tensor& remote = ctx->input(1);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(remote.shape()),
        errors::InvalidArgument("var and remote do not have the same shape",
                                var.shape().DebugString(), " ",
                                remote.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyModelAveraging<Device, T>()(
        device, var.flat<T>(), remote.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyModelAveraging").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyModelAveragingOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyModelAveraging")                \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyModelAveragingOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS
    
}  // namespace tensorflow
