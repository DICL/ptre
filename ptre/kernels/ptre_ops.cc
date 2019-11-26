//#include "ptre/core/ptre_global.h"

#include <thread>
#include <iostream>
#include <vector>

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

  // Background thread running PTRE communication.
  std::thread grpc_server_thread;
  std::thread background_thread;

  bool new_incoming;

  int rank;
  int size;

  ~PtreGlobal() {
    if (background_thread.joinable()) {
      //shut_down = true;
      background_thread.join();
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

void InitComm(int size, int rank) {
  ptre_global.size = size;
  ptre_global.rank = rank;
  std::cout << "Initializing RdmaManager" << std::endl;
  ptre_global.rdma_manager = new RdmaManager(size, rank);
  std::cout << "Running grpc server" << std::endl;
  ptre_global.grpc_server_thread = std::thread(RunGrpcServer);
}

void InitRemoteMR() {
}

void BackgroundThreadLoop() {
  /// TODO: 1. Init MR
  while(true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    /// TODO: Fetch a push task from a task queue.
  }
}


void InitPtreOnce() {
  //while(true) {
  //  if (ptre_global.rdma_manager->IsMRInitialized()) {
  //    break;
  //  }
  //}

  //ptre_global.background_thread = std::thread(BackgroundThreadLoop);
}

}  // namespace

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
    std::vector<const Tensor*> inputs;
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      const Tensor &input = context->input(i);
      inputs.push_back(&input);
    }
    ptre_global.cm.InitGlobalConsensus(inputs);
    auto recv_tensors = ptre_global.cm.GetGlobalConsensusList();
    auto send_tensors = ptre_global.cm.GetSendTensorsList();
    for (int i = 0; i < ptre_global.size; i++) {
      if (i == ptre_global.rank) {
        continue;
      }
      for (int j = 0; j < num_inputs; j++) {
        ptre_global.rdma_manager->InitTensorMR(i, names_[j], *recv_tensors[j],
                                               *send_tensors[j]);
      }
    }
    ptre_global.rdma_manager->MarkMRInitialized();
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
        GrpcClient grpc_client(i);
        grpc_client.SetRdmaManager(ptre_global.rdma_manager);
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

REGISTER_OP("PushModel")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T");
class PushModelOp : public OpKernel {
 public:
  explicit PushModelOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    std::vector<const Tensor*> inputs;
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      const Tensor &input = context->input(i);
      inputs.push_back(&input);
    }
    ptre_global.cm.EnqueuePush(inputs);
  }
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
    
}  // namespace tensorflow
