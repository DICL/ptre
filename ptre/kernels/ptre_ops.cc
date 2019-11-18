//#include "ptre/core/ptre_global.h"

#include <thread>

#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace {

struct PtreGlobal {

  ptre::ConsensusManager cm;
  //std::vector<Tensor*> remote_tensors;

  mutex mu;

  //std::queue<PtreRequest> request_queue;

  // Background thread running PTRE communication.
  std::thread background_thread;

  bool new_incoming;

  int rank = 0;
  int size = 1;

  ~PtreGlobal() {
    if (background_thread.joinable()) {
      //shut_down = true;
      background_thread.join();
    }
  }

};

void InitRdma() {
}

void BackgroundThreadLoop() {
  //TODO: 1. Init Ptre
  while(true) {
    // TODO: Eliminate the need for thread sleep by making all activity
    // depend on other activity (e.g. condition or MPI waits).
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    //TODO: Fetch a push task from a task queue.
    
  }
}

//void PtreRemoteVariables(

static PtreGlobal ptre_global;

}  // namespace

REGISTER_OP("InitGlobalConsensus")
    .Attr("T: {float32}")
    .Attr("NumTensors: int")
    .Input("vars: NumTensors * T");
class InitGlobalConsensusOp : public OpKernel {
 public:
  explicit InitGlobalConsensusOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    std::vector<const Tensor*> inputs;
    int num_inputs = context->num_inputs();
    for (int i = 0; i < num_inputs; i++) {
      const Tensor &input = context->input(i);
      inputs.push_back(&input);
    }
    ptre_global.cm.InitGlobalConsensus(inputs);
  }
};
REGISTER_KERNEL_BUILDER(Name("InitGlobalConsensus").Device(DEVICE_CPU),
                        InitGlobalConsensusOp);

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
