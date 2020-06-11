#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace {
struct TestGlobal {
  //bool is_incoming;
  Tensor* is_incoming;

  TestGlobal() {
    is_incoming = new Tensor(DT_INT32, TensorShape({}));
    auto flat = is_incoming->flat<int32>();
    flat(0) = 3;
  }

};

static TestGlobal test_global;
}

REGISTER_OP("IsIncoming")
    .Output("incoming: int32");
class IsIncomingOp : public OpKernel {
 public:
  explicit IsIncomingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output_tensor));
    //auto output_flat = output_tensor->flat<int32>();
    //output_flat(0) = 1;
    //output_tensor = test_global.is_incoming;
  }
};
REGISTER_KERNEL_BUILDER(Name("IsIncoming").Device(DEVICE_CPU), IsIncomingOp);

//REGISTER_OP("AverageRemote")
//    .Input("in

}  // namespace tensorflow
