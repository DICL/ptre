#include "ptre/kernels/ptre_op_helpers.h"

namespace tensorflow {

void MaybeForwardRefInputToRefOutput(OpKernelContext* ctx, int input,
                                     int output) {
  if (ctx->input_dtype(input) != DT_RESOURCE) {
    ctx->forward_ref_input_to_ref_output(input, output);
  }
}

}  // namespace tensorflow
