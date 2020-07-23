#define EIGEN_USE_THREADS
#define PTRE_GPU_REDUCE
//#define PTRE_CPU_REDUCE

#include "ptre/tensorflow/kernels/ptre_ops.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <typeinfo>
#include <unistd.h>
#include <vector>
#include <functional>
#include <mutex>
#include <sstream>
//#include <shared_mutex>
//#include "ptre/lib/shared_mutex.h"

#include <absl/base/macros.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>

#include "ptre/common/operations.h"
#include "ptre/common/communication/rdma/rdma.h"
//#include "ptre/tensorflow/types.h"
#include "ptre/tensorflow/kernels/job_def.h"
#include "ptre/lib/cache_ctl.h"
#include "ptre/lib/concurrent_queue.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"


//using ptre::PushJob;

using namespace tensorflow;
using namespace ptre::common;

namespace ptre {
namespace tensorflow {

using std::string;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;


// --------------------------------------------------------------------------
// Common functors

namespace functor {

template <typename T>
struct MemcpyToHost<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Flat src,
                  typename TTypes<T>::Flat dst) {
    auto bytes = sizeof(T) * src.size();
    memcpy(dst.data(), src.data(), bytes);
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void MemcpyToHost<GPUDevice, T>::operator()(   \
      const GPUDevice& d, typename TTypes<T>::Flat src, \
      typename TTypes<T>::Flat dst);                    \
  extern template struct MemcpyToHost<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
#endif  // GOOGLE_CUDA

}  // namespace functor

// --------------------------------------------------------------------------

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

REGISTER_OP("RegisterVariables")
    .Input("vars: NumTensors * T")
    .Attr("T: numbertype")
    .Attr("NumTensors: int")
    .Attr("names: list(string)");
class RegisterVariablesOp : public OpKernel {
 public:
  explicit RegisterVariablesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    context->GetAttr("names", &names_);
  }
  void Compute(OpKernelContext* context) override {
    ptre::common::RegisterTrainableVariables(context, names_);
  }
 private:
  std::vector<string> names_;
};
REGISTER_KERNEL_BUILDER(Name("RegisterVariables").Device(DEVICE_CPU),
                        RegisterVariablesOp);

REGISTER_OP("Broadcast")
    .Input("var: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("root_rank: int");
class BroadcastOp : public OpKernel {
 public:
  explicit BroadcastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("root_rank", &root_rank_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto node_name = name();
    auto tensor = ctx->input(0);
    Tensor* output = nullptr;
    if (ptre::common::ptre_rank() == root_rank_) {
      ctx->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor.shape(), &output));
    }
    if (output == nullptr) {
      output = ctx->mutable_output(0);
    }
    ptre::common::PtreBroadcast(
        const_cast<char*>(output->tensor_data().data()),
        output->tensor_data().size(), root_rank_, node_name);
  }

 private:
  int root_rank_;
};
REGISTER_KERNEL_BUILDER(Name("Broadcast").Device(DEVICE_CPU), BroadcastOp);

// --------------------------------------------------------------------------

REGISTER_OP("PtreModelaverage")
  .Input("tensor: T")
  .Output("avg: T")
  .Attr("T: numbertype")
  .Attr("var_name: string")
  .Attr("modelaverage_op: int")
  .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
  });
template <typename Device, typename T>
class PtreModelaverageOp : public AsyncOpKernel {
 public:
  explicit PtreModelaverageOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("modelaverage_op", &modelaverage_op_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto node_name = name();
    auto tensor = ctx->input(0);
    ptre::common::ModelaverageOp modelaverage_op =
      static_cast<ptre::common::ModelaverageOp>(modelaverage_op_);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, tensor.shape(), &output), done);
//LOG(INFO) << __FUNCTION__ << "tensor name=" << var_name_;
    Status enqueue_result = EnqueueTensorModelaverage(
        ctx, tensor, *output, var_name_,
        [ctx, done](const Status& status) {
          ctx->SetStatus(status);
          done();
        }, modelaverage_op);
    OP_REQUIRES_OK_ASYNC(ctx, enqueue_result, done);
  }

 private:
  string var_name_;
  int modelaverage_op_;
};
REGISTER_KERNEL_BUILDER(Name("PtreModelaverage")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        PtreModelaverageOp<CPUDevice, float>);

// --------------------------------------------------------------------------

REGISTER_OP("PtrePublish")
  .Input("var: T")
  .Attr("T: numbertype")
  .Attr("var_name: string")
  .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s = ShapeOrHandleShape(c, 0);
      if (c->num_outputs() > 0) {
        c->set_output(0, s);
      }
      return Status::OK();
  });
template <typename Device, typename T>
class PtrePublishOp : public AsyncOpKernel {
 public:
  explicit PtrePublishOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto var = ctx->input(0);
    const Device& d = ctx->template eigen_device<Device>();
    Tensor* ready_tensor = GetReadyTensor(var_name_);
    functor::MemcpyToHost<Device, T>()(d, var.flat<T>(),
                                       ready_tensor->flat<T>());
    EnqueueTensorPull(var_name_);
    done();
    // TODO: Will this async memcpy be effective?
    //Status enqueue_result = EnqueueTensorMemcpyToHost(...
    //    [ctx, done](const Status& status) {
    //      ctx->SetStatus(status);
    //      done();
    //    });
    //OP_REQUIRES_OK_ASYNC(ctx, enqueue_result, done);
  }

 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("PtrePublish") \
                              .Device(DEVICE_##D)      \
                              .TypeConstraint<T>("T"), \
                          PtrePublishOp<D##Device, T>);

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);
TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS
  
// --------------------------------------------------------------------------

namespace functor {

template <typename Device, typename T>
struct Foo {
  void operator()(const Device& d);
};

template <typename T>
struct Foo<CPUDevice, T> {
  void operator()(const CPUDevice& d) {
    //LOG(INFO) << d.DebugString();
  }
};

template <typename T>
struct Foo<GPUDevice, T> {
  void operator()(const GPUDevice& d) {
    //LOG(INFO) << d.DebugString();
  }
};

}  // namespace functor

REGISTER_OP("PtreAllreduce")
  .Input("tensor: T")
  .Output("sum: T")
  .Attr("T: numbertype")
  .Attr("reduce_op: int")
  .SetShapeFn([](InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });
template <typename Device, typename T>
class PtreAllreduceOp : public AsyncOpKernel {
 public:
  explicit PtreAllreduceOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    auto node_name = name();
    //auto device = GetDeviceID(ctx);
    auto d = ctx->device();
//LOG(INFO) << "Device=" << d->name() << ", Name=" << node_name;
    //string device_name;
    //functor::Foo<Device, T>(d);
    auto tensor = ctx->input(0);
    ptre::common::ReduceOp reduce_op =
        static_cast<ptre::common::ReduceOp>(reduce_op_);
    Tensor* output;
#if 1
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, tensor.shape(), &output), done);
#else
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor.shape(), &output));
#endif
//LOG(INFO) << __FUNCTION__ << "\n***tensor=" << (uint64_t) tensor.tensor_data().data() << ", output=" << (uint64_t) output->tensor_data().data() << ", name=" << node_name << ", num_elements=" << tensor.NumElements();
    Status enqueue_result = EnqueueTensorAllreduce(
        ctx, tensor, *output, node_name,
        [ctx, done](const Status& status) {
//LOG(INFO) << __FUNCTION__ << ": ctx=" << (uint64_t) ctx << ", num_elem=" << ctx->input(0).NumElements();
          ctx->SetStatus(status);
          done();
        }, reduce_op);
    OP_REQUIRES_OK_ASYNC(ctx, enqueue_result, done);
  }

 private:
  int reduce_op_;
};
REGISTER_KERNEL_BUILDER(Name("PtreAllreduce").Device(DEVICE_CPU),
                        PtreAllreduceOp<CPUDevice, float>);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER_KERNEL_BUILDER(Name("PtreAllreduce").Device(DEVICE_GPU),
                        PtreAllreduceOp<GPUDevice, float>);
#endif

// --------------------------------------------------------------------------

REGISTER_OP("PtreResourceRemoteVariable")
  .Input("var: resource")
  .Output("output: T")
  .Output("num_agg: int32")
  .Attr("T: numbertype")
  .Attr("var_name: string")
  .SetShapeFn([](InferenceContext* c) {
    ShapeHandle s = ShapeOrHandleShape(c, 0);
    if (c->num_outputs() > 0) {
      c->set_output(0, s);
    }
    return Status::OK();
  });
class PtreRemoteVariableOp : public AsyncOpKernel {
 public:
  explicit PtreRemoteVariableOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    Tensor var = *ref->tensor();

    Tensor* output = NULL;
    Tensor* num_agg = NULL;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, var.shape(), &output),
                         done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(1, TensorShape({}),
                                                   &num_agg),
                         done);

    auto enqueue_result = EnqueueGetRemoteVariable(ctx, var_name_, output,
        num_agg,
        [ctx, done](const Status& status) {
          ctx->SetStatus(status);
          done();
        });
    OP_REQUIRES_OK_ASYNC(ctx, enqueue_result, done);
  }

 private:
  string var_name_;
};
REGISTER_KERNEL_BUILDER(Name("PtreResourceRemoteVariable").Device(DEVICE_CPU),
                        PtreRemoteVariableOp);


namespace functor {
template <typename T>
struct Modelaverage<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar m,
                  typename TTypes<T>::ConstFlat other) {
    var.device(d) = (var + other) / m();
  }
};

template <typename T>
struct LinearWeightedAverageApprox<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar c1,
                  typename TTypes<T>::ConstFlat other,
                  typename TTypes<T>::ConstScalar c2) {
    var.device(d) = c1() * var + c2() * other;
  }
};

template <typename T>
struct CopyTensorToSendBuf<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Flat src,
                  typename TTypes<T>::Flat dst) {
    auto bytes = sizeof(T) * src.size();
    memcpy(dst.data(), src.data(), bytes);
  }
};

template <typename T>
struct CopyRemoteToVar<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstFlat remote) {
    auto bytes = sizeof(T) * var.size();
    memcpy(var.data(), remote.data(), bytes);
  }
};

}  // namespace functor

/*
void StopPullTasks() {

}
*/

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
  .Attr("T: numbertype")
  .Attr("var_name: string")
  .SetShapeFn(ModelaverageShapeFn);
template <typename Device, typename T>
class ModelaverageOp : public AsyncOpKernel {
 public:
  explicit ModelaverageOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
#if 0
    auto rvar = ptre_global.cm->remote_variable(var_name_);
    if (!rvar) {
      done();
      return;
    }

    StopPullTasks(var_name_);
    rvar->StopAggregation();  // DEBUG
#ifdef PTRE_CPU_REDUCE
    int num_incomings = rvar->agg_count() - 1;
    if (num_incomings > 0) {
      Eigen::ThreadPoolDevice rdev(ptre_global.reduce_eigen_pool,
          NUM_REDUCE_EIGEN_THREADS);
      rvar->Reduce(rdev);  // DEBUG
      core::RefCountPtr<Var> ref;
      LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
      Tensor var = *ref->tensor();
      const Tensor remote = *rvar->tensor();
      const Device& d = ctx->template eigen_device<Device>();
      functor::CopyRemoteToVar<Device, T>()(d, var.flat<T>(), remote.flat<T>());
    }
#else
#ifdef PTRE_GPU_REDUCE
    int num_incomings = rvar->agg_count();
    if (num_incomings > 0) {
      core::RefCountPtr<Var> ref;
      LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
      Tensor var = *ref->tensor();

      T m_type_t = T(num_incomings + 1);
      typename TTypes<T>::ConstScalar m(&m_type_t, 1);

      const Tensor remote = *rvar->tensor();
      const Device& d = ctx->template eigen_device<Device>();
      functor::Modelaverage<Device, T>()(d, var.flat<T>(), m,
          remote.flat<T>());
      //rvar->set_last_key(ptre_global.local_step);
    }
#endif  // #ifdef PTRE_GPU_REDUCE
#endif  // #ifdef PTRE_CPU_REDUCE
    done();
    rvar->StartAggregation();
    EnqueuePullTasks(var_name_, ptre_global.num_push);

    ptre_global.agg_cnts[ptre_global.local_step][var_name_] = num_incomings;
#else
    LOG(ERROR) << "NOT IMPLEMENTED.";
#endif
  }

 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("ResourceModelaverage") \
                              .Device(DEVICE_##D)      \
                              .TypeConstraint<T>("T"), \
                          ModelaverageOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void Modelaverage<GPUDevice, T>::operator()(          \
      const GPUDevice& d, typename TTypes<T>::Flat var, \
      typename TTypes<T>::ConstScalar m,                \
      typename TTypes<T>::ConstFlat other);             \
  extern template struct Modelaverage<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
}  // namespace functor
#endif  // GOOGLE_CUDA

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if 0
void PtreClearQueueAndEnqueueRequest() {
  auto&& req_q = ptre_global.req_q;
  auto&& req_mu = ptre_global.req_q_mu;
  req_mu.lock();
  while (req_q.size() > 0) {
    req_q.pop();
  }
  for (int i = 0; i < ptre_global.num_trainable_variables; i++) {
    ptre_global.rdma_mgr->InitPush(i);
  }
  auto req = std::make_shared<PushRequest>(ptre_global.num_push,
      ptre_global.local_step, ptre_global.size,
      ptre_global.cm->variable_names());
  req_q.push(req);
  /*
  LOG(INFO) << "SEND COUNT TOTAL = " << ptre_global.send_cnt_total;
  ptre_global.send_cnt_total = 0;
  */
  req_mu.unlock();
}
#endif

void ClearTasks(const string& var_name) {
#if 0
  //LOG(INFO) << "Clear Tasks: " << var_name;
  auto&& q = ptre_global.push_q;
  auto&& mu = ptre_global.push_q_mu;
  mu.lock();
  for (int i = 0; i < q.size(); i++) {
    auto task = q.front();
    q.pop();
    if (task->var_name().compare(var_name)) {  // they differ
      q.push(task);
    } else {
      CancelPushVar(task->dst(), task->var_name());
    }
  }
  mu.unlock();
  ptre_global.rpn_checker_mu.lock();
  auto it = ptre_global.rpn_checker.begin();
  while (it != ptre_global.rpn_checker.end()) {
    if (!it->second.compare(var_name)) {
      it = ptre_global.rpn_checker.erase(it);
    } else {
      it++;
    }
  }
  ptre_global.rpn_checker_mu.unlock();
#endif
}


REGISTER_OP("ResourcePushTensor")
  .Input("var: resource")
  .Attr("T: numbertype")
  .Attr("var_name: string");
template <typename Device, typename T>
class PushTensorOp : public OpKernel {
 public:
  explicit PushTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void Compute(OpKernelContext* ctx) {
#if 0
    if (ptre_global.push_step_state != 1) return;
    std::lock_guard<std::mutex> guard(ptre_global.push_var_mus[var_name_]);

    auto pvar = ptre_global.rdma_mgr->push_variable(var_name_);
    if (!pvar) return;

    pvar->StopPush();
    ClearTasks(var_name_);

    /*
    ptre_global.push_op_mu.lock();
    ptre_global.push_op_cnt++;
    if (ptre_global.push_op_cnt == 1) {
      PtreClearQueueAndEnqueueRequest();
    } else if (ptre_global.push_op_cnt == ptre_global.num_trainable_variables) {
      ptre_global.push_op_cnt = 0;
    }
    ptre_global.push_op_mu.unlock();
    */

    Tensor var;
    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    var = *ref->tensor();
    const Device& d = ctx->template eigen_device<Device>();
    /*
    struct ibv_mr* mr = ptre_global.rdma_mgr->GetMR(ptre::BUF_TYPE_SEND_BUF,
        var_name_);
    if (!mr) {
      LOG(ERROR) << "buf not found: " << ptre::BUF_TYPE_SEND_BUF << ", " << var_name_;
      exit(1);
    }
    T* send_buf = (T*) mr->addr;
    */
    T* send_buf = (T*) pvar->data();
    typename TTypes<T>::Flat send_flat(send_buf, var.flat<T>().size());
    /*
    LOG(INFO) << "\n" << "COPYTOSENDBUF " << var_name_;
    */
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), send_flat);

    pvar->StartPush();
    //ptre_global.rdma_mgr->SetPushReady(var_name_);
#else
    LOG(ERROR) << "NOT IMPLEMENTED.";
#endif
  }
 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("ResourcePushTensor")   \
                              .Device(DEVICE_##D)      \
                              .HostMemory("var")       \
                              .TypeConstraint<T>("T"), \
                          PushTensorOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void CopyTensorToSendBuf<GPUDevice, T>::operator()(   \
      const GPUDevice& d, typename TTypes<T>::Flat src, \
      typename TTypes<T>::Flat dst);                    \
  extern template struct CopyTensorToSendBuf<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
}  // namespace functor
#endif  // GOOGLE_CUDA

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

REGISTER_OP("PtreResourcePublishVariable")
  .Input("var: resource")
  .Attr("T: numbertype")
  .Attr("var_name: string");
template <typename Device, typename T>
class PublishVariableOp : public AsyncOpKernel {
 public:
  explicit PublishVariableOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
#if 1
    /*
    auto enqueue_result = EnqueuePublish(ctx, var_name_,
        [ctx, done](const Status& status) {
          ctx->SetStatus(status);
          done();
        });
    OP_REQUIRES_OK_ASYNC(ctx, enqueue_result, done);
    */
    LOG(ERROR) << "NOT IMPLEMENTED.";
#else

    auto pvar = ptre_global.rdma_mgr->pull_variable(var_name_);
    if (!pvar) {
      done();
      return;
    }

    core::RefCountPtr<Var> ref;
    LookupResource(ctx, HandleFromInput(ctx, 0), &ref);
    Tensor var = *ref->tensor();

    T* next_buf = (T*) pvar->next_data();
    typename TTypes<T>::Flat next_flat(next_buf, var.flat<T>().size());
    const Device& d = ctx->template eigen_device<Device>();
    pvar->SetNextKey(ptre_global.local_step + 1);
    functor::CopyTensorToSendBuf<Device, T>()(d, var.flat<T>(), next_flat);
    done();

    pvar->Switch();

#ifdef PTRE_CPU_REDUCE
    auto rvar = ptre_global.cm->remote_variable(var_name_);
    if (rvar) {
      Eigen::ThreadPoolDevice adev(ptre_global.agg_eigen_pool,
          NUM_AGG_EIGEN_THREADS);
      rvar->Aggregate(pvar->curr_data(), adev);
//LOG(INFO) << var.DeviceSafeDebugString() << "==" << ((float*) pvar->curr_data())[0] << "==" << rvar->tensor()->DebugString();  // DEBUG
    }
#endif
#endif
  }
 private:
  string var_name_;
};
#define REGISTER_KERNELS(D, T)                         \
  REGISTER_KERNEL_BUILDER(Name("PtreResourcePublishVariable")   \
                              .Device(DEVICE_##D)      \
                              .TypeConstraint<T>("T"), \
                          PublishVariableOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void CopyTensorToSendBuf<GPUDevice, T>::operator()(   \
      const GPUDevice& d, typename TTypes<T>::Flat src, \
      typename TTypes<T>::Flat dst);                    \
  extern template struct CopyTensorToSendBuf<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
}  // namespace functor
#endif  // GOOGLE_CUDA

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace tensorflow


}  // namespace ptre
