from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptre.tensorflow import call_generic
from ptre.tensorflow import allreduce
from ptre.tensorflow import barrier
from ptre.tensorflow import finalize
from ptre.tensorflow import init
from ptre.tensorflow import is_new_incoming, get_incoming
from ptre.tensorflow import print_counter_summary
from ptre.tensorflow import print_counter_summary_epoch
from ptre.tensorflow import push_model
from ptre.tensorflow import rank, size
from ptre.tensorflow import resource_modelaverage
from ptre.tensorflow import resource_push_tensor
from ptre.tensorflow import resource_update_pull_variable
#from ptre.tensorflow import create_pull_job
from ptre.tensorflow.keras import callbacks
from ptre.tensorflow.keras.optimizers.allreduce_optimizer import PtreAllreduceOptimizer
from ptre.tensorflow.keras.optimizers.pull_optimizer import PtreModelaverageOptimizer


from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import training_ops

def create_modelaverage_optimizer(optimizer, name, apply_after_reduce, mode):
  class _PullOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name, config):
      if name is None:
        name = "Custom3%s" % self.__class__.__base__.__name__
      self._name = name
      super(self.__class__, self).__init__(**config)

    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
      """`apply_gradients` using a `DistributionStrategy`."""
      reduced_grads = distribution.extended.batch_reduce_to(
          ds_reduce_util.ReduceOp.SUM, grads_and_vars)
      var_list = [v for _, v in grads_and_vars]
      grads_and_vars = zip(reduced_grads, var_list)

      def apply_grad_to_update_var(var, grad):
        """Apply gradient to variable."""
        if isinstance(var, ops.Tensor):
          raise NotImplementedError("Trying to update a Tensor ", var)

        apply_kwargs = {}
        if isinstance(grad, ops.IndexedSlices):
          if var.constraint is not None:
            raise RuntimeError(
                "Cannot use a constraint function on a sparse variable.")
          if "apply_state" in self._sparse_apply_args:
            apply_kwargs["apply_state"] = apply_state
          return self._resource_apply_sparse_duplicate_indices(
              grad.values, var, grad.indices, **apply_kwargs)

        if "apply_state" in self._dense_apply_args:
          apply_kwargs["apply_state"] = apply_state
        update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
        if var.constraint is not None:
          with ops.control_dependencies([update_op]):
            return var.assign(var.constraint(var))
        else:
          return update_op

      def apply_ma_and_grad(var, grad):
        def _apply_ma(var):
          return resource_modelaverage(var.handle, var.name)
        def _update_pull_var(var):
          return resource_update_pull_variable(var.handle, var.name)
        apply_ma_op = _apply_ma(var)
        with ops.control_dependencies([apply_ma_op]):
          apply_grad_op = apply_grad_to_update_var(var, grad)
          with ops.control_dependencies([apply_grad_op]):
            return _update_pull_var(var)
      def apply_grad_and_ma(var, grad):
        def _apply_ma(var):
          return resource_modelaverage(var.handle, var.name)
        def _update_pull_var(var):
          return resource_update_pull_variable(var.handle, var.name)
        apply_grad_op = apply_grad_to_update_var(var, grad)
        with ops.control_dependencies([apply_grad_op]):
          apply_ma_op = _apply_ma(var)
          with ops.control_dependencies([apply_ma_op]):
            return _update_pull_var(var)

      update_ops = []
      with backend.name_scope(name or self._name):
        for grad, var in grads_and_vars:
          scope_name = ("update" if ops.executing_eagerly_outside_functions() else
                        "update_" + var.op.name)
          # Colocate the update with variables to avoid unnecessary communication
          # delays. See b/136304694.
          with backend.name_scope(
              scope_name), distribution.extended.colocate_vars_with(var):
            update_ops.extend(
                distribution.extended.update(
                    #var, apply_ma_and_grad, args=(grad,), group=False))
                    var, apply_grad_and_ma, args=(grad,), group=False))

        any_symbolic = any(isinstance(i, ops.Operation) or
                           tf_utils.is_symbolic_tensor(i) for i in update_ops)
        if not context.executing_eagerly() or any_symbolic:
          # If the current context is graph mode or any of the update ops are
          # symbolic then the step update should be carried out under a graph
          # context. (eager updates execute immediately)
          with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
            with ops.control_dependencies(update_ops):
              return self._iterations.assign_add(1).op

        return self._iterations.assign_add(1)

  class _Custom3(optimizer_v2.OptimizerV2):
    def __init__(self, name, config):
      if name is None:
        name = "Custom3%s" % self.__class__.__base__.__name__
      self._name = name
      super(self.__class__, self).__init__(**config)

    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
      """`apply_gradients` using a `DistributionStrategy`."""
      reduced_grads = distribution.extended.batch_reduce_to(
          ds_reduce_util.ReduceOp.SUM, grads_and_vars)
      var_list = [v for _, v in grads_and_vars]
      grads_and_vars = zip(reduced_grads, var_list)

      def apply_grad_to_update_var(var, grad):
        """Apply gradient to variable."""
        if isinstance(var, ops.Tensor):
          raise NotImplementedError("Trying to update a Tensor ", var)

        apply_kwargs = {}
        if isinstance(grad, ops.IndexedSlices):
          if var.constraint is not None:
            raise RuntimeError(
                "Cannot use a constraint function on a sparse variable.")
          if "apply_state" in self._sparse_apply_args:
            apply_kwargs["apply_state"] = apply_state
          return self._resource_apply_sparse_duplicate_indices(
              grad.values, var, grad.indices, **apply_kwargs)

        if "apply_state" in self._dense_apply_args:
          apply_kwargs["apply_state"] = apply_state
        update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
        if var.constraint is not None:
          with ops.control_dependencies([update_op]):
            return var.assign(var.constraint(var))
        else:
          return update_op

      def apply_ma_and_grad(var, grad):
        def _apply_ma(var):
          return resource_modelaverage(var.handle, var.name)
        def _push_tensor(var):
          return resource_push_tensor(var.handle, var.name)

        #if var.name.startswith("bn_"):
        if False:
          return apply_grad_to_update_var(var, grad)
        else:
          apply_ma_op = _apply_ma(var)
          #return apply_grad_op # For debugging
          with ops.control_dependencies([apply_ma_op]):
            apply_grad_op = apply_grad_to_update_var(var, grad)
            with ops.control_dependencies([apply_grad_op]):
              return _push_tensor(var)

      update_ops = []
      with backend.name_scope(name or self._name):
        for grad, var in grads_and_vars:
          scope_name = ("update" if ops.executing_eagerly_outside_functions() else
                        "update_" + var.op.name)
          # Colocate the update with variables to avoid unnecessary communication
          # delays. See b/136304694.
          with backend.name_scope(
              scope_name), distribution.extended.colocate_vars_with(var):
            update_ops.extend(
                distribution.extended.update(
                    var, apply_ma_and_grad, args=(grad,), group=False))

        any_symbolic = any(isinstance(i, ops.Operation) or
                           tf_utils.is_symbolic_tensor(i) for i in update_ops)
        if not context.executing_eagerly() or any_symbolic:
          # If the current context is graph mode or any of the update ops are
          # symbolic then the step update should be carried out under a graph
          # context. (eager updates execute immediately)
          with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
            with ops.control_dependencies(update_ops):
              return self._iterations.assign_add(1).op

        return self._iterations.assign_add(1)

  class _Custom2(optimizer_v2.OptimizerV2):
    def __init__(self, name, config):
      if name is None:
        name = "Custom2%s" % self.__class__.__base__.__name__
      self._name = name
      self._remotes = {}
      super(self.__class__, self).__init__(**config)

    def init_remotes(self, var_list):
      for var in var_list:
        with ops.device("/cpu:0"):
          remote = variables.Variable(initial_value=var,
                                      #name="remote_{}".format(var.name),
                                      dtype=var.dtype, shape=var.shape)
          self._remotes[var.name] = remote

    def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
      """`apply_gradients` using a `DistributionStrategy`."""
      reduced_grads = distribution.extended.batch_reduce_to(
          ds_reduce_util.ReduceOp.SUM, grads_and_vars)
      var_list = [v for _, v in grads_and_vars]
      grads_and_vars = zip(reduced_grads, var_list)

      def apply_grad_to_update_var(var, grad):
        """Apply gradient to variable."""
        if isinstance(var, ops.Tensor):
          raise NotImplementedError("Trying to update a Tensor ", var)

        apply_kwargs = {}
        if isinstance(grad, ops.IndexedSlices):
          if var.constraint is not None:
            raise RuntimeError(
                "Cannot use a constraint function on a sparse variable.")
          if "apply_state" in self._sparse_apply_args:
            apply_kwargs["apply_state"] = apply_state
          return self._resource_apply_sparse_duplicate_indices(
              grad.values, var, grad.indices, **apply_kwargs)

        if "apply_state" in self._dense_apply_args:
          apply_kwargs["apply_state"] = apply_state
        update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
        if var.constraint is not None:
          with ops.control_dependencies([update_op]):
            return var.assign(var.constraint(var))
        else:
          return update_op

      def apply_grad_and_ma(var, grad):
        def _apply_ma(var):
          #TODO: ptre.resource_apply_modelaverage(var.handle)
          #remote = get_incoming(var.name)
          #remote = self._remotes[var.name]
          #return training_ops.resource_apply_model_average(var.handle, remote)
          return resource_modelaverage(var.handle, var.name)

        def _push_tensor(var):
          return resource_push_tensor(var.handle, var.name)

        #if var.name.startswith("bn_"):
        if False:
          return apply_grad_to_update_var(var, grad)
        else:
          apply_grad_op = apply_grad_to_update_var(var, grad)
          #return apply_grad_op # For debugging
          with ops.control_dependencies([apply_grad_op]):
            #with ops.device("/cpu:0"):
            apply_ma_op = _apply_ma(var)
            with ops.control_dependencies([apply_ma_op]):
              return _push_tensor(var)

        #with ops.control_dependencies([apply_grad_op]):
        #  avg_flag = is_new_incoming()
        #  return control_flow_ops.cond(avg_flag, lambda: _apply_ma(var), lambda: apply_grad_op)

      def apply_ma_to_update_var(var):
        return resource_modelaverage(var.handle, var.name)

      update_ops = []
      with backend.name_scope(name or self._name):
        for grad, var in grads_and_vars:
          scope_name = ("update" if ops.executing_eagerly_outside_functions() else
                        "update_" + var.op.name)
          # Colocate the update with variables to avoid unnecessary communication
          # delays. See b/136304694.
          ### ORIGINAL
          #with backend.name_scope(
          #    scope_name), distribution.extended.colocate_vars_with(var):
          #  update_ops.extend(
          #      distribution.extended.update(
          #          var, apply_grad_to_update_var, args=(grad,), group=False))
          ### V2
          with backend.name_scope(
              scope_name), distribution.extended.colocate_vars_with(var):
            update_ops.extend(
                distribution.extended.update(
                    var, apply_grad_and_ma, args=(grad,), group=False))

        any_symbolic = any(isinstance(i, ops.Operation) or
                           tf_utils.is_symbolic_tensor(i) for i in update_ops)
        if not context.executing_eagerly() or any_symbolic:
          # If the current context is graph mode or any of the update ops are
          # symbolic then the step update should be carried out under a graph
          # context. (eager updates execute immediately)
          with ops._get_graph_from_inputs(update_ops).as_default():  # pylint: disable=protected-access
            with ops.control_dependencies(update_ops):
              return self._iterations.assign_add(1).op

        return self._iterations.assign_add(1)


  class _ModelaverageOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name, config):
      if name is None:
        name = "Modelaverage%s" % self.__class__.__base__.__name__
      self._name = name
      self._is_new_incoming = None
      super(self.__class__, self).__init__(**config)

    def apply_gradients(self, grads_and_vars, name=None):
      self._is_new_incoming = is_new_incoming()
      return super(self.__class__, self).apply_gradients(grads_and_vars, name)

    def _resource_apply_dense(self, grad, var, apply_state=None):
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = ((apply_state or {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      def _apply_ma():
        remote = get_incoming(var.name)
        if self._momentum:
          momentum_var = self.get_slot(var, "momentum")
          return training_ops.resource_apply_keras_momentum_modelaverage(
              var.handle,
              momentum_var.handle,
              coefficients["lr_t"],
              grad,
              coefficients["momentum"],
              remote,
              use_locking=self._use_locking,
              use_nesterov=self.nesterov)
        else:
          return training_ops.resource_apply_gradient_descent_modelaverage(
              var.handle, coefficients["lr_t"], grad, remote,
              use_locking=self._use_locking)

      def _apply():
        if self._momentum:
          momentum_var = self.get_slot(var, "momentum")
          return training_ops.resource_apply_keras_momentum(
              var.handle,
              momentum_var.handle,
              coefficients["lr_t"],
              grad,
              coefficients["momentum"],
              use_locking=self._use_locking,
              use_nesterov=self.nesterov)
        else:
          return training_ops.resource_apply_gradient_descent(
              var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

      apply_op = tf.cond(self._is_new_incoming, _apply_ma, _apply)
      return apply_op

    @classmethod
    def from_config(cls, cfg):
      return cls(name, cfg)

  # We dynamically create a new class that inherits from the optimizer that was passed in.
  # The goal is to override get_gradients() method with an allreduce implementation.
  # This class will have the same name as the optimizer it's wrapping, so that the saved
  # model could be easily restored without Horovod.
  if mode == "push":
    if apply_after_reduce:
      cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_Custom3.__dict__))
    else:
      cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_Custom2.__dict__))
  elif mode == "pull":
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
            dict(_PullOptimizer.__dict__))
  else:
    print("Unknown mode:" + mode)

  return cls(name, optimizer.get_config())

def ModelaverageOptimizer(optimizer, name=None, apply_after_reduce=True,
                          mode="pull"):
  return create_modelaverage_optimizer(optimizer, name, apply_after_reduce,
      mode)

def AllreduceOptimizer(optimizer, name=None):
  return PtreAllreduceOptimizer(optimizer, name)

def PullOptimizer(optimizer, name=None):
  return PtreModelaverageOptimizer(optimizer, name)
#
#def modelaverage(tensor):
