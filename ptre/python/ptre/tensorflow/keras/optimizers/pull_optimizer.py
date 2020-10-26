from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ptre.tensorflow as ptre
import tensorflow as tf

from ptre.tensorflow import modelaverage
from ptre.tensorflow import publish

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_utils

class _PullOptimizer(optimizer_v2.OptimizerV2):
  def __init__(self, name, config):
    if name is None:
      name = "PullOptimizer%s" % self.__class__.__base__.__name__
    self._name = name
    super(self.__class__, self).__init__(**config)

  def _async_comm(self, var):
    if ptre.size() > 0:
      async_op = ptre.async_comm(var)
      with tf.name_scope(self._name + "_AsyncComm"):
        return async_op
    else:
      return tf.no_op

  def _await_comm(self, var):
    if ptre.size() > 0:
      with tf.name_scope(self._name + "_AwaitComm"):
        fetch_op = ptre.await_comm(var)
        return fetch_op 
    else:
      return tf.no_op

  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    """Wrap `_distributed_apply` to publish variable after update."""
    def _apply_grad_to_update_var(var, grad):
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

    def apply_grad_to_update_var(var, grad):
      """Fetch avg var -> Apply grad -> Async avg var."""
      if True:
        # Wait async allreduce var issued from the previous step
        await_op = self._await_comm(var)
        # Apply grad
        with ops.control_dependencies([await_op]):
          apply_op = _apply_grad_to_update_var(var, grad)
          # Start Async Comm var
          with ops.control_dependencies([apply_op]):
            enqueue_op = self._async_comm(var)
            return enqueue_op
      else:
        #print("Sync: ", var.name)
        # Apply grad
        apply_op = _apply_grad_to_update_var(var, grad)
        # Average var
        with ops.control_dependencies([apply_op]):
          update_op = self._allreduce(var)
        return update_op

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
                  var, apply_grad_to_update_var, args=(grad,), group=False))

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

def create_modelaverage_optimizer(optimizer, name):
  # create_modelaverage_optimizer
  cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_PullOptimizer.__dict__))
  return cls(name, optimizer.get_config())

def PtreModelaverageOptimizer(optimizer, name=None):
  return create_modelaverage_optimizer(optimizer, name)
