from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptre.tensorflow import modelaverage
from ptre.tensorflow import publish

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

class _PullOptimizer(optimizer_v2.OptimizerV2):
  def __init__(self, name, config):
    if name is None:
      name = "PullOptimizer%s" % self.__class__.__base__.__name__
    self._name = name
    super(self.__class__, self).__init__(**config)

  #def get_updates(self, loss, params):
  #  grads = self.get_gradients(loss, params)
  #  grads_and_vars = list(zip(grads, params))
  #  self._assert_valid_dtypes([
  #      v for g, v in grads_and_vars
  #      if g is not None and v.dtype != dtypes.resource
  #  ])
  #  return [self.apply_gradients(grads_and_vars)]

  #def apply_gradients(self, grads_and_vars, name=None):
  #  """Wrap `apply_gradients` to apply gradient on averaged variable"""
  #  avg_ops = []
  #  with backend.name_scope(name or self._name):
  #    for grad, var in grads_and_vars:
  #      scope_name = (
  #          "modelaverage" if ops.executing_eagerly_outside_functions() else
  #          "modelaverage_" + var.op.name)
  #      with ops.control_dependencies([grad]), backend.name_scope(scope_name):
  #        avg_var = self._modelaverage(var)
  #        print("DEBUG: ", var.name, avg_var.name, scope_name)
  #      avg_ops.append(avg_var)

  #  # Apply gradients to newly averaged weights
  #  gradients, variables = list(zip(*grads_and_vars))
  #  grads_and_new_vars = zip(gradients, avg_ops)
  #  return (super(self.__class__, self)
  #                .apply_gradients(grads_and_new_vars, name))

  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    """Wrap `_distributed_apply` to publish variable after update."""
    #reduced_grads = distribution.extended.batch_reduce_to(
    #    ds_reduce_util.ReduceOp.SUM, grads_and_vars)
    #var_list = [v for _, v in grads_and_vars]
    #grads_and_vars = zip(reduced_grads, var_list)

    def _modelaverage(var):
      return var.assign(modelaverage(var))

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
      """Average var -> Apply grad -> publish var."""
      # Average var
      modelaverage_scope_name = (
          "modelaverage" if ops.executing_eagerly_outside_functions() else
          "modelaverage_" + var.op.name)
      #with (ops.control_dependencies([grad]),
      #      backend.name_scope(modelaverage_scope_name)):
      with backend.name_scope(modelaverage_scope_name):
        avg_op = _modelaverage(var)

      # Apply grad
      with ops.control_dependencies([avg_op]):
        apply_op = _apply_grad_to_update_var(var, grad)

      # Publish var
      # TODO: check if this op is executed
      publish_scope_name = (
          "publish" if ops.executing_eagerly_outside_functions() else
          "publish_" + var.op.name)
      with ops.control_dependencies(
          [apply_op]), backend.name_scope(publish_scope_name):
        return publish(var)

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
