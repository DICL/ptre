from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptre.tensorflow import resource_remote_variable
from ptre.tensorflow import resource_publish_variable

from tensorflow.python.keras.optimizer_v2 import optimizer_v2

def create_modelaverage_optimizer(optimizer, name):
  class _PullOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name, config):
      if name is None:
        name = "PullOptimizer%s" % self.__class__.__base__.__name__
      self._name = name
      super(self.__class__, self).__init__(**config)

    def _remote_variables(variables):
      rvars = []
      num_aggs = []
      for var in variables:
        rvar, num_agg = resource_remote_variable(var.handle, var.name)
        rvars.append(rvar)
        num_aggs.append(num_agg)
      return rvars, num_aggs

    def _model_average(locals_and_remotes):
      def average_tensor(var, rvar, num_agg):
        return tf.cond(num_agg > 1, 
              var.assign((var + rvar) / (num_agg + 1)),
              var)

      avg_vars = []
      for var, rvar, num_agg in locals_and_remotes:
        avg_vars.append(average_tensor(var, rvar, num_agg))
      return avg_vars

    def _publish(var):
      return resource_publish_variable(var.handle, var.name)

    def apply_gradients(self, grads_and_vars, name=None):
      gradients, variables = list(zip(*grads_and_vars))
      rvars, num_aggs = _remote_variables(variables)
      locals_and_remotes = zip(variables, rvars, num_aggs)
      avg_vars = _model_average(locals_and_remotes)
      new_grads_and_vars = zip(gradients, avg_vars)
      apply_ops = (super(self.__class__, self)
                    .apply_gradients(new_grads_and_vars, name))
      publish_ops = []
      apply_and_vars = zip(apply_ops, avg_vars)
      for apply_op, var in apply_and_vars:
        with ops.control_dependencies([apply_op]):
          publish_ops.append(_publish(var))
      return publish_ops
      

  # create_modelaverage_optimizer
  cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_PullOptimizer.__dict__))
  return cls(name, optimizer.get_config())

def ModelaverageOptimizerV2(optimizer, name=None):
  return create_modelaverage_optimizer(optimizer, name)
