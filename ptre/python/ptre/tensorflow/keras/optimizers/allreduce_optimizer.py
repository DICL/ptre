from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptre.tensorflow import allreduce
from ptre.tensorflow import resource_remote_variable
from ptre.tensorflow import resource_publish_variable

from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
import tensorflow as tf

def create_distributed_optimizer(optimizer, name):
  class _AllreduceOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, name, config):
      if name is None:
        name = "AllreduceOptimizer%s" % self.__class__.__base__.__name__
      self._name = name
      super(self.__class__, self).__init__(**config)

    def get_gradients(self, loss, params):
      gradients = super(self.__class__, self).get_gradients(loss, params)
      return self._allreduce(gradients)

    def _allreduce(self, gradients):
      avg_grads = []
      with tf.name_scope(self._name + "_Allreduce"):
        for grad in gradients:
          with ops.device('/device:cpu:0'):
            #grad_cpu = tf.convert_to_tensor(grad)
            avg_cpu = allreduce(grad)
          avg_grads.append(avg_cpu)
          #print(type(avg_cpu))
          #avg_grads.append(tf.constant(avg_cpu))
      return avg_grads
  # create_distributed_optimizer
  cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_AllreduceOptimizer.__dict__))
  return cls(name, optimizer.get_config())

def PtreAllreduceOptimizer(optimizer, name=None):
  return create_distributed_optimizer(optimizer, name)
