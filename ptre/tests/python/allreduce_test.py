from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import ptre.tensorflow.keras as ptre

#tf.compat.v1.disable_eager_execution()

def run():
  num_procs = ptre.size()
  myid = ptre.rank()
  my_val = 0.1 * (myid + 1)

  with tf.device('/device:gpu:0'):
    grad = tf.Variable([[my_val, my_val], [my_val, my_val]])

  print(grad)
  print(grad.name)
  print(grad.device)

  with tf.device('/device:cpu:0'):
    new_grad = ptre.allreduce(grad)

  print(new_grad)
  #print(new_grad.name)
  print(new_grad.device)


if __name__ == '__main__':
  ptre.init()
  run()
  ptre.finalize()
