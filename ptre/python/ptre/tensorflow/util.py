import tensorflow as tf
from tensorflow.python.eager import context

def _make_subgraph(f):
  if hasattr(tf, 'function'):
    # TensorFlow 1.14.0+
    return tf.function(f)
  else:
    return tf.contrib.eager.defun(f)

def _executing_eagerly():
  return context.executing_eagerly()
