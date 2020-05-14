from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptre.tensorflow.ptre_ops import _get_remote_variable
from ptre.tensorflow.ptre_ops import connect_qps
from ptre.tensorflow.ptre_ops import count_step
from ptre.tensorflow.ptre_ops import enqueue_push
from ptre.tensorflow.ptre_ops import finalize
from ptre.tensorflow.ptre_ops import init
from ptre.tensorflow.ptre_ops import init_global_consensus
from ptre.tensorflow.ptre_ops import init_rdma_grpc_service
from ptre.tensorflow.ptre_ops import init_remote_mr
from ptre.tensorflow.ptre_ops import init_step_one
from ptre.tensorflow.ptre_ops import is_new_incoming
from ptre.tensorflow.ptre_ops import mark_no_new
from ptre.tensorflow.ptre_ops import push_model
from ptre.tensorflow.ptre_ops import rank
from ptre.tensorflow.ptre_ops import resource_modelaverage
from ptre.tensorflow.ptre_ops import resource_push_tensor
from ptre.tensorflow.ptre_ops import set_push
from ptre.tensorflow.ptre_ops import size
from ptre.tensorflow.ptre_ops import unset_push
from ptre.tensorflow.ptre_ops import broadcast_model
from ptre.tensorflow.ptre_ops import set_broadcast_not_done
from ptre.tensorflow.ptre_ops import set_local_step
from ptre.tensorflow.ptre_ops import synchronization_barrier
from ptre.tensorflow.ptre_ops import init_num_rcv_tensors

from ptre.tensorflow.util import _make_subgraph

import tensorflow as tf

def get_incoming(var_name):
  return _get_remote_variable(var_name)

def _make_broadcast_group_fn():
  def broadcast_group(variables, root_rank):
    for var in variables:
      var.assign(broadcast(var, root_rank))

  return _make_subgraph(broadcast_group)


def broadcast_variables(variables, root_rank):
  broadcast_group = _make_broadcast_group_fn()
  return broadcast_group(variables, root_rank)
