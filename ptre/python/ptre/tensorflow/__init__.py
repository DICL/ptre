from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ptre.tensorflow.ptre_ops import _async_comm as async_comm
from ptre.tensorflow.ptre_ops import _await_comm as await_comm
from ptre.tensorflow.ptre_ops import allreduce
from ptre.tensorflow.ptre_ops import _get_remote_variable
from ptre.tensorflow.ptre_ops import barrier
from ptre.tensorflow.ptre_ops import broadcast
from ptre.tensorflow.ptre_ops import connect_qps
from ptre.tensorflow.ptre_ops import count_step
from ptre.tensorflow.ptre_ops import create_pull_job
from ptre.tensorflow.ptre_ops import finalize
from ptre.tensorflow.ptre_ops import init as _init
from ptre.tensorflow.ptre_ops import init_global_consensus
from ptre.tensorflow.ptre_ops import init_rdma_grpc_service
from ptre.tensorflow.ptre_ops import init_remote_mr
from ptre.tensorflow.ptre_ops import init_step_one
from ptre.tensorflow.ptre_ops import is_new_incoming
from ptre.tensorflow.ptre_ops import print_counter_summary
from ptre.tensorflow.ptre_ops import print_counter_summary_epoch
from ptre.tensorflow.ptre_ops import print_recv_count
from ptre.tensorflow.ptre_ops import push_model
from ptre.tensorflow.ptre_ops import rank
from ptre.tensorflow.ptre_ops import register_variables
from ptre.tensorflow.ptre_ops import resource_modelaverage
from ptre.tensorflow.ptre_ops import resource_push_tensor
from ptre.tensorflow.ptre_ops import resource_update_pull_variable
from ptre.tensorflow.ptre_ops import resource_remote_variable
from ptre.tensorflow.ptre_ops import resource_publish_variable
from ptre.tensorflow.ptre_ops import set_broadcast_not_done
from ptre.tensorflow.ptre_ops import set_local_step
from ptre.tensorflow.ptre_ops import set_push
from ptre.tensorflow.ptre_ops import size
from ptre.tensorflow.ptre_ops import synchronization_barrier
from ptre.tensorflow.ptre_ops import unset_push
from ptre.tensorflow.ptre_ops import modelaverage
from ptre.tensorflow.ptre_ops import publish

from ptre.tensorflow.util import _make_subgraph

import tensorflow as tf

import argparse

def get_incoming(var_name):
  return _get_remote_variable(var_name)

def _make_broadcast_group_fn():
  def broadcast_group(variables, root_rank):
    for var in variables:
      var.assign(broadcast(var, root_rank, None))

  return _make_subgraph(broadcast_group)

def broadcast_variables(variables, root_rank):
  broadcast_group = _make_broadcast_group_fn()
  return broadcast_group(variables, root_rank)

def init():
  parser = argparse.ArgumentParser(description='ptre run arguments',
      formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('-hostfile', required=True, type=str)
  parser.add_argument('-np', required=True, type=int)
  parser.add_argument('-rank', required=True, type=int)
  parser.add_argument('-num_push', default=1, type=int)
  args = parser.parse_args()
  _init(args.np, args.rank, args.hostfile, selection_strategy=0,
      num_push=args.num_push)
