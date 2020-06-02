from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import ctypes
import time

from tensorflow.python.framework import load_library

PTRE_LIB_PATH = '/home/wkim/ptre/build/ptre/kernels/libptre_ops.so'
#PTRE_LIB_PATH = '/home/wkim/.local/lib/python2.7/site-packages/ptre/tensorflow/libptre_ops.so'
PTRE_LIB = load_library.load_op_library(PTRE_LIB_PATH)
PTRE_CDLL = ctypes.CDLL(PTRE_LIB_PATH, mode=ctypes.RTLD_GLOBAL)

def _normalize_name(name):
  """Normalizes operation name to TensorFlow rules."""
  return re.sub('[^a-zA-Z0-9_]', '_', name)

def init(comm_size, comm_rank, grpc_hosts_file=None, comm=None,
         selection_strategy=0, num_push=1):
  if grpc_hosts_file is None:
    grpc_hosts_file = "/home/wkim/experiments/grpc_hosts"
  PTRE_CDLL.ptre_init(comm_size, comm_rank, grpc_hosts_file, selection_strategy, num_push)

def init_rdma_grpc_service():
  PTRE_CDLL.ptre_init_rdma_grpc_service()

def finalize(wait_sec=0):
  barrier(1)
  PTRE_CDLL.ptre_finalize(wait_sec)

def size():
  return PTRE_CDLL.ptre_size()

def rank():
  return PTRE_CDLL.ptre_rank()

def set_push():
  PTRE_CDLL.ptre_set_push()

def unset_push():
  PTRE_CDLL.ptre_unset_push()

def is_new_incoming():
  return PTRE_CDLL.ptre_is_new_incoming()

def resource_modelaverage(var, var_name):
  return PTRE_LIB.resource_modelaverage(var, 'float32', var_name=var_name)

def resource_push_tensor(var, var_name):
  return PTRE_LIB.resource_push_tensor(var, 'float32', var_name=var_name)

def _get_remote_variable(var_name):
  return PTRE_LIB.get_remote_variable(var_name=var_name)

def register_variables(variables):
  names = [ v.name for v in variables ]
  PTRE_LIB.register_variables(variables, names=names)

def init_global_consensus(var_list):
  names = [ v.name for v in var_list ]
  PTRE_LIB.init_global_consensus(var_list, names=names)

def init_remote_mr(var_list):
  names = [ v.name for v in var_list ]
  PTRE_LIB.init_remote_mr(names=names)

def connect_qps():
  PTRE_LIB.connect_qps()

def init_step_one(var_list):
  names = [ v.name for v in var_list ]
  PTRE_LIB.init_step_one(var_list, names=names)

def push_model(var_list):
  names = [ v.name for v in var_list ]
  PTRE_LIB.push_model(var_list, names=names)
#def _modelaverage(tensor, name=None

def set_broadcast_not_done():
  PTRE_CDLL.ptre_set_broadcast_not_done()

def synchronization_barrier():
  PTRE_CDLL.ptre_synchronization_barrier()

def count_step():
  PTRE_CDLL.ptre_count_step()

def set_local_step(step):
  PTRE_CDLL.ptre_set_local_step(step)

def broadcast(tensor, root_rank, name):
  name = 'PtreBroadcast_%s' % _normalize_name(tensor.name)
  #name = 'PtreBroadcast_%s' % tensor.name
  return PTRE_LIB.broadcast(tensor, name=name, root_rank=root_rank)

def barrier(wait_sec=0):
  PTRE_CDLL.ptre_barrier()
  time.sleep(wait_sec)

def print_recv_count():
  PTRE_CDLL.ptre_print_recv_count()

def print_counter_summary():
  PTRE_CDLL.ptre_print_counter_summary()

def print_counter_summary_epoch():
  PTRE_CDLL.ptre_print_counter_summary_epoch()
