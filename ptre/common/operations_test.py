from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes

from tensorflow.python.framework import load_library

PTRE_LIB_PATH = '/home/wkim/ptre/build/ptre/tensorflow/kernels/libptre_ops.so'
PTRE_LIB = load_library.load_op_library(PTRE_LIB_PATH)
PTRE_CDLL = ctypes.CDLL(PTRE_LIB_PATH, mode=ctypes.RTLD_GLOBAL)

print(PTRE_LIB.__dict__)
print(PTRE_CDLL.ptre_size())
