from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ptre.tensorflow as ptre

from tensorflow.python.keras.callbacks import Callback

class InitTrainableVariablesCallback(Callback):
  def __init__(self):
    super(InitTrainableVariablesCallback, self).__init__()
    self._step = 0

  def on_train_begin(self, logs={}):
    ptre.register_variables(self.model.trainable_variables)

  def on_batch_begin(self, batch, logs={}):
    ptre.set_local_step(self._step)

  def on_batch_end(self, batch, logs={}):
    ptre.count_step()
    if self._step == 0:
      ptre.init_num_rcv_tensors()
    self._step = self._step + 1

class BroadcastModelCallback(Callback):
  def __init__(self, root_rank):
    super(BroadcastModelCallback, self).__init__()
    self._broadcast_done = False
    self._root_rank = root_rank

  def on_batch_end(self, batch, logs=None):
    if self._broadcast_done:
      return
    elem_before = self.model.variables[0].numpy()[0][0][0][0]
    #self.model.variables
    #self.model.optimizer.variables()
    ptre.broadcast_variables(self.model.variables, self._root_rank)
    ptre.synchronization_barrier()
    elem_after = self.model.variables[0].numpy()[0][0][0][0]
    print("\n[DEBUG] BroadcastModelCallback"
          "\nBefore: {}"
          "\nAfter : {}".format(elem_before, elem_after))
    self._broadcast_done = True


class PushModelCallback(Callback):
  def __init__(self, period=10):
    super(PushModelCallback, self).__init__()
    self._step = 0
    self._period = period

  def on_batch_begin(self, batch, logs=None):
    if (self._step + 1) % self._period == 0:
      ptre.set_push()
    else:
      ptre.unset_push()

  def on_batch_end(self, batch, logs=None):
    self._step = self._step + 1

class PrintVariablesCallback(Callback):
  def __init__(self, verbose=True):
    super(PrintVariablesCallback, self).__init__()
    self._verbose = verbose

  def on_train_begin(self, batch, logs=None):
    if self._verbose:
      non_tvars = self.model.non_trainable_variables
      names = [ v.name for v in non_tvars ]
      print(names)

class PrintLocalRemoteCallback(Callback):
  def __init__(self, verbose=True):
    super(PrintLocalRemoteCallback, self).__init__()
    self._verbose = verbose

  def on_batch_end(self, batch, logs=None):
    if self._verbose:
      var_list = self.model.variables
      local = var_list[0].numpy()[0][0][0][0]
      local1 = var_list[0].numpy()[0][0][0][1]
      var_name = var_list[0].name
      remote = ptre.get_incoming(var_name)
      remote0 = remote.numpy()[0][0][0][0]
      remote1 = remote.numpy()[0][0][0][1]
      ret = ptre.is_new_incoming()
      print("    ", var_name, ", local:", local, local1, " / remote:", remote0, remote1, " / is_new_incoming:", ret)

class NumPushScheduleCallback(Callback):
  def __init__(self, schedule):
    super(NumPushScheduleCallback, self).__init__()
    self._schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    ptre.set_num_push(self._schedule(epoch))
