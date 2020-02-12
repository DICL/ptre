#ifndef PTRE_COMMUNICATION_COMMUNICATOR_H_
#define PTRE_COMMUNICATION_COMMUNICATOR_H_

namespace ptre {

class Communicator {
 public:
  int AttemptToPushModel(int target);
  int AckPushDone


  door->close();
  if (tensor->status == NO_INCOMING) {
    return 0;
  } else if (tensor->status == BEING_RECEIVED) {
    wait_for_incoming();
    reduce();
  } else if (tensor->status == RECEIVE_DONE) {
    reduce();
  }


  engine::ProcessOneStep() {
    AttemptToPushModel();
    for (auto layer : model) {
      PushTensor(layer);
    }
    AckPushModelDone();
    grads = ComputeGradients(model);
    NoMoreReceiveModel();
    for (auto layer, grad : zip(model, grads)) {
      ReduceTensor(layer);
      ApplyGradient(layer, grad);
    }
    OpenReceiveModel(layer);
  }

  int PushTensor(int target, Tensor* tensor) {

  }




  server::AttemptToPushModel() {
    int ret = 0;
    lock(model);
    if (model.door->is_open()) {
      ret = 1;
      model_rcv_ing_cnt++;
      for (auto tensor : tensors) {
        lock_guard(tensor);
        tensor->rcv_ing_cnt++;
      }
    }
    unlock(model);
    return ret;
  }

  server::AckPushTensorDone() {
    tensor->lock();
    tensor->done_counter++;
    tensor->unlock();
  }

};

}  // namespace ptre
#endif  // PTRE_COMMUNICATION_COMMUNICATOR_H_
