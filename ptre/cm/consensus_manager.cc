#include "ptre/cm/consensus_manager.h"

#include <iostream>
#include <random>
#include <set>
#include <chrono>
#include <stdlib.h>

namespace ptre {

ConsensusManager::~ConsensusManager() {
  if (rdma_manager_ != nullptr) {
    delete rdma_manager_;
  }
  if (peer_selector_ != nullptr) {
    delete peer_selector_;
  }
}
void ConsensusManager::InitGlobalConsensus(std::vector<const Tensor*>& vars) {
  int num_vars = vars.size();
  for (int i = 0; i < num_vars; i++) {
    const Tensor* var = vars[i];
    Tensor* tensor = new Tensor(var->dtype(), var->shape());
    std::copy(var->tensor_data().begin(), var->tensor_data().end(),
              const_cast<char*>(tensor->tensor_data().begin()));
    global_consensus_.push_back(tensor);
    //send_tensors_.push_back(new Tensor(var->dtype(), var->shape()));
  }
  num_vars_ = num_vars;
  is_initialized_ = true;
}

void ConsensusManager::InitPeerSelector(int strategy) {
  PeerSelectorFactory::NewPeerSelector(ptre_size_, ptre_rank_,
                                       SelectionStrategy(strategy),
                                       peer_selector_);
  //for (int i = 0; i < ptre_size_ * 2; i++) {
  //  std::cout << peer_selector_->get_peer() << std::endl;
  //}
  //usleep(5000000);
  //exit(EXIT_FAILURE);
}

void ConsensusManager::InitBufTensor(const std::string& name,
                                     const Tensor& tensor) {
  Tensor* recv_tensor = new Tensor(tensor.dtype(), tensor.shape());
  std::copy(tensor.tensor_data().begin(), tensor.tensor_data().end(),
            const_cast<char*>(recv_tensor->tensor_data().begin()));
  recv_tensors_.emplace(name, recv_tensor);
  global_consensus_.push_back(recv_tensor);

  Tensor* send_tensor = new Tensor(tensor.dtype(), tensor.shape());
  send_tensors_.emplace(name, send_tensor);
  send_tensors_list_.push_back(send_tensor);

  rdma_manager_->InitTensorMR(0, name, recv_tensor, send_tensor);
  //for (int i = 0; i < ptre_size_; i++) {
  //  if (i == ptre_rank_) {
  //    continue;
  //  }
  //  rdma_manager_->InitTensorMR(i, name, recv_tensor, send_tensor);
  //}
}

void ConsensusManager::InitBufParam() {
  is_new_incoming_ = new bool(false);
  rdma_manager_->InitParamMR(is_new_incoming_, &flag_to_send_);
}

void ConsensusManager::SetRdmaManager(RdmaManager* rdma_manager) {
  rdma_manager_ = rdma_manager;
}

void ConsensusManager::EnqueuePushList(std::vector<const Tensor*>& vars) {
  //std::lock_guard<std::mutex> guard(mu_);
  int num_vars = vars.size();
  for (int i = 0; i < num_vars; i++) {
    const Tensor* var = vars[i];
    Tensor* tensor = new Tensor(var->dtype(), var->shape());
    std::copy(var->tensor_data().begin(), var->tensor_data().end(),
              const_cast<char*>(tensor->tensor_data().begin()));
    for_push_.push_back(tensor);
  }
  ready_to_push_ = true;
}


void ConsensusManager::CopyTensorSend(const std::string& name,
                                      const Tensor& tensor) {
  Tensor* send_tensor = send_tensors_[name];
  const char* send_data = send_tensor->tensor_data().data();
  auto tensor_strpc = tensor.tensor_data();
  std::copy(tensor_strpc.begin(), tensor_strpc.end(),
            const_cast<char*>(send_data));
}

void ConsensusManager::PushModel(int dst_rank) {
  bool can_push = rdma_manager_->AttemptPush(dst_rank);
  if (!can_push) {
    return;
  }

  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_manager_->PushTensor(dst_rank, name, *t);  // num_comps + 1
  }

  rdma_manager_->AckPushDone(dst_rank);
}

void ConsensusManager::PushTensors(int dst_rank) {
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_manager_->RdmaWriteTensor(dst_rank, name, *t, false);
  }
  flag_to_send_ = true;
  rdma_manager_->RdmaWriteIncomingFlag(dst_rank, &flag_to_send_);
  int num_comps = send_tensors_.size() + 1;
  rdma_manager_->Poll(num_comps);
}

void ConsensusManager::PushTensors2(int dst_rank) {
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_manager_->RdmaWriteTensor(dst_rank, name, *t, true);
  }
  //flag_to_send_ = true;
  //rdma_manager_->RdmaWriteIncomingFlag(dst_rank, &flag_to_send_);
  int num_comps = send_tensors_.size() + 1;
  rdma_manager_->Poll(num_comps);
}

void ConsensusManager::TcpPushTensors(int dst_rank) {
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    //tcp_manager_->TcpSendTensor(dst_rank, name, *t);
  }
  //tcp_manager_->TcpSendIncomingFlag(dst_rank, true);
}

bool ConsensusManager::CanReceive(int src_rank) {
  std::lock_guard<std::mutex> guard(rcv_mu_);
  if (rcv_open_) {
    auto ret = rcv_status_.emplace(src_rank, RECV_IN_PROGRESS);
    if (!ret.second) {
      return false;
    }
    rcv_ing_cnt_++;
    return true;
  }
  return false;
}

int ConsensusManager::FinalizeRecv(int src_rank) {
  //std::lock_guard<std::mutex> guard(rcv_mu_);
  rcv_mu_.lock();
  rcv_status_.erase(src_rank);
  rcv_done_cnt_++;
  rcv_mu_.unlock();
  rcv_cv_.notify_all();
  return 0;
}

int ConsensusManager::OpenReceive() {
  // TODO: INIT RECV BUF AND RCV COUNTERS BEFORE OPEN!
  std::lock_guard<std::mutex> guard(rcv_mu_);
  rcv_open_ = true;
  return 0;
}

int ConsensusManager::CloseReceive() {
  // TODO: DO NOT OPEN AGAIN BEFORE REDUCE FINISHES.
  //std::lock_guard<std::mutex> guard(rcv_mu_);
  rcv_mu_.lock();
  rcv_open_ = false;
  rcv_mu_.unlock();
  rcv_cv_.notify_all();
  return 0;
}

bool ConsensusManager::IsReceiveDone() {
  std::lock_guard<std::mutex> guard(rcv_mu_);
  if (!rcv_open_ && rcv_ing_cnt_ == rcv_done_cnt_) {
    return true;
  }
  return false;
}

int ConsensusManager::GetNumIncomingsOrWait() {
  CloseReceive();
  std::unique_lock<std::mutex> lk(rcv_mu_);
  rcv_cv_.wait(lk, [&] {
        return (!rcv_open_ && rcv_ing_cnt_ == rcv_done_cnt_);
      });
  lk.unlock();
  // TODO: Must return zero and don't average if counts don't match.
  return rcv_done_cnt_;
}

int ConsensusManager::GetRandomTarget() {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, ptre_size_ - 1);
  int ret = ptre_rank_;
  while (ret == ptre_rank_) {
    ret = distribution(generator);
  }
  return ret;
}

int ConsensusManager::GetIncNeighbor() {
  int ret = ptre_rank_ + 1;
  if (ret >= ptre_size_) {
    ret = 0;
  }
  return ret;
}

int ConsensusManager::get_peer() {
  peer_selector_->get_peer();
  return 1;
}

int ConsensusManager::get_peers(int num_peer, int* peers) {
  std::set<int> checker;
  int cnt = 0;
  auto start = std::chrono::system_clock::now();
  while (cnt < num_peer) {
    int peer = peer_selector_->get_peer();
    auto ret = checker.emplace(peer);
    if (!ret.second) {
      continue;
    } else {
      peers[cnt] = peer;
      cnt++;
    }
    std::chrono::duration<float> dur = std::chrono::system_clock::now() - start;
    if (dur.count() > 1) {
      break;
    }
  }
  return cnt;
}

const Tensor& ConsensusManager::global_consensus(int index) {
  const Tensor& tensor = *(global_consensus_[index]);
  return tensor;
}

const Tensor& ConsensusManager::global_consensus(const std::string& name) {
  const Tensor& tensor = *(recv_tensors_[name]);
  return tensor;
}

const std::vector<Tensor*>& ConsensusManager::GetGlobalConsensusList() {
  return global_consensus_;
}

const std::vector<Tensor*>& ConsensusManager::GetSendTensorsList() {
  return send_tensors_list_;
}

}  // namespace ptre
