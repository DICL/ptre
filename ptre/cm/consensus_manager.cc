#include "ptre/cm/consensus_manager.h"

#include <iostream>
#include <random>
#include <set>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <algorithm>

#define DDLOG DLOG(INFO) << "RANK:" << ptre_rank_ << " "

namespace ptre {

ConsensusManager::~ConsensusManager() {
  //if (rdma_manager_ != nullptr) {
  //  delete rdma_manager_;
  //}
  if (peer_selector_ != nullptr) {
    delete peer_selector_;
  }
  if (tensor_aggregator_ != nullptr) {
    tensor_aggregator_->Terminate();
  }
}

/// NOT USED at least until 5f1352f07118881c8c5319e341fde8633905b42f
/// See ConsensusManager::InitBufTensor
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

int ConsensusManager::InitGlobalConsensusV2(const std::vector<string>& names,
    const std::vector<const Tensor*>& vars) {

  /// 1. Trainable variables (Recv Buf)   : N * T
  /// 2. Send Buf for trainable variables : N * T
  /// 3. Agg Buf for trainable variables  : N * T   ->  TensorAggregator
  ///   3-1. Agg Buf States               : N * int ->  TensorAggregator

  /// 1. Recv buf
  num_vars_ = vars.size();
  for (int i = 0; i < num_vars_; i++) {
    Tensor* recv_tensor = new Tensor(vars[i]->dtype(), vars[i]->shape());
    global_consensus_.push_back(recv_tensor);
    // Register Buf
    tensorflow::StringPiece strpc = recv_tensor->tensor_data();
    void* buf = (void*) strpc.data();
    size_t length = strpc.size();
    string buf_name(names[i] + "/RecvBuf");
    buf_names_.push_back(buf_name);
    buf_types_.push_back(BUF_TYPE_RECV_BUF);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_RECV_BUF].emplace(names[i], num_bufs_);
    buf_name_to_index_.emplace(buf_name, num_bufs_);
    num_bufs_++;
    //name_to_index_.emplace(names[i], i);
    recv_tensors_.emplace(names[i], recv_tensor);
    tensor_names_.push_back(names[i]);
    // Register MR
    rdma_manager_->RegisterMR(BUF_TYPE_RECV_BUF, names[i], buf, length, true);
  }
  /// 2. Send Buf
  for (int i = 0; i < num_vars_; i++) {
    size_t length = vars[i]->TotalBytes();
    void* buf = malloc(length);
    string buf_name(names[i] + "/SendBuf");
    buf_names_.push_back(buf_name);
    buf_types_.push_back(BUF_TYPE_SEND_BUF);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_SEND_BUF].emplace(names[i], num_bufs_);
    buf_name_to_index_.emplace(buf_name, num_bufs_);
    num_bufs_++;
    rdma_manager_->RegisterMR(BUF_TYPE_SEND_BUF, names[i], buf, length, false);
  }
  /// 3. Agg Buf
  std::vector<Flat> recv_flats;
  for (int i = 0; i < num_vars_; i++) {
    recv_flats.push_back(global_consensus_[i]->flat<float>());
  }
  // TODO: Poll RECV CQ and get IMM DATA in TensorAggregator?
  struct ibv_cq* local_cq = rdma_manager_->local_cq();
  struct ibv_qp* local_qp = rdma_manager_->local_qp();
  tensor_aggregator_ = new TensorAggregator(nullptr, 0,
      rdma_manager_->rdma_env(),
      local_cq, local_qp,
      names, recv_flats);
  for (int i = 0; i < num_vars_; i++) {
    void* buf = (void*) tensor_aggregator_->buf_ptr(names[i]);
    size_t length = tensor_aggregator_->buf_length(names[i]);
    string buf_name(names[i] + "/AggBuf");
    buf_names_.push_back(buf_name);
    buf_types_.push_back(BUF_TYPE_AGG_BUF);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_AGG_BUF].emplace(names[i], num_bufs_);
    buf_name_to_index_.emplace(buf_name, num_bufs_);
    num_bufs_++;
    rdma_manager_->RegisterMR(BUF_TYPE_AGG_BUF, names[i], buf, length, true);
  }
  for (int i = 0; i < num_vars_; i++) {
    void* buf = (void*) tensor_aggregator_->state_ptr(names[i]);
    size_t length = sizeof(uint64_t);
    string buf_name(names[i] + "/AggBufState");
    buf_names_.push_back(buf_name);
    buf_types_.push_back(BUF_TYPE_AGG_BUF_STATE);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_AGG_BUF_STATE]
        .emplace(names[i], num_bufs_);
    buf_name_to_index_.emplace(buf_name, num_bufs_);
    num_bufs_++;
    rdma_manager_->RegisterMR(BUF_TYPE_AGG_BUF_STATE, names[i], buf, length,
        true);
    struct ibv_mr* mr = rdma_manager_->GetMR(BUF_TYPE_AGG_BUF_STATE, names[i]);
    tensor_aggregator_->SetStateMR(names[i], mr);
  }
  /// 4. Push Permit Array
  push_permits_.resize(num_vars);
  for (int i = 0; i < num_vars; i++) {
    int* elem = new int(-1);
    push_permits_[i] = elem;
    rdma_manager_->RegisterMR(BUF_TYPE_PUSH_PERMIT_SRC, names[i], (void*) elem,
        sizeof(int), true);
  }

  /// For Backward-Compatibility
  /// TODO: This logic should be deprecated and replaced by counting logic.
  /// Do the same thing as in InitBufParam()
  if (true) {
    string name("is_new_incoming");
    is_new_incoming_ = new bool(false);
    void* buf = (void*) is_new_incoming_;
    size_t length = sizeof(bool);
    buf_types_.push_back(BUF_TYPE_FLAG_RECV);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_FLAG_RECV].emplace(name, num_bufs_);
    num_bufs_++;
    rdma_manager_->RegisterMR(BUF_TYPE_FLAG_RECV, name, buf, length, true);
    buf = (void*) &flag_to_send_;
    buf_types_.push_back(BUF_TYPE_FLAG_SEND);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_FLAG_SEND].emplace(name, num_bufs_);
    num_bufs_++;
    rdma_manager_->RegisterMR(BUF_TYPE_FLAG_SEND, name, buf, length, false);
  }
}

void ConsensusManager::InitPeerSelector(int strategy, int num_push) {
  PeerSelectorFactory::NewPeerSelector(ptre_size_, ptre_rank_,
                                       SelectionStrategy(strategy),
                                       peer_selector_,
                                       num_push);
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
  tensor_names_.push_back(name);

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
#if 0
  Tensor* send_tensor = send_tensors_[name];
  const char* send_data = send_tensor->tensor_data().data();
  auto tensor_strpc = tensor.tensor_data();
  std::copy(tensor_strpc.begin(), tensor_strpc.end(),
            const_cast<char*>(send_data));
#else
  int idx = buf_type_name_index_map_[BUF_TYPE_SEND_BUF][name];
  char* send_buf = (char*) bufs_[idx];
  auto strpc = tensor.tensor_data();
  std::copy(strpc.begin(), strpc.end(), send_buf);
#endif
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

  rdma_manager_->NotifyPushDone(dst_rank);
}

void ConsensusManager::PushTensors(int dst_rank) {
  for (auto it : buf_type_name_index_map_[BUF_TYPE_SEND_BUF]) {
    rdma_manager_->RdmaWriteBufRemote(dst_rank, BUF_TYPE_SEND_BUF,
        BUF_TYPE_RECV_BUF, it.first, true);
  }
}

void ConsensusManager::PushTensors2(int dst_rank) {
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_manager_->RdmaWriteTensor(dst_rank, name, *t, true);
  }
  //std::cout << "\n[RANK=" << ptre_rank_ << "]: RdmaWriteTensor Done.\n";
  //flag_to_send_ = true;
  //rdma_manager_->RdmaWriteIncomingFlag(dst_rank, &flag_to_send_);
  int num_comps = send_tensors_.size();
  rdma_manager_->Poll(num_comps);
  //std::cout << "\n[RANK=" << ptre_rank_ << "]: Poll Done.\n";
}

void ConsensusManager::PushTensorsV3(int dst_rank) {
  send_mu_.lock();
  send_status_ = SEND_IN_PROGRESS;
  send_mu_.unlock();
#if 0
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_manager_->PushTensorAtomicAddBatch(dst_rank, name, *t);
  }
#elif 0
  //for (auto& name : actual_comm_tensors_) {
  //  if (ptre_rank_ == 0) {
  //    Tensor* t = send_tensors_[name];
  //    std::cout << t->NumElements() << std::endl;
  //  }
  //}
  for (auto& name : actual_comm_tensors_) {
    //DDLOG << name;
    if (name.rfind("block4", 0) == 0) {
      Tensor* t = send_tensors_[name];
      try {
        rdma_manager_->PushTensorAtomicAddBatch(dst_rank, name, *t);
      } catch (std::exception& e) {
        std::cerr << "Exception caught : " << e.what() << std::endl;
      }
    }
  }
#elif 0
  /// Use RdmaAggWriter
  /// NOTE: AggBuf at dst_rank must be initialized.
  /// Init AggBuf when OpenRecv
  for (auto& name : actual_comm_tensors_) {
    rdma_manager_->PushTensorBufferedAggregation(dst_rank, name);
  }
#else
  rdma_manager_->PushTensorBufferedAggregation(dst_rank, actual_comm_tensors_);
#endif
  send_mu_.lock();
  send_status_ = SEND_IDLE;
  send_cv_.notify_all();
  send_mu_.unlock();
}

void ConsensusManager::TcpPushTensors(int dst_rank) {
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    //tcp_manager_->TcpSendTensor(dst_rank, name, *t);
  }
  //tcp_manager_->TcpSendIncomingFlag(dst_rank, true);
}

Tensor* ConsensusManager::send_tensor(int index) {
  LOG(ERROR) << "This function is temporaly deprecated.";
  exit(EXIT_FAILURE);
  return send_tensors_list_[index];
}

Tensor* ConsensusManager::send_tensor(const string& name) {
  LOG(ERROR) << "This function is temporaly deprecated.";
  exit(EXIT_FAILURE);
  return send_tensors_[name];
}

bool ConsensusManager::CanReceive(int src_rank, int src_vstep) {
#if 0
  std::lock_guard<std::mutex> rcv_guard(rcv_mu_);
  if (rcv_open_) {
    rcv_ing_cnt_++;
    for (int i = 0; i < num_vars_; i++) {
      std::lock_guard<std::mutex> guard(var_rcv_mus_);
      if (var_rcv_doors_[i]) {
        permit_manager_->EnqueuePeer(src_rank);
      }
    }
  }
#endif
  if (virtual_step_ > src_vstep) {
    return false;
  }
  std::lock_guard<std::mutex> guard(rcv_mu_);
  if (rcv_open_) {
    if (virtual_step_ < 10) {
      if (rcv_ing_cnt_ >= 1) {
        return false;
      }
    } else {
      if (rcv_ing_cnt_ >= MAX_RECV_THRESHOLD) {
        return false;
      }
    }
    auto ret = rcv_status_.emplace(src_rank, RECV_IN_PROGRESS);
    if (!ret.second) {
      return false;
    }
    // Push available
    rcv_ing_cnt_++;
    rcv_steps_sum_ += src_vstep;
    tensor_aggregator_->EnqueuePeer(src_rank);
    return true;
  }
  return false;
}

int ConsensusManager::FinalizeRecv(int src_rank) {
  //std::lock_guard<std::mutex> guard(rcv_mu_);
  rcv_mu_.lock();
  rcv_status_.erase(src_rank);
  rcv_done_cnt_++;
  //std::cout << "\n[RANK=" << ptre_rank_ << "]: Got Ack from rank=" << src_rank << ", rcv_done_cnt=" << rcv_done_cnt_ << "/" << rcv_ing_cnt_ << ", open=" << rcv_open_ << std::endl;
  rcv_cv_.notify_all();
  rcv_mu_.unlock();
  return 0;
}

int ConsensusManager::EnqueueRecvTask(int from, int idx) {
  permit_scheduler_->EnqueueRecvTask(from, idx);
}

int ConsensusManager::PrepareReceive() {
  // Init recv bufs (global consensus)
  for (auto& it : recv_tensors_) {
    Tensor* t = it.second;
    int size = t->tensor_data().size();
    memset(const_cast<char*>(t->tensor_data().data()), 0, size);
  }
  // Init agg done count
  tensor_aggregator_->InitAggBufStates();
}

int ConsensusManager::OpenReceive() {
  // TODO: INIT RECV BUF AND RCV COUNTERS BEFORE OPEN!
  std::lock_guard<std::mutex> guard(rcv_mu_);
  rcv_open_ = true;
  return 0;
}

int ConsensusManager::GetGlcTensor(const int& idx, Tensor*& out) {
  auto&& var = REMOTE_VARIABLES_[idx];
  int ret = var->GetGlcTensor(out);
  return ret;
}

int ConsensusManager::GetGlcTensor(const string& var_name, Tensor*& out) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  int ret = GetGlcTensor(idx, out);
  return ret;
}

void ConsensusManager::OpenReceive(int idx) {
  auto&& var = REMOTE_VARIABLES_[idx];
  var->StartRecv();
#if 0
  auto&& mu = var_rcv_mus_[idx];
  auto&& cv = var_rcv_cvs_[idx];
  mu.lock();
  // Init Global Consensus Buffer
  char* buf = (char*) glc_flats_[idx]->data();
  size_t size = glc_flats_[idx]->size() * sizeof(float);
  memset(buf, 0, size);
  // Init agg done count
  var_agg_done_cnts_[idx] = 0;
  // Init Receive Counters
  var_rcv_ing_cnts_[idx] = 0;
  var_rcv_done_cnts_[idx] = 0;
  var_rcv_doors_[idx] = 1;
  mu.unlock();
#endif
}

void ConsensusManager::OpenReceive(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  OpenReceive(idx);
}

int ConsensusManager::CloseReceive() {
  // TODO: DO NOT OPEN AGAIN BEFORE REDUCE FINISHES.
  //std::lock_guard<std::mutex> guard(rcv_mu_);
  rcv_mu_.lock();
  rcv_open_ = false;
  rcv_cv_.notify_all();
  rcv_mu_.unlock();
  return 0;
}

void ConsensusManager::CloseReceive(int idx) {
  auto&& mu = var_rcv_mus_[idx];
  auto&& cv = var_rcv_cvs_[idx];
  mu.lock();
  var_rcv_doors_[idx] = 0;
  cv.notify_all();
  mu.unlock();
}

void ConsensusManager::CloseReceive(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  CloseReceive(idx);
}

bool ConsensusManager::IsReceiveDone() {
  std::lock_guard<std::mutex> guard(rcv_mu_);
  if (!rcv_open_ && rcv_ing_cnt_ == rcv_done_cnt_) {
    return true;
  }
  return false;
}

int ConsensusManager::WaitAndGetNumIncomings() {
  CloseReceive();
  //if (rcv_ing_cnt_ > 1) {
  //  LOG(INFO) << "[DEBUG] WaitAndGetNumIncomings(): rcv_ing_cnt=" << rcv_ing_cnt_;
  //}
#if 1
  //auto start_time = std:chrono::system_clock::now();
  //auto last_time = start_time;
  std::unique_lock<std::mutex> lk(rcv_mu_);
  rcv_cv_.wait(lk, [&] {
#if 1
      //auto curr_time = std::chrono::system_clock::now();
      //std::chrono::duration<double> time_diff = curr_time - last_time;
      bool recv_done = (!rcv_open_ && rcv_ing_cnt_ == rcv_done_cnt_);
      if (!recv_done) {
        for (auto it : rcv_status_) {
          //LOG(INFO) << "[DEBUG] rcving from: " << it.first << ", status=" << it.second;
        }
      }
      //LOG(INFO) << "[DEBUG] rcv_ing_cnt=" << rcv_ing_cnt_ << ", rcv_done_cnt=" << rcv_done_cnt_;
      return recv_done;
#else
        return (!rcv_open_ && rcv_ing_cnt_ == rcv_done_cnt_);
#endif
        //return (rcv_ing_cnt_ == rcv_done_cnt_);
      });
  lk.unlock();
#else

#endif
  /// Wait for aggregation done
  /// rcv_ing_cnt_ == rcv_done_cnt_ ensures that
  /// each state of all aggbuf is one of AggReady, AggInProgress or RecvReady
  if (rcv_done_cnt_ > 0) {
    //LOG(INFO) << "[DEBUG] WaitForAggregations() rcv_done_cnt=" << rcv_done_cnt_;
  }
#if 1
  bool all_agg_done = false;
  while (!all_agg_done) {
    all_agg_done = true;
    for (const auto& name : actual_comm_tensors_) {
      int agg_done_cnt = tensor_aggregator_->agg_done_cnt(name);
      all_agg_done &= agg_done_cnt == rcv_done_cnt_;
      if (!all_agg_done) {
        //LOG(INFO) << "[DEBUG] agg_done_cnt=" << agg_done_cnt << ", rcv_done_cnt=" << rcv_done_cnt_;
        break;
      }
    }
  }
#endif
  // TODO: Must return zero and don't average if counts don't match.
  return rcv_done_cnt_;
}

int ConsensusManager::GetNumIncomings(int idx) {
  auto&& var = REMOTE_VARIABLES_[idx];
  var->StopRecv();
  int ret = var->AggCount();
  return ret;
#if 0
  auto&& mu = var_rcv_mus_[idx];
  mu.lock();
  int ret = var_agg_done_cnts_[idx];
  mu.unlock();
  return ret;
#endif
}

int ConsensusManager::GetNumIncomings(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  int ret = GetNumIncomings(idx);
  return ret;
}

int ConsensusManager::GetNumIncomings() {
  LOG(ERROR) << "Not implemented. Terminating.";
  exit(1);
  int num_incomings = 0;
  rcv_mu_.lock();
  rcv_mu_.unlock();
  return num_incomings;
}

int ConsensusManager::CountReduceAndOpenRecv(std::string& name) {
  rcv_mu_.lock();
  reduce_cnt_++;
  if (is_init_num_rcv_tensors_) {
    if (reduce_cnt_ == num_rcv_tensors_) {
      /// Open Receive
      rcv_ing_cnt_ = 0;
      rcv_done_cnt_ = 0;
      rcv_steps_sum_ = 0;
      PrepareReceive();
      rcv_open_ = true;
      reduce_cnt_ = 0;
    }
  } else {
    actual_comm_tensors_.push_back(name);
    num_rcv_tensors_ = reduce_cnt_;
  }
  rcv_mu_.unlock();
  return 0;
}

void ConsensusManager::CountReduce(int idx) {
  auto&& mu = var_rcv_mus_[idx];
  mu.lock();
  var_reduce_dones_[idx] = 1;
  reduce_cnt_++;
  mu.unlock();
}

void ConsensusManager::CountReduce(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    exit(EXIT_FAILURE);
  }
  int idx = search->second;
  CountReduce(idx);
}

bool ConsensusManager::IsInitNumApplyOps() {
  return is_init_num_rcv_tensors_;
}

int ConsensusManager::InitNumRecvTensors() {
  rcv_mu_.lock();
  /// Open Receive
  rcv_ing_cnt_ = 0;
  rcv_done_cnt_ = 0;
  rcv_steps_sum_ = 0;
  PrepareReceive();
  rcv_open_ = true;
  reduce_cnt_ = 0;

  std::vector<std::string> tmp_vec;
  for (const auto& name : actual_comm_tensors_) {
    tmp_vec.push_back(name);
  }
  actual_comm_tensors_.clear();
  for (const auto& name : tensor_names_) {
    if (std::find(tmp_vec.begin(), tmp_vec.end(), name) != tmp_vec.end()) {
      actual_comm_tensors_.push_back(name);
    }
  }
  is_init_num_rcv_tensors_ = true;
  std::cout << "RANK:" << ptre_rank_ << " NUM_RECV_TENSORS = " << num_rcv_tensors_ << std::endl;
  rcv_mu_.unlock();
  return 0;
}

int ConsensusManager::ProcessAggregation() {
  return tensor_aggregator_->ProcessAggregation();
  //return tensor_aggregator_->ProcessAggregationNoVerbs();
}

int ConsensusManager::ProcessReceive() {
  LOG(ERROR) << "NOT IMPLEMENTED.";
  exit(1);
  //return rdma_manager_->ProcessReceive();
}

int ConsensusManager::get_peer() {
  return peer_selector_->get_peer();
}

void ConsensusManager::next_peer() {
  LOG(ERROR) << "THIS FUNCTION IS NOT READY.";
  exit(1);
  peer_selector_->next();
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

const std::vector<string>& ConsensusManager::GetGlcNameList() {
  return tensor_names_;
}

const std::vector<Tensor*>& ConsensusManager::GetSendTensorsList() {
  return send_tensors_list_;
}

void* ConsensusManager::buf_ptr(const BufType type, const string& name) {
  int idx = -1;
  auto it = buf_type_name_index_map_.find(type);
  if (it != buf_type_name_index_map_.end()) {
    auto it2 = it->second.find(name);
    if (it2 != it->second.end()) {
      idx = it2->second;
    } else {
      LOG(ERROR) << "name not exist: type=" << type << ", name=" << name;
      exit(EXIT_FAILURE);
    }
  } else {
    LOG(ERROR) << "BufType not exist: type=" << type;
    exit(EXIT_FAILURE);
  }
  /// Found
  if (idx >= 0) {
    return bufs_[idx];
  }
}

void ConsensusManager::ReceivePushNotify(int dst) {
  int ret = rdma_manager_->ReceivePushNotify(dst);
#if 0
  if (ret >= 0) {
    int idx = ret;
    // Set state to init aggregation
    auto&& mu = var_rcv_mus_[idx];
    mu.lock();
    var_rcv_done_cnts_[idx]++;
    if (var_rcv_ing_cnts_[idx] >= var_rcv_done_cnts_[idx]) {
      recv_status_[idx] = 1;
    } else {
      // Stale incoming. NOT USE THIS
    }
    mu.unlock();
  }
#endif
  if (ret >= 0) {
    int idx = ret;
    auto&& var = REMOTE_VARIABLES_[idx];
    var->RecvDone();
  }
}

void ConsensusManager::ProcessAggregation(int idx) {
  auto&& var = REMOTE_VARIABLES_[idx];
  var->Aggregate();
#if 0
  auto&& mu = var_rcv_mus_[idx];
  //auto&& cv = var_rcv_cvs_[idx];
  mu.lock();
  if (var_rcv_doors_[idx] && recv_status_[idx]) {
    AggregateSum(d, *glc_flats_[idx], *agg_flats_[idx]);
    var_agg_done_cnts_[idx]++;
    recv_status_[idx] = 0;
    permit_manager_->NextPeer(idx);
  }
  mu.unlock();
#endif
}

}  // namespace ptre
