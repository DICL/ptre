#include "ptre/common/cm/consensus_manager.h"

#include <iostream>
#include <random>
#include <set>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <algorithm>

#include "ptre/common/cm/ready_tensor.h"

#define DDLOG DLOG(INFO) << "RANK:" << ptre_rank_ << " "

namespace ptre {
namespace common {

namespace {

int kCommbufStateIdle = COMMBUF_STATE_IDLE;
int kCommbufStateBusy = COMMBUF_STATE_BUSY;

}

ConsensusManager::ConsensusManager(int ptre_size, int ptre_rank,
    const std::vector<const Tensor*>& vars, const std::vector<string>& names)
    : commbuf_state_(COMMBUF_STATE_IDLE) {
  ptre_size_ = ptre_size;
  ptre_rank_ = ptre_rank;
  num_vars_ = vars.size();

  // Init Allocator
  size_t size_total = 0;
  std::vector<size_t> sizes;
  for (int i = 0; i < num_vars_; i++) {
    sizes.push_back(vars[i]->AllocatedBytes());  // Receive Buffer
    sizes.push_back(sizeof(int));  // Permit Buffer
  }
  allocator_ = new Allocator(sizes);
  // Init Remote Variable
  for (int i = 0; i < num_vars_; i++) {
    //RemoteVariable* rvar = new RemoteVariable(*vars[i], names[i], allocator_);
    //remote_variables_.push_back(rvar);
    /*
    LOG(INFO) << names[i] << ": TotalBytes()=" << vars[i]->TotalBytes()
        << ", AllocatedBytes()=" << vars[i]->AllocatedBytes();
    */
    var_names_.push_back(names[i]);
    var_name_to_index_[names[i]] = i;
  }
  // Init Ready Tensors
  ready_tensors_.reserve(num_vars_);
  for (int i = 0; i < num_vars_; i++) {
    ReadyTensor* t = new ReadyTensor(vars[i]->dtype(), vars[i]->shape());
    ready_tensors_.push_back(t);
  }
}

ConsensusManager::~ConsensusManager() {
  //if (rdma_mgr_ != nullptr) {
  //  delete rdma_mgr_;
  //}
  if (peer_selector_ != nullptr) {
    delete peer_selector_;
  }
  //if (tensor_aggregator_ != nullptr) {
  //  tensor_aggregator_->Terminate();
  //}
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

#if 0
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
    rdma_mgr_->RegisterMR(BUF_TYPE_RECV_BUF, names[i], buf, length, true);
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
    rdma_mgr_->RegisterMR(BUF_TYPE_SEND_BUF, names[i], buf, length, false);
  }
  /// 3. Agg Buf
  std::vector<Flat> recv_flats;
  for (int i = 0; i < num_vars_; i++) {
    recv_flats.push_back(global_consensus_[i]->flat<float>());
  }
  // TODO: Poll RECV CQ and get IMM DATA in TensorAggregator?
  struct ibv_cq* local_cq = rdma_mgr_->local_cq();
  struct ibv_qp* local_qp = rdma_mgr_->local_qp();
  tensor_aggregator_ = new TensorAggregator(nullptr, 0,
      rdma_mgr_->rdma_env(),
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
    rdma_mgr_->RegisterMR(BUF_TYPE_AGG_BUF, names[i], buf, length, true);
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
    rdma_mgr_->RegisterMR(BUF_TYPE_AGG_BUF_STATE, names[i], buf, length,
        true);
    struct ibv_mr* mr = rdma_mgr_->GetMR(BUF_TYPE_AGG_BUF_STATE, names[i]);
    tensor_aggregator_->SetStateMR(names[i], mr);
  }
  /// 4. Push Permit Array
  push_permits_.resize(num_vars);
  for (int i = 0; i < num_vars; i++) {
    int* elem = new int(-1);
    push_permits_[i] = elem;
    rdma_mgr_->RegisterMR(BUF_TYPE_PUSH_PERMIT_SRC, names[i], (void*) elem,
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
    rdma_mgr_->RegisterMR(BUF_TYPE_FLAG_RECV, name, buf, length, true);
    buf = (void*) &flag_to_send_;
    buf_types_.push_back(BUF_TYPE_FLAG_SEND);
    bufs_.push_back(buf);
    buf_lengths_.push_back(length);
    buf_type_name_index_map_[BUF_TYPE_FLAG_SEND].emplace(name, num_bufs_);
    num_bufs_++;
    rdma_mgr_->RegisterMR(BUF_TYPE_FLAG_SEND, name, buf, length, false);
  }
}
#endif

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

#if 0
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

  rdma_mgr_->InitTensorMR(0, name, recv_tensor, send_tensor);
  //for (int i = 0; i < ptre_size_; i++) {
  //  if (i == ptre_rank_) {
  //    continue;
  //  }
  //  rdma_mgr_->InitTensorMR(i, name, recv_tensor, send_tensor);
  //}
}

void ConsensusManager::InitBufParam() {
  is_new_incoming_ = new bool(false);
  rdma_mgr_->InitParamMR(is_new_incoming_, &flag_to_send_);
}
#endif

void ConsensusManager::SetRdmaMgr(RdmaMgr* rdma_mgr) {
  rdma_mgr_ = rdma_mgr;
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

#if 0
void ConsensusManager::PushModel(int dst_rank) {
  bool can_push = rdma_mgr_->AttemptPush(dst_rank);
  if (!can_push) {
    return;
  }

  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_mgr_->PushTensor(dst_rank, name, *t);  // num_comps + 1
  }

  rdma_mgr_->NotifyPushDone(dst_rank);
}


void ConsensusManager::PushTensors(int dst_rank) {
  for (auto it : buf_type_name_index_map_[BUF_TYPE_SEND_BUF]) {
    rdma_mgr_->RdmaWriteBufRemote(dst_rank, BUF_TYPE_SEND_BUF,
        BUF_TYPE_RECV_BUF, it.first, true);
  }
}

void ConsensusManager::PushTensors2(int dst_rank) {
  for (auto it : send_tensors_) {
    const std::string& name = it.first;
    Tensor* t = it.second;
    rdma_mgr_->RdmaWriteTensor(dst_rank, name, *t, true);
  }
  //std::cout << "\n[RANK=" << ptre_rank_ << "]: RdmaWriteTensor Done.\n";
  //flag_to_send_ = true;
  //rdma_mgr_->RdmaWriteIncomingFlag(dst_rank, &flag_to_send_);
  int num_comps = send_tensors_.size();
  rdma_mgr_->Poll(num_comps);
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
    rdma_mgr_->PushTensorAtomicAddBatch(dst_rank, name, *t);
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
        rdma_mgr_->PushTensorAtomicAddBatch(dst_rank, name, *t);
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
    rdma_mgr_->PushTensorBufferedAggregation(dst_rank, name);
  }
#else
  rdma_mgr_->PushTensorBufferedAggregation(dst_rank, actual_comm_tensors_);
#endif
  send_mu_.lock();
  send_status_ = SEND_IDLE;
  send_cv_.notify_all();
  send_mu_.unlock();
}
#endif

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
  LOG(INFO) << "DEBUG: CanReceive" << std::boolalpha << commbuf_state_;
  bool ret = commbuf_state_.compare_exchange_strong(
      kCommbufStateIdle, kCommbufStateBusy);
  return ret;
  LOG(ERROR) << "Deprecated.";
  exit(EXIT_FAILURE);
#if 0
  bool result = false;
  for (auto rvar : remote_variables_) {
    int ret = rvar->EnqueueSenderCandidate(src_rank);
    if (ret >= 0) {
      result = true;
    }
  }
  return result;
#endif
#if 0
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
#endif
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

int ConsensusManager::get_peer() {
  if (peer_selector_ != nullptr) {
    return peer_selector_->get_peer();
  }
  return -1;
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
      return NULL;
    }
  } else {
    LOG(ERROR) << "BufType not exist: type=" << type;
    return NULL;
  }
  /// Found
  if (idx >= 0) {
    return bufs_[idx];
  }
}

void ConsensusManager::ReceivePushNotify(int dst) {
#if 0
  int ret = rdma_mgr_->ReceivePushNotify(dst);
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
    auto&& var = remote_variables_[idx];
    var->SetAggState(1);
  }
#endif
}

//RemoteVariable* ConsensusManager::remote_variable(int idx) {
//  if (idx < num_vars_) {
//    return remote_variables_[idx];
//  }
//  return NULL;
//}

//RemoteVariable* ConsensusManager::remote_variable(const string& var_name) {
//  auto search = var_name_to_index_.find(var_name);
//  if (search == var_name_to_index_.end()) {
//    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
//    return NULL;
//  }
//  int idx = search->second;
//  return remote_variable(idx);
//}

//std::vector<RemoteVariable*>& ConsensusManager::remote_variables() {
//  return remote_variables_;
//}

ReadyTensor* ConsensusManager::ready_tensor(int idx) {
  if (idx < num_vars_) return ready_tensors_[idx];
  return NULL;
}

ReadyTensor* ConsensusManager::ready_tensor(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    return NULL;
  }
  int idx = search->second;
  return ready_tensor(idx);
}


const std::vector<string>& ConsensusManager::variable_names() {
  return var_names_;
}

int ConsensusManager::var_name_to_index(const string& var_name) {
  auto search = var_name_to_index_.find(var_name);
  if (search == var_name_to_index_.end()) {
    LOG(ERROR) << "KEY NOT FOUND: " << var_name;
    return -1;
  }
  int idx = search->second;
  return idx;
}

}  // namespace common
}  // namespace ptre
