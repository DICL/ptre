#include "ptre/tests/test_server_lib.h"

#include <fstream>

using std::string;

int kSize;
int kRank;
string kHostFile;
std::vector<string> kGrpcHosts;
std::thread kGrpcServerThread;

// Grpc Server
std::unique_ptr<grpc::Server> kGrpcServer = nullptr;
std::shared_ptr<ptre::GrpcClientCache> kGrpcClientCache = nullptr;

namespace ptre {

void InitTestPtre(const string& hostFile, int comm_size, int comm_rank,
    std::vector<string>& names, const std::vector<Tensor*>& tensors,
    ConsensusManager*& consensus_manager,
    RdmaMgr*& rdma_mgr) {
  kHostFile = hostFile;
  kSize = comm_size;
  kRank = comm_rank;
  LoadGrpcHosts(kHostFile, kGrpcHosts);
  std::cout << "Load Grpc Hosts done.\n";
  std::cout << "host_file: " << kHostFile << std::endl;
  std::cout << "size: " << kSize << std::endl;
  std::cout << "rank: " << kRank << std::endl;
  kGrpcClientCache = std::make_shared<GrpcClientCache>(kRank, kGrpcHosts);
  std::cout << "Init Grpc Client Cache done.\n";

  /// Init RdmaMgr & ConsensusManager
  rdma_mgr = new RdmaMgr(kSize, kRank, false);
  std::cout << "Init RdmaMgr done.\n";
  consensus_manager = new ConsensusManager();
  std::cout << "Init ConsensusManager done.\n";
  consensus_manager->SetRdmaMgr(rdma_mgr);
  std::cout << "Set RdmaMgr to ConsensusManager done.\n";

  /// Init Global Consensus
  std::vector<const Tensor*> inputs;
  for (int i = 0; i < tensors.size(); i++) {
    const Tensor* in = new Tensor(*tensors[i]);
    inputs.push_back(in);
  }
  consensus_manager->InitGlobalConsensusV2(names, inputs);
  std::cout << "Init Global Consensus done.\n";

  /// Run Grpc Server
  kGrpcServerThread = std::thread(RunGrpcServer, consensus_manager, rdma_mgr);
  std::cout << "Launching Grpc Server done.\n";

  /// Init Remote MR
  std::vector<BufType> buf_types;
  std::vector<string> buf_names;
  int num_bufs = rdma_mgr->GetRemoteAccessBufInfos(&buf_types, &buf_names);
  std::cout << "Retrieve Remote Access Buf Infos done.\n";
  for (int i = 0; i < num_bufs; i++) {
    std::cout << "name: " << buf_names[i] << std::endl;
  }
  for (int i = 0; i < kSize; i++) {
    GrpcClient* grpc_client;
    kGrpcClientCache->GetClient(i, &grpc_client);
    grpc_client->SetRdmaMgr(rdma_mgr);
  }
  std::cout << "Set Rdma Manager for grpc clients done.\n";
  bool peer_flag[kSize] = { };

  // Set Local Remote MR
  //RdmaEnv* env = rdma_mgr->rdma_env();
  //rdma_mgr->SetDlid(kRank, env->port_attr.lid);
  //rdma_mgr->set_qpn(kRank, rdma_mgr->qp(kRank)->qp_num);
  //rdma_mgr->set_snp(kRank, env->gid.global.subnet_prefix);
  //rdma_mgr->set_iid(kRank, env->gid.global.interface_id);
  //peer_flag[kRank] = true;

  int done_flag = 0;
  while (!done_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    done_flag = 1;
    for (int i = 0; i < kSize; i++) {
      if (peer_flag[i]) {
        continue;
      }
      GrpcClient* grpc_client;
      kGrpcClientCache->GetClient(i, &grpc_client);
      if (!rdma_mgr->IsDlidSet(i)) {
        int ret = grpc_client->GetRemoteEnv();
        if (ret < 0) {
          done_flag = 0;
          continue;
        }
        if (i == kRank) {
          std::cout << "local_lid=" << rdma_mgr->lid() << ", remote_lid=" << rdma_mgr->remote_lid(i) << std::endl;
        }
      }
      int client_status = 0;
      for (int j = 0; j < num_bufs; j++) {
        if (rdma_mgr->IsRemoteMRSetV2(i, buf_types[j], buf_names[j])) {
          continue;
        }
        RemoteMR rmr;
        int ret = grpc_client->GetRemoteAddressV2(buf_types[j], buf_names[j], &rmr);
        if (ret < 0) {
          client_status = -1;
          break;
        } else {
#if 0
          if (i == kRank) {
            struct ibv_mr* mr = rdma_mgr->GetMR(buf_types[j], buf_names[j]);
            std::cout << "mr.addr=" << (uint64_t) mr->addr << std::endl;
            std::cout << "remote_addr=" << rmr.remote_addr << std::endl;
            std::cout << "mr.lkey=" << mr->lkey << std::endl;
            std::cout << "mr.rkey=" << mr->rkey << std::endl;
            std::cout << "rkey=" << rmr.rkey << std::endl;
            //rmr.remote_addr = (uint64_t) mr->addr;
            //rmr.rkey = mr->lkey;
          }
#endif
          rdma_mgr->SetRemoteMRV2(i, buf_types[j], buf_names[j],
              rmr.remote_addr, rmr.rkey);
        }
      }
      if (client_status < 0) {
        done_flag = 0;
        continue;
      }
      peer_flag[i] = true;
    }
  }
  std::cout << "Set Remote MR done.\n";

  // Init Rdma Aggregation Writer
  rdma_mgr->InitAggWriter();
  std::cout << "Init Agg Writer done.\n";

  //  Connect QPs
  done_flag = 0;
  while (!done_flag) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    done_flag = 1;
    for (int i = 0; i < kSize; i++) {
      int r = rdma_mgr->ConnectQP(i);
      if (r < 0) {
        done_flag = 0;
      }
    }
  }
  std::cout << "Connect QPs done.\n";

}

void LoadGrpcHosts(const string& hostFile, std::vector<string>& grpcHosts) {
  std::ifstream in(hostFile);
  string in_line;
  while (std::getline(in, in_line)) {
    if (in_line[0] == '#') continue;
    grpcHosts.emplace_back(in_line);
  }
  in.close();
}

void RunGrpcServer(ConsensusManager* cm, RdmaMgr* rdma_mgr) {
  RdmaServiceImpl service;
  service.SetConsensusManager(cm);
  service.SetRdmaMgr(rdma_mgr);
  //service.SetBarrierVariable(barrier_variable);
  std::string server_address("0.0.0.0:50051");
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  kGrpcServer = builder.BuildAndStart();
  //std::cout << "Grpc server listening on " << server_address << std::endl;
  kGrpcServer->Wait();
}

}
