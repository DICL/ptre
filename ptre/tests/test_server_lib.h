#ifndef PTRE_TESTS_TEST_SERVER_LIB_H_
#define PTRE_TESTS_TEST_SERVER_LIB_H_

#include <string>
#include <vector>
#include <memory>
#include <thread>

#include "ptre/cm/consensus_manager.h"
#include "ptre/communication/rdma/rdma.h"
#include "ptre/communication/rdma/rdma_manager.h"
#include "ptre/communication/rdma/grpc_server.h"
#include "ptre/communication/rdma/grpc_client.h"
#include "ptre/communication/grpc/grpc_client_cache.h"

namespace ptre {

void InitTestPtre(const string& hostFile, int comm_size, int comm_rank,
    std::vector<string>& names, const std::vector<Tensor*>& tensors,
    ConsensusManager*& consensus_manager,
    RdmaManager*& rdma_manager);
void LoadGrpcHosts(const string& hostFile, std::vector<string>& grpcHosts);
void RunGrpcServer(ConsensusManager* cm, RdmaManager* rdma_manager);

}
#endif  // PTRE_TESTS_TEST_SERVER_LIB_H_
