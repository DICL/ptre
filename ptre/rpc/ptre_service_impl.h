#ifndef TENSORFLOW_PTRE_RPC_PTRE_SERVICE_IMPL_H_
#define TENSORFLOW_PTRE_RPC_PTRE_SERVICE_IMPL_H_

//#include "grpcpp/grpcpp.h"
#include "ptre/rpc/ptre_service.h"

namespace tensorflow {

class PtreServiceImpl final : public grpc::PtreService::AsyncService {

};

}  // namespace tensorflow

#endif  // TENSORFLOW_PTRE_RPC_PTRE_SERVICE_IMPL_H_
