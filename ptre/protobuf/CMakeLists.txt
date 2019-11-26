add_library(rdma_service_proto SHARED rdma_service.pb.cc)
target_link_libraries(rdma_service_proto protobuf grpc++)

add_library(rdma_service_grpc_proto SHARED rdma_service.grpc.pb.cc)
target_link_libraries(rdma_service_grpc_proto rdma_service_proto)
