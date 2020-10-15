add_library          (rdma_service_proto      SHARED rdma_service.pb.cc)
target_link_libraries(rdma_service_proto
                        protobuf
                        grpc++)

add_library          (rdma_service_grpc_proto SHARED rdma_service.grpc.pb.cc)
target_link_libraries(rdma_service_grpc_proto
                        rdma_service_proto)

add_library          (tcp_service_proto       SHARED tcp_service.pb.cc)
target_link_libraries(tcp_service_proto
                        protobuf
                        grpc++)

add_library          (tcp_service_grpc_proto  SHARED tcp_service.grpc.pb.cc)
target_link_libraries(tcp_service_grpc_proto
                        tcp_service_proto)

add_library(messages_proto SHARED messages.pb.cc)
target_link_libraries(messages_proto
  protobuf)
