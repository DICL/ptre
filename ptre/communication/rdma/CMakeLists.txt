add_library(rdma_grpc_server SHARED grpc_server.cc)
target_link_libraries(rdma_grpc_server :libtensorflow_framework.so.2)
