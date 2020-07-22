add_library (grpc_client_cache SHARED grpc_client_cache.cc)
target_link_libraries(grpc_client_cache
  rdma_grpc_client
  tcp_grpc_client
  )
