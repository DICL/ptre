add_library (grpc_client_cache SHARED grpc_client_cache.cc)
target_link_libraries(grpc_client_cache
  rdma_grpc_client
  tcp_grpc_client
  )
target_compile_definitions(grpc_client_cache PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
