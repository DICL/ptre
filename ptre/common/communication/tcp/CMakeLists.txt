add_library          (tcp_grpc_client         SHARED tcp_grpc_client.cc)
target_link_libraries(tcp_grpc_client
  :libtensorflow_framework.so.2
  tcp_service_grpc_proto)
target_compile_definitions(tcp_grpc_client PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

add_library          (tcp_service_impl        SHARED tcp_service_impl.cc)
target_link_libraries(tcp_service_impl
  :libtensorflow_framework.so.2
  consensus_manager
  ready_tensor
  tcp_service_grpc_proto
  common
  )
target_compile_definitions(tcp_service_impl PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

#add_executable(tcp_service_test tcp_service_test.cc)
#target_link_libraries(tcp_service_test
#  consensus_manager
#  rdma_grpc_server
#  tcp_grpc_client
#  tcp_service_impl
#  )
