set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -pthread")

include_directories(
  /home/wkim/.local/include/libmongoc-1.0
  /home/wkim/.local/include/bsoncxx/v_noabi
  /home/wkim/.local/include/mongocxx/v_noabi)


add_executable(send_recv_test send_recv_test.cc)
target_link_libraries(send_recv_test
  operations
  rdma_mpi
  rdma_context
  ptre_global)

add_executable(allreduce_test allreduce_test.cc)
target_link_libraries(allreduce_test
  operations
  ibverbs)

add_executable(send_test send_test.cc)
target_link_libraries(send_test
  operations
  mongocxx
  bsoncxx
  ibverbs)


#add_executable(cas_test cas_test.cc)
#target_link_libraries(cas_test
#  rdma_mgr
#  :libtensorflow_framework.so.2
#  grpc_client_cache
#  pthread
#  consensus_manager
#  rdma_grpc_server
#  rdma_grpc_client)

#add_executable(agg_test agg_test.cc)
#target_link_libraries(agg_test
#  pthread
#  :libtensorflow_framework.so.2
#  tensor_aggregator)

#add_library(test_server_lib SHARED test_server_lib.cc)
#target_link_libraries(test_server_lib
#  :libtensorflow_framework.so.2
#  rdma
#  pthread
#  consensus_manager
#  rdma_mgr
#  rdma_grpc_server
#  rdma_grpc_client
#  grpc_client_cache
#  )


#add_executable(state_test state_test.cc)
#target_link_libraries(state_test
#  test_server_lib
#  pthread
#  consensus_manager
#  tensor_aggregator
#  rdma_mgr
#  rdma_grpc_server
#  rdma_grpc_client
#  grpc_client_cache
#  )
