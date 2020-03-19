add_executable(cas_test cas_test.cc)
target_link_libraries(cas_test rdma_manager
                               :libtensorflow_framework.so.2
                      grpc_client_cache
                      pthread
                      consensus_manager
                      rdma_grpc_server
                      rdma_grpc_client
                      )

add_executable(agg_test agg_test.cc)
target_link_libraries(agg_test 
  pthread
  :libtensorflow_framework.so.2
  tensor_aggregator)
