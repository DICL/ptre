add_executable(cas_test cas_test.cc)
target_link_libraries(cas_test rdma_manager
                               :libtensorflow_framework.so.2
                      grpc_client_cache
                      pthread
                      consensus_manager
                      rdma_grpc_server
                      rdma_grpc_client
                      )
