add_library(ptre_ops SHARED ptre_ops.cc)
target_link_libraries(ptre_ops
                      :libtensorflow_framework.so.2
                      consensus_manager
                      rdma_manager
                      rdma_grpc_server
                      rdma_grpc_client)
