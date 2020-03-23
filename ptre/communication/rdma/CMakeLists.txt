#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_library          (rdma SHARED rdma.cc)
target_link_libraries(rdma
                        ibverbs)

add_library          (rdma_manager SHARED rdma_manager.cc)
target_link_libraries(rdma_manager
                        :libtensorflow_framework.so.2
                        rdma
                        )

add_library          (rdma_grpc_server SHARED grpc_server.cc)
target_link_libraries(rdma_grpc_server
                        :libtensorflow_framework.so.2
                        rdma_service_grpc_proto
                        rdma_manager
                        consensus_manager
                        )

add_library          (rdma_grpc_client SHARED grpc_client.cc)
target_link_libraries(rdma_grpc_client
                        rdma_service_grpc_proto
                        rdma_manager
                        glog)

add_library (rdma_task SHARED rdma_task.cc)
target_link_libraries(rdma_task rdma)

add_library (rdma_worker SHARED rdma_worker.cc)
target_link_libraries(rdma_worker rdma_task)

add_library (rdma_agg_writer SHARED rdma_agg_writer.cc)
target_link_libraries(rdma_agg_writer
  rdma
  tensor_aggregator)

add_executable       (grpc_server_test grpc_server_test.cc)
target_link_libraries(grpc_server_test
                        pthread
                        rdma_grpc_server)
