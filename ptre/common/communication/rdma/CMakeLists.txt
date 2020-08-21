#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

add_library          (rdma SHARED rdma.cc)
target_link_libraries(rdma
                        ibverbs)

add_library          (rdma_mgr SHARED rdma_mgr.cc)
target_link_libraries(rdma_mgr
  :libtensorflow_framework.so.2
  rdma
  remote_variable
  pull_variable
  push_variable
  rdma_channel
  allocator
  #rdma_agg_writer
  )

add_library          (rdma_grpc_server SHARED grpc_server.cc)
target_link_libraries(rdma_grpc_server
                        :libtensorflow_framework.so.2
                        rdma_service_grpc_proto
                        rdma_mgr
                        consensus_manager
  buffer_table
                        )

add_library          (rdma_grpc_client SHARED grpc_client.cc)
target_link_libraries(rdma_grpc_client
  rdma_service_grpc_proto
  rdma_mgr
  )

add_library (rdma_task SHARED rdma_task.cc)
target_link_libraries(rdma_task
  rdma
  rdma_channel
  )
#add_library (rdma_worker SHARED rdma_worker.cc)
#target_link_libraries(rdma_worker rdma_task)

#add_library (rdma_agg_writer SHARED rdma_agg_writer.cc)
#target_link_libraries(rdma_agg_writer
#  rdma
#  tensor_aggregator)

#add_executable       (grpc_server_test grpc_server_test.cc)
#target_link_libraries(grpc_server_test
#                        pthread
#                        rdma_grpc_server)

add_library (rdma_channel SHARED rdma_channel.cc)
target_link_libraries(rdma_channel rdma)

add_library (rdma_pull_job SHARED pull_job.cc)
target_link_libraries(rdma_pull_job
  rdma_task)
