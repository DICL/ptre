add_subdirectory(cm)
add_subdirectory(communication)

add_library(ptre_global SHARED ptre_global.cc)
target_link_libraries(ptre_global
  consensus_manager
  rdma_mgr
  rdma_grpc_server
  rdma_grpc_client
  grpc_client_cache
  rdma_task
  rdma_channel
  rdma_pull_job
  )


add_library(operations SHARED operations.cc)
target_link_libraries(operations
  :libtensorflow_framework.so.2
  ptre_global
  )
