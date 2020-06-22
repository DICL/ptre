add_subdirectory(cm)
add_subdirectory(communication)
add_subdirectory(rdma)

add_library(common SHARED common.cc)

add_library(message SHARED message.cc)
target_link_libraries(message
  common
  )

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
  message
  )

add_library(operations SHARED operations.cc)
target_link_libraries(operations
  :libtensorflow_framework.so.2
  ptre_global
  rdma_context
  rdma_mpi
  rdma_request)

add_executable(operations_test operations_test.cc)
target_link_libraries(operations_test
  operations)
#target_compile_definitions(operations_test
#  PRIVATE NDEBUG)