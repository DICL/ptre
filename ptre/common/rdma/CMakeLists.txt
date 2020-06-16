add_library(rdma_context SHARED rdma_context.cc)
target_link_libraries(rdma_context
  rdma_channel)

add_library(rdma_request SHARED rdma_request.cc)
target_link_libraries(rdma_request
  ibverbs)

add_library(rdma_mpi SHARED rdma_mpi.cc)
target_link_libraries(rdma_mpi
  message
  rdma_request)
