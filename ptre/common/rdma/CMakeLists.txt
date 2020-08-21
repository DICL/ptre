add_library(rdma_context SHARED rdma_context.cc)
target_link_libraries(rdma_context
  rdma_channel)

add_library(rdma_request SHARED rdma_request.cc)
target_link_libraries(rdma_request
  ibverbs)
#target_compile_definitions(rdma_request PRIVATE RDMA_REQUEST_BUSY_WAIT)

#set(CMAKE_CXX_FLAGS "-O3")
add_library(rdma_mpi SHARED rdma_mpi.cc)
target_link_libraries(rdma_mpi
  message
  rdma_request)
#target_compile_options(rdma_mpi BEFORE PRIVATE "-O3")

add_library(rdma_mpi_v2 SHARED rdma_mpi_v2.cc)
target_link_libraries(rdma_mpi_v2
  rdma_mpi
  message
  rdma_request)

add_library(rdma_controller SHARED rdma_controller.cc)
target_link_libraries(rdma_controller
  common)
