add_subdirectory(grpc)
add_subdirectory(rdma)
add_subdirectory(tcp)

add_library(push_variable SHARED push_variable.cc)
target_link_libraries(push_variable
  :libtensorflow_framework.so.2
  allocator
  )

add_library(pull_variable SHARED pull_variable.cc)
target_link_libraries(pull_variable
  :libtensorflow_framework.so.2
  allocator
  )
