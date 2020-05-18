add_subdirectory(grpc)
add_subdirectory(rdma)
add_subdirectory(tcp)

add_library(push_variable SHARED push_variable.cc)
#target_link_libraries(push_variable)
