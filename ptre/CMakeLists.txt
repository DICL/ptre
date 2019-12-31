#add_library(ptre_global SHARED ptre_global.cc)
#target_link_libraries(ptre_global
#                      consensus_manager
#                      rdma_manager)
                    

add_subdirectory(cm)
add_subdirectory(communication)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(protobuf)
