#add_library(ptre_global SHARED ptre_global.cc)
#target_link_libraries(ptre_global
#                      consensus_manager
#                      rdma_mgr)
                    

add_subdirectory(common)
add_subdirectory(tensorflow)
add_subdirectory(core)
add_subdirectory(protobuf)
add_subdirectory(lib)

#add_subdirectory(tests)
