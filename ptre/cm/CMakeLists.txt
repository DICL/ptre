#add_library(remote_store SHARED remote_store.cc)
add_library(consensus_manager SHARED consensus_manager.cc)
target_link_libraries(consensus_manager
                      :libtensorflow_framework.so.2
                      rdma_manager)
#target_compile_definitions(consensus_manager PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
