add_library(peer_selector SHARED peer_selector.cc)

#add_library(remote_store SHARED remote_store.cc)
add_library(consensus_manager SHARED consensus_manager.cc)
target_link_libraries(consensus_manager
                      :libtensorflow_framework.so.2
                      rdma_manager
                      peer_selector
                      grpc_client_cache
                      glog
                      )

add_library(tensor_aggregator SHARED tensor_aggregator.cc)
target_link_libraries(tensor_aggregator
  pthread
)

add_executable(test_peer_selector test_peer_selector.cc)
target_link_libraries(test_peer_selector peer_selector)
#target_compile_definitions(consensus_manager PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
