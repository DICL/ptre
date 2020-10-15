add_library(peer_selector SHARED peer_selector.cc)
target_link_libraries(peer_selector
                      :libtensorflow_framework.so.2
                      )


#add_library(remote_store SHARED remote_store.cc)
add_library(consensus_manager SHARED consensus_manager.cc)
target_link_libraries(consensus_manager
  :libtensorflow_framework.so.2
  allocator
  grpc_client_cache
  message
  peer_selector
  rdma_mgr
  ready_tensor
  #remote_variable
  )
target_compile_definitions(consensus_manager PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

#add_library(tensor_aggregator SHARED tensor_aggregator.cc)
#target_link_libraries(tensor_aggregator
#  rdma)

add_library(push_permit SHARED push_permit.cc)
target_link_libraries(push_permit
  allocator
  )

#add_library(remote_variable SHARED remote_variable.cc)
#target_link_libraries(remote_variable
#  :libtensorflow_framework.so.2
#  #push_permit
#  allocator
#  )
#target_compile_definitions(remote_variable PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

add_executable(test_peer_selector test_peer_selector.cc)
target_link_libraries(test_peer_selector peer_selector)
#target_compile_definitions(consensus_manager PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)

add_library(ready_tensor SHARED ready_tensor.cc)
target_link_libraries(ready_tensor
  common)
