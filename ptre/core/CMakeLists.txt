#add_library(ptre_server ptre_server.cc)

add_library(allocator SHARED allocator.cc)
target_link_libraries(allocator
  :libtensorflow_framework.so.2
  )
