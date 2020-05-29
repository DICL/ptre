add_executable(concurrent_queue_test concurrent_queue_test.cc)
target_link_libraries(concurrent_queue_test
    pthread)

add_library(shared_mutex SHARED shared_mutex.cc)
set_target_properties(shared_mutex PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  )
