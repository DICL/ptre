add_executable(concurrent_queue_test concurrent_queue_test.cc)
target_link_libraries(concurrent_queue_test
    pthread)
