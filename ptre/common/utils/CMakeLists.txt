add_library(host_file_parser SHARED host_file_parser.cc)
target_link_libraries(host_file_parser
  common
  )
