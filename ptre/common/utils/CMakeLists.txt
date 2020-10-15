add_library(host_file_parser SHARED host_file_parser.cc)
target_link_libraries(host_file_parser
  common
  )
target_compile_definitions(host_file_parser PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)
