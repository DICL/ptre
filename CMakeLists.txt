cmake_minimum_required(VERSION 3.10)

project(ptre)

include_directories(.
                    /home/wkim/.local/lib/python2.7/site-packages/tensorflow_core/include
                    /home/wkim/.local/include
                    )
link_directories(/home/wkim/.local/lib
                 /home/wkim/.local/lib/python2.7/site-packages/tensorflow_core
                )

add_subdirectory(ptre)

#add_library(test_ops SHARED ptre/kernels/test_ops.cc)
#target_link_libraries(test_ops :libtensorflow_framework.so.2)

#add_executable(ptre
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE MinSizeRel)
#set(CMAKE_CXX_FLAGS_RELEASE "-O2")
