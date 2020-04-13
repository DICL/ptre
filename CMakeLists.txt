cmake_minimum_required(VERSION 3.10)

project(ptre)

include_directories(.
                    /home/wkim/.local/include
                    /home/wkim/.local/lib/python2.7/site-packages/tensorflow_core/include
                    /usr/local/cuda/include
)

link_directories(/home/wkim/.local/lib
                 /home/wkim/.local/lib/python2.7/site-packages/tensorflow_core
                 /usr/local/cuda/lib64
)

#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
add_definitions(-DGOOGLE_CUDA=1)

add_subdirectory(ptre)

#add_library(test_ops SHARED ptre/kernels/test_ops.cc)
#target_link_libraries(test_ops :libtensorflow_framework.so.2)

#add_executable(ptre
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_BUILD_TYPE MinSizeRel)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS_RELEASE "-O2")
