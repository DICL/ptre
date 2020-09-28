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
# PTRE_RDMA
add_definitions(-DPTRE_RDMA_PUSH)
add_definitions(-DPTRE_RDMA)
add_definitions(-DSKIP_HTOD_IF_NOT_READY)
#add_definitions(-DATOMIC_MODEL)
#add_definitions(-DMTR_ENABLED)

link_libraries(-pthread)
add_compile_options(
  -fno-strict-aliasing
  -fstack-protector-strong
  -fassociative-math
  -ffast-math
  -ftree-vectorize
  -funsafe-math-optimizations
  -mf16c -mavx -mfma)


#add_library(test_ops SHARED ptre/tensorflow/kernels/test_ops.cc)
#target_link_libraries(test_ops :libtensorflow_framework.so.2)

#add_executable(ptre
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread")
#set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -pthread")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pthread")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG -Wall")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_BUILD_TYPE Debug)
#message(${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
#message(${CMAKE_CXX_FLAGS_DEBUG})
#message(${CMAKE_CXX_FLAGS_RELEASE})
#message(${CMAKE_CXX_FLAGS_MINSIZEREL})
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE MinSizeRel)
#set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2")
#set(CMAKE_CXX_FLAGS "-O2")
#set(CMAKE_CXX_FLAGS "-Os -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -finline-functions -pthread")
#set(CMAKE_CXX_FLAGS_RELEASE "-O0 -pthread")

add_subdirectory(ptre)
add_subdirectory(test)
add_subdirectory(third_party)
