#add_library(ptre_op_helpers SHARED ptre_op_helpers.cc)
#target_link_libraries(ptre_op_helpers
#                      :libtensorflow_framework.so.2)

find_package(CUDA 10.0 REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-D_GLIBCXX_USE_CXX11_ABI=0)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DGOOGLE_CUDA=1)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-x cu)
#set(CUDA_SEPARABLE_COMPILATION ON)
#CUDA_COMPILE(PTRE_OPS_GPU_O ptre_ops_gpu.cu.cc)
#cuda_add_library(ptre_ops_gpu SHARED ptre_ops_gpu.cu.cc)
#add_library(ptre_ops_gpu SHARED ptre_ops_gpu.cu.cc)
#target_link_libraries(ptre_ops_gpu
#                      :libtensorflow_framework.so.2
#)
#cuda_target_compile_definitions(PTRE_OPS_GPU_O PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)

#add_library(ptre_ops SHARED ptre_ops.cc ${PTRE_OPS_GPU_O})
add_library(ptre_ops SHARED ptre_ops.cc ptre_ops_gpu.cu.o)
#add_library(ptre_ops SHARED ptre_ops.cc ptre_ops_gpu.cu.cc)
#add_library(ptre_ops SHARED ptre_ops.cc)
target_link_libraries(ptre_ops
                      #ptre_global
                      #ptre_ops_gpu
                      :libtensorflow_framework.so.2
                      cudart
                      cublas
                      consensus_manager
                      rdma_manager
                      rdma_grpc_server
                      rdma_grpc_client
                      grpc_client_cache
                      glog
)
#target_compile_definitions(ptre_ops PUBLIC -D_GLIBCXX_USE_CXX11_ABI=0)
