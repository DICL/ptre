#add_library(ptre_op_helpers SHARED ptre_op_helpers.cc)
#target_link_libraries(ptre_op_helpers
#                      :libtensorflow_framework.so.2)

find_package(CUDA 10.1 REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DGOOGLE_CUDA=1)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-x cu)
#set(CUDA_SEPARABLE_COMPILATION ON)
#CUDA_COMPILE(PTRE_OPS_GPU_O ptre_ops_gpu.cu.cc)
#cuda_add_library(ptre_ops_gpu SHARED ptre_ops_gpu.cu.cc)
#add_library(ptre_ops_gpu SHARED ptre_ops_gpu.cu.cc)
#target_link_libraries(ptre_ops_gpu
#                      :libtensorflow_framework.so.2
#)

#add_library(job_def SHARED job_def.cc)

#add_library(ptre_ops SHARED ptre_ops.cc ${PTRE_OPS_GPU_O})
add_library(ptre_ops SHARED ptre_ops.cc ptre_ops_gpu.cu.o)
#set_target_properties(ptre_ops PROPERTIES
#  CXX_STANDARD 17
#  CXX_EXTENSIONS OFF
#  )
#add_library(ptre_ops SHARED ptre_ops.cc ptre_ops_gpu.cu.cc)
#add_library(ptre_ops SHARED ptre_ops.cc)
target_link_libraries(ptre_ops
                      #ptre_global
                      #ptre_ops_gpu
                      :libtensorflow_framework.so.2
                      cudart
                      cublas
                      operations
                      ready_tensor
                      #shared_mutex
)

add_library(tf_ops_test SHARED tf_ops_test.cc)
target_link_libraries(tf_ops_test :libtensorflow_framework.so.2)
#target_link_libraries(tf_ops_test :libtensorflow_framework.so.2 operations)
