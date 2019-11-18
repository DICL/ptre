add_library(ptre_ops SHARED ptre_ops.cc)
target_link_libraries(ptre_ops :libtensorflow_framework.so.2)
target_link_libraries(ptre_ops consensus_manager)
