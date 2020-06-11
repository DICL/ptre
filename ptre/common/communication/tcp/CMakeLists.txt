add_library          (tcp_grpc_client         SHARED tcp_grpc_client.cc)
target_link_libraries(tcp_grpc_client
                        :libtensorflow_framework.so.2
                        tcp_service_grpc_proto)

add_library          (tcp_service_impl        SHARED tcp_service_impl.cc)
target_link_libraries(tcp_service_impl
                        :libtensorflow_framework.so.2
                        tcp_service_grpc_proto
                        consensus_manager)
