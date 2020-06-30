#/home/wkim/.local/bin/protoc -I=ptre/core --cpp_out=ptre/core ptre/core/types.proto
#protoc -I=ptre/core --cpp_out=ptre/core ptre/core/types.proto
export LD_LIBRARY_PATH=/home/wkim/.local/lib
export GRPC_CPP_PLUGIN=/home/wkim/.local/bin/grpc_cpp_plugin
PROTOC3=/home/wkim/.local/bin/protoc
PROTO_FILES=(
"ptre/protobuf/rdma_service.proto"
"ptre/protobuf/tcp_service.proto"
)
PROTO_ONLY_FILES=(
"ptre/protobuf/messages.proto"
)
for proto_file in ${PROTO_FILES[@]}
do
  echo ${proto_file}
  ${PROTOC3} -I=ptre/protobuf --grpc_out=ptre/protobuf --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} ${proto_file}
  ${PROTOC3} -I=ptre/protobuf --cpp_out=ptre/protobuf ${proto_file}
done
for proto_file in ${PROTO_ONLY_FILES[@]}
do
  echo ${proto_file}
  ${PROTOC3} -I=ptre/protobuf --cpp_out=ptre/protobuf ${proto_file}
done
