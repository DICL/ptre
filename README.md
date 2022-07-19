# PTRE: P2P-based asynchronous distributed Training Runtime Environment

## Introduction
PTRE is a distributed deep learning training runtime for Tensorflow.
PTRE uses peer-to-peer based asynchronous communication for data synchronization.
The communication component is implemented with RDMA technology to reduce communication costs and achieve better training throughput.

## 1. Build Project
```
cd $PROJECT_ROOT
git submodule update —init --recursive
mkdir build
cd build
cmake ..
cmake --build .
```

## 2. Set Environment Variables (Install)
```
export PTRE_SHARED_LIB=${PROJECT_ROOT}/build/ptre/tensorflow/kernels/libptre_ops.so
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/ptre/python"
```

## 3. Example
```
cd $PROJECT_ROOT/examples
./run-ptre tensorflow2_keras_mnist.py hosts 1
```
Command just `./run-ptre` without any argument to see what each argument means. \
To run a training on multiple hosts, provide your host file that contains your hosts information.
Each line of a host file should be formatted as:
```
GRPC_HOSTNAME:GRPC_PORT
```
