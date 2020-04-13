#include <vector>
#include <iostream>
#include <thread>
#include <string>
#include <chrono>

#include "ptre/cm/tensor_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"

using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::DT_FLOAT;
//using Flat = Eigen::TensorMap<Eigen::Tensor<float, 1>>;
using TFFlat = tensorflow::TTypes<float>::Flat;
using ptre::TensorAggregator;
using std::string;

#define NUM_TENSORS 10

namespace ptre {
}

int main() {
  /// Init tensors
  Tensor** tensors = (Tensor**) malloc(sizeof(Tensor) * NUM_TENSORS);
  //std::vector<Eigen::TensorMap<Eigen::Tensor<float, 1>>> flats;
  std::vector<TFFlat> flats;
  std::vector<string> names;
  for (int i = 0; i < NUM_TENSORS; i++) {
    auto t = new Tensor(DT_FLOAT, TensorShape({7, 7, 5}));
    //auto tfflat = t->flat<float>();
    //Eigen::TensorMap<Eigen::Tensor<float, 1>> tmap(tfflat.data(),
    //                                               tfflat.size());
    tensors[i] = t;
    //flats.emplace_back(tmap);
    flats.push_back(t->flat<float>());
    names.push_back(std::to_string(i));
  }
  Eigen::ThreadPool pool(32);
  TensorAggregator agg(&pool, 32, names, flats);

  std::cout << "sizeof(TFFlat) = " << sizeof(TFFlat) << std::endl;
  std::cout << tensors[0]->DebugString() << std::endl;

  auto buf = agg.agg_buf_ptr(0);
  while (*buf->state != ptre::StatefulAggBuf::kRecvReady);
  (*buf->flat).setConstant(0.1);
  *buf->state = ptre::StatefulAggBuf::kAggReady;

  while (*buf->state != ptre::StatefulAggBuf::kRecvReady);
  std::cout << tensors[0]->DebugString() << std::endl;

  (*buf->flat).setConstant(0.2);
  *buf->state = ptre::StatefulAggBuf::kAggReady;
  while (*buf->state != ptre::StatefulAggBuf::kRecvReady);
  std::cout << tensors[0]->DebugString() << std::endl;

  std::cout << agg.buf_ptr(0) << std::endl;
  std::cout << agg.buf_ptr("0") << std::endl;

  agg.Terminate();

  return 0;
}
