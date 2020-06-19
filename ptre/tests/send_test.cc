#include <chrono>
#include <iostream>
#include <thread>

#include "ptre/common/operations.h"
#include "ptre/common/rdma/rdma_mpi.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/ptre_global.h"

#include <bsoncxx/builder/stream/array.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/stdx.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/instance.hpp>

using bsoncxx::builder::stream::close_array;
using bsoncxx::builder::stream::close_document;
using bsoncxx::builder::stream::document;
using bsoncxx::builder::stream::finalize;
using bsoncxx::builder::stream::open_array;
using bsoncxx::builder::stream::open_document;

using namespace std;
using namespace ptre::common;

int main(int argc, char* argv[]) {
  int rank, size;
  ptre_init(atoi(argv[argc - 3]), atoi(argv[argc - 1]), argv[argc - 5], 0, 1);
  rank = ptre_rank();
  size = ptre_size();
  PtreGlobal& ptre_global = PtreGlobalState();
  RdmaContext ctx(ptre_global.rdma_mgr);

  const size_t bytes = (argc > 7) ? atol(argv[1]) : 4096;
  const int warmup_iters = 5;
  const int iters = 20;
  chrono::system_clock::time_point tps[iters][2];
  void* arr = aligned_alloc(64, bytes);
  const int count = bytes / sizeof(float);

  if (rank == 0) {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
      RdmaSend((void*) arr, count, DataType::DT_FLOAT, 1, 0, &ctx);
    }
    for (int i = 0; i < warmup_iters; i++) {
      RdmaSend((void*) arr, 1, DataType::DT_FLOAT, 1, 0, &ctx);
    }
    PtreBarrier();
    // Benchmark
    for (int i = 0; i < iters; i++) {
      tps[i][0] = chrono::system_clock::now();
      RdmaSend((void*) arr, count, DataType::DT_FLOAT, 1, 0, &ctx);
      tps[i][1] = chrono::system_clock::now();
    }
  } else {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
      RdmaRecv((void*) arr, count, DataType::DT_FLOAT, 0, 0, &ctx, NULL);
    }
    for (int i = 0; i < warmup_iters; i++) {
      RdmaRecv((void*) arr, 1, DataType::DT_FLOAT, 0, 0, &ctx, NULL);
    }
    PtreBarrier();
    // Benchmark
    for (int i = 0; i < iters; i++) {
      RdmaRecv((void*) arr, count, DataType::DT_FLOAT, 0, 0, &ctx, NULL);
    }
  }

  ptre_finalize(0);

  if (rank == 0) {
    array<long int, iters> ns_arr;
    array<long int, iters> us_arr;
    array<long int, iters> ms_arr;
    array<long int, iters> s_arr;
    for (int i = 0; i < iters; i++) {
      auto dur_ns = chrono::duration_cast<chrono::nanoseconds>(tps[i][1] - tps[i][0]).count();
      auto dur_us = chrono::duration_cast<chrono::microseconds>(tps[i][1] - tps[i][0]).count();
      auto dur_ms = chrono::duration_cast<chrono::milliseconds>(tps[i][1] - tps[i][0]).count();
      auto dur_s = chrono::duration_cast<chrono::seconds>(tps[i][1] - tps[i][0]).count();
      ns_arr[i] = dur_ns;
      us_arr[i] = dur_us;
      ms_arr[i] = dur_ms;
      s_arr[i] = dur_s;
    }
    sort(ns_arr.begin(), ns_arr.end());
    sort(us_arr.begin(), us_arr.end());
    sort(ms_arr.begin(), ms_arr.end());
    sort(s_arr.begin(), s_arr.end());

    long int ns_sum;
    long int us_sum;
    long int ms_sum;
    long int s_sum;
    for (int i = iters / 4; i < 3 * iters / 4; i++) {
      ns_sum += ns_arr[i];
      us_sum += us_arr[i];
      ms_sum += ms_arr[i];
      s_sum += s_arr[i];
    }
    long int ns_mean = 2 * ns_sum / iters;
    long int us_mean = 2 * us_sum / iters;
    long int ms_mean = 2 * ms_sum / iters;
    long int s_mean = 2 * s_sum / iters;
    cout << ns_mean << endl;

    auto builder = bsoncxx::builder::stream::document{};
    bsoncxx::document::value doc_value = builder
        << "name" << "RdmaSend"
        << "lib" << "ptre"
        << "lib_info" << bsoncxx::builder::stream::open_document
          << "commit" << "469d34e30ab5eede225b17b4c0696d1ebc921466"
        << close_document
        << "optimizer" << "-O3"
        << "parameters" << bsoncxx::builder::stream::open_document
          << "warmup_iters" << warmup_iters
          << "iters" << iters
          << "size" << int64_t(bytes)
          << "mean_type" << "interquartile"
        << close_document
        << "measures" << bsoncxx::builder::stream::open_document
          << "ns" << bsoncxx::builder::stream::open_document
            << "mean" << ns_mean
          << close_document
          << "us" << bsoncxx::builder::stream::open_document
            << "mean" << us_mean
          << close_document
          << "ms" << bsoncxx::builder::stream::open_document
            << "mean" << ms_mean
          << close_document
          << "s" << bsoncxx::builder::stream::open_document
            << "mean" << s_mean
          << close_document
        << close_document
        << bsoncxx::builder::stream::finalize;

#if 0
    cout << bsoncxx::to_json(doc_value) << endl;
#else
    mongocxx::instance instance{};
    mongocxx::client client(mongocxx::uri("mongodb://localhost:27018"));

    mongocxx::database db = client["research"];
    mongocxx::collection coll = db["collectiveCommunication"];

    bsoncxx::stdx::optional<mongocxx::result::insert_one> result =
        coll.insert_one(std::move(doc_value));
#endif
  }

  return 0;
}
