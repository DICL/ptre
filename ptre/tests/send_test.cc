#include <chrono>
#include <iostream>
#include <thread>

#include "ptre/common/operations.h"
#include "ptre/common/rdma/rdma_mpi.h"
#include "ptre/common/rdma/rdma_context.h"
#include "ptre/common/ptre_global.h"

#include <infiniband/verbs.h>

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
  const int warmup_iters = 0;
  const int iters = 1;
  chrono::system_clock::time_point tps[iters][2];
  void* arr = aligned_alloc(64, bytes);
  const int count = bytes / sizeof(float);
  //struct ibv_mr* mr;

  if (rank == 0) {
    //mr = ibv_reg_mr(ptre_global.rdma_mgr->pd(), arr, bytes, 0);
    //RdmaContext ctx(ptre_global.rdma_mgr, mr);

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
    //mr = ibv_reg_mr(ptre_global.rdma_mgr->pd(), arr, bytes,
    //    IBV_ACCESS_LOCAL_WRITE);
    //RdmaContext ctx(ptre_global.rdma_mgr, NULL, mr);

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

  //ibv_dereg_mr(mr);
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

    long int ns_sum, us_sum, ms_sum, s_sum;
    long int ns_mean, us_mean, ms_mean, s_mean;
    string mean_type;
    if (iters % 4 == 0 && iters / 4 > 0) {
      mean_type = "interquartile";
      for (int i = iters / 4; i < 3 * iters / 4; i++) {
        ns_sum += ns_arr[i];
        us_sum += us_arr[i];
        ms_sum += ms_arr[i];
        s_sum += s_arr[i];
      }
      ns_mean = 2 * ns_sum / iters;
      us_mean = 2 * us_sum / iters;
      ms_mean = 2 * ms_sum / iters;
      s_mean = 2 * s_sum / iters;
    } else {
      mean_type = "average";
      for (int i = 0; i < iters; i++) {
        ns_sum += ns_arr[i];
        us_sum += us_arr[i];
        ms_sum += ms_arr[i];
        s_sum += s_arr[i];
      }
      ns_mean = ns_sum / iters;
      us_mean = us_sum / iters;
      ms_mean = ms_sum / iters;
      s_mean = s_sum / iters;
    }
    cout << ns_mean << endl;

    auto builder = bsoncxx::builder::stream::document{};
    bsoncxx::document::value doc_value = builder
        << "name" << "RdmaSend"
        << "lib" << "ptre"
        << "lib_info" << bsoncxx::builder::stream::open_document
          << "commit" << "5a6f00041d14b1aeb336e890261b3b7e49b15d57"
        << close_document
        << "optimizer" << "-O3"
        << "parameters" << bsoncxx::builder::stream::open_document
          << "warmup_iters" << warmup_iters
          << "iters" << iters
          << "size" << int64_t(bytes)
          << "mean_type" << mean_type
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

#if 1
    cout << bsoncxx::to_json(doc_value) << endl;
#else
    mongocxx::instance instance{};
    mongocxx::client client(mongocxx::uri("mongodb://localhost:27018"));

    mongocxx::database db = client["research"];
    mongocxx::collection coll = db["coll_staging"];

    bsoncxx::stdx::optional<mongocxx::result::insert_one> result =
        coll.insert_one(std::move(doc_value));
#endif
  }

  return 0;
}
