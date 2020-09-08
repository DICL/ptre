#include "ptre/common/operations.h"

#include <string>
#include <sstream>

using std::string;
using namespace ptre::common;

const size_t kSize_1 = 128;

void print_arr(float arr[], int count) {
  std::stringstream ss;
  for (int i = 0; i < count ;i++) {
    ss << arr[i];
    if (i + 1 < count) ss << ",";
  }
  LOG(INFO) << ss.str();
}

int main(int argc, char* argv[]) {
#if 1
  LOG(ERROR) << "Not Implemented.";
  exit(1);
#else
  // TODO: Use a command line argument parsing library.
  string hostfile = argv[2];
  int comm_size = atoi(argv[4]);
  int comm_rank = atoi(argv[6]);

  LOG(INFO) << hostfile;
  LOG(INFO) << comm_rank << " / " << comm_size;

  ptre_init(comm_size, comm_rank, hostfile.c_str(), 0, 1);

#if 0
  float t1[kSize_1];
  float r1[kSize_1];
#else
  float t1[1024];
  float r1[1024];
#endif
  for (int i = 0; i < kSize_1; i++) {
    t1[i] = 0.1 * comm_rank;
  }
  print_arr(t1, kSize_1);

#if 0
  PtreAllreduce((void*) t1, (void*) r1, kSize_1);
  print_arr(r1, kSize_1);
#else
  PtreAllreduce(COMM_IN_PLACE, (void*) t1, kSize_1);
  print_arr(t1, kSize_1);
#endif


  ptre_finalize(1);

  return 0;
#endif
}
