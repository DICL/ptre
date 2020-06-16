#include "ptre/common/operations.h"

#include <string>
#include <sstream>

using std::string;
using namespace ptre::common;

const size_t kSize_1 = 64;

void print_arr(float arr[], int count) {
  std::stringstream ss;
  for (int i = 0; i < count ;i++) {
    ss << arr[i];
    if (i + 1 < count) ss << ",";
  }
  LOG(INFO) << ss.str();
}

int main(int argc, char* argv[]) {
  // TODO: Use a command line argument parsing library.
  string hostfile = argv[2];
  int comm_size = atoi(argv[4]);
  int comm_rank = atoi(argv[6]);

  LOG(INFO) << hostfile;
  LOG(INFO) << comm_rank << " / " << comm_size;

  ptre_init(comm_size, comm_rank, hostfile.c_str(), 0, 1);

  float t1[kSize_1];
  float r1[kSize_1];
  for (int i = 0; i < kSize_1; i++) {
    t1[i] = 0.1 * comm_rank;
  }
  print_arr(t1, kSize_1);

  //PtreAllreduce(COMM_IN_PLACE, (void*) t1, kSize_1);
  PtreAllreduce((void*) t1, (void*) r1, kSize_1);

  print_arr(r1, kSize_1);

  return 0;
}
