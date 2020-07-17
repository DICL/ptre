#include <iostream>

#include "ptre/common/utils/host_file_parser.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " hostfile size\n";
    return 1;
  }
  std::string hostfile(argv[1]);
  int size = atoi(argv[2]);
  std::cout << "hostfile=" << hostfile << std::endl;
  std::cout << "size=" << size << std::endl;
  ptre::common::HostFileParser p(hostfile);
  std::cout << "Initialized parser\n";
  p.Parse(size);
  std::cout << "Parsing done.\n";

  for (auto& worker : p.workers()) {
    std::cout << "rank=" << worker.rank << ", local_rank=" << worker.local_rank
        << ", grpc_host=" << worker.grpc_host << std::endl;
  }

  return 0;
}
