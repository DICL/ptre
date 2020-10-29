#include "ptre/common/utils/host_file_parser.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace ptre {
namespace common {

HostFileParser::HostFileParser(const string& hostfile) {
  hostfile_ = hostfile;
}

PtreNode ParseLine(string line) {
  // Find hostname
  auto search = line.find(':');
  string hostname = line.substr(0, search);
  line.erase(0, search + 1);
  // Find ports
  std::vector<int> ports;
  while (line.length() > 0) {
    auto psrch = line.find(',');
    string port_str = line.substr(0, psrch);
    ports.push_back(std::stoi(port_str));
    if (psrch != std::string::npos) {
      line.erase(0, psrch + 1);
    } else {
      line.erase(0, line.length());
    }
  }
  PtreNode node;
  node.hostname = std::move(hostname);
  node.local_size = ports.size();
  node.grpc_ports = std::move(ports);

  return std::move(node);
}

std::vector<PtreWorker> ParseNode(const PtreNode& node) {
  std::vector<PtreWorker> workers;
  for (int i = 0; i < node.local_size; i++) {
    PtreWorker worker;
    worker.local_rank = i;
    std::stringstream ss;
    ss << node.hostname << ":" << node.grpc_ports[i];
    string tcp_host = ss.str();
    worker.tcp_host = std::move(tcp_host);
    worker.host = node;
    worker.port = node.grpc_ports[i];
    workers.push_back(std::move(worker));
  }
  return std::move(workers);
}

int HostFileParser::Parse(int size) {
  int cnt = 0;
  nodes_.clear();
  workers_.clear();
  string line;
  std::ifstream ifs(hostfile_);
  while (std::getline(ifs, line) && cnt < size) {
    auto node = ParseLine(line);
    auto local_workers = ParseNode(node);
    for (auto& worker : local_workers) {
      worker.rank = cnt++;
      workers_.push_back(std::move(worker));
      if (size == cnt) break;
    }
    nodes_.push_back(std::move(node));
  }
  if (cnt < size) {
    LOG(ERROR) << "Not enough hosts in the hostfile";
    exit(1);
  }
  size_ = size;
  is_parsed_ = true;
  ifs.close();
  return 0;
}

}  // namespace common
}  // namespace ptre
