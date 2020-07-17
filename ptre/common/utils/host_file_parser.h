#ifndef PTRE_COMMON_UTILS_HOST_FILE_PARSER_H_
#define PTRE_COMMON_UTILS_HOST_FILE_PARSER_H_

#include <map>
#include <string>

#include "ptre/common/common.h"

namespace ptre {
namespace common {

class HostFileParser {
 public:
  HostFileParser(const string& hostfile);
  //PtreNode ParseLine(const string& line);
  int Parse(int size);
  std::vector<PtreNode>& nodes() { return nodes_; }
  std::vector<PtreWorker>& workers() { return workers_; }

 private:
  string hostfile_;
  bool is_parsed_ = false;
  int size_;
  std::vector<PtreNode> nodes_;
  std::vector<PtreWorker> workers_;
};

}  // namespace common
}  // namespace ptre


#endif  // PTRE_COMMON_UTILS_HOST_FILE_PARSER_H_
