#include "ptre/common/operations.h"

int main(int argc, char* argv[]) {
  for (int i = 0; i < argc; i++) {
    LOG(INFO) << argv[i];
  }
  return 0;
}
