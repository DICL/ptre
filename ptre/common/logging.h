#ifndef PTRE_COMMON_LOGGING_H_
#define PTRE_COMMON_LOGGING_H_

#include "tensorflow/core/platform/logging.h"

#define DBG(x) if (x == "predictions_kernel_0") DVLOG(0)
#define DBGR(x, r) \
    if (x == "predictions_kernel_0") \
      DVLOGR(0, r) << __FUNCTION__ << "] "

#endif  // PTRE_COMMON_LOGGING_H_
