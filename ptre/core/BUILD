cc_library(
    name = "types_proto",
    srcs = ["types.pb.cc"],
    hdrs = ["types.pb.h"],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    deps = [
        ":types_proto",
    ],
)

cc_library(
    name = "tensor",
    srcs = ["tensor.cc"],
    hdrs = ["tensor.h"],
    linkopts = ["-libtensorflow_framework"],
    deps = [
        ":types",
    ],
)
