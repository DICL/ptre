package(default_visibility = ["//visibility:public"])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library_cc")

tf_proto_library_cc(
    name = "ptre_service_proto",
    srcs = ["protobuf/ptre_service.proto"],
    has_services = 1,
    cc_api_version = 2,
)

cc_library(
    name = "ptre_service",
    srcs = ["rpc/ptre_service.cc"],
    hdrs = ["rpc/ptre_service.h"],
    deps = [
        ":ptre_service_proto_cc",
        "//tensorflow:grpc++",
    ],
)

cc_library(
    name = "ptre_service_impl",
    srcs = ["rpc/ptre_service_impl.cc"],
    hdrs = ["rpc/ptre_service_impl.h"],
    deps = [
        ":ptre_service",
    ],
)

cc_library(
    name = "ptre_server",
    srcs = ["core/ptre_server.cc"],
    hdrs = ["core/ptre_server.h"],
    deps = [
        ":ptre_service_impl",
        ":remote_store",
        "//tensorflow/core:framework",
    ]
)

# Consensus Manager

cc_library(
    name = "remote_store",
    srcs = ["cm/remote_store.cc"],
    hdrs = [
        "cm/remote_store.h",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
)
