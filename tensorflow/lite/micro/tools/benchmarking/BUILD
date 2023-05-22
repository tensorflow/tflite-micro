cc_library(
    name = "op_resolver",
    hdrs = ["op_resolver.h"],
    deps = ["//third_party/tflite_micro/tensorflow/lite/micro:op_resolvers"],
)

py_binary(
    name = "generate_model_specific_op_resolver",
    srcs = ["generate_model_specific_op_resolver.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/tensorflow/lite/python:schema_py",
        "//third_party/tensorflow/lite/tools:visualize_lib",
    ],
)

#cc_binary(
#    name = "tflm_benchmark",
#    srcs = ["internal/generic_model_benchmark.cc"],
#    deps = ["//tensorflow/lite/micro/tools/benchmarking/internal:generic_benchmark_lib"],
#)
