load("@tflm_pip_deps//:requirements.bzl", "requirement")

licenses(["notice"])

py_library(
    name = "test_model_lib",
    srcs = ["test_model.py"],
    deps = [
        "//tensorflow/lite/python:schema_py",
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        requirement("flatbuffers"),
    ],
)

py_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    main = "interpreter_test.py",
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from interpreter_wrapper_pybind.so
        "notap",  # TODO(b/247808903)
        "noubsan",
    ],
    deps = [
        ":test_model_lib",
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)

# py_binary(
#     name = "new_test",
#     srcs = ["new_test.py"],
#     python_version = "PY3",
#     srcs_version = "PY3ONLY",
#     tags = [
#         "nomicro_static",  # TF dep incompatible w/ TF_LITE_STATIC_MEMORY.
#         "noubsan",  # TODO(b/144512025): Fix raw_to_bitmap_test to fix ubsan failure.
#     ],
#     deps = [
#         "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
#         "//tensorflow/lite/micro/testing:create_test_model_lib",
#         "//tensorflow/lite/tools:flatbuffer_utils",
#         "@absl_py//absl:app",
#     ],
# )
