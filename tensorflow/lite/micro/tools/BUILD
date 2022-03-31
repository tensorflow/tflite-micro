py_library(
    name = "generate_test_for_model",
    srcs = ["generate_test_for_model.py"],
    srcs_version = "PY3",
    visibility = ["//:__subpackages__"],
    deps = [
        "//tensorflow/lite/python:schema_py",
    ],

)

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_binary(
    name = "generate_cc_arrays",
    srcs = ["generate_cc_arrays.py"],
)

cc_binary(
    name = "tflite_flatbuffer_align",
    srcs = [
        "tflite_flatbuffer_align.cc",
    ],
    deps = [
        "//tensorflow/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)
