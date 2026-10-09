load("//python:py_rules.bzl", "tflm_py_binary")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    default_visibility = ["//:__subpackages__"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_py_binary(
    name = "generate_micro_mutable_op_resolver_from_model",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model.py",
    ],
    deps = [
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/python:schema_util",
        "//tensorflow/lite/micro/tools:flatbuffer_utils",
        "//tensorflow/lite/micro/tools:visualize",
    ],
)

genrule(
    name = "generate_person_detect_op_resolver_hdr",
    srcs = [
        "//tensorflow/lite/micro/models:person_detect.tflite",
    ],
    outs = [
        "person_detect/gen_micro_mutable_op_resolver.h",
    ],
    cmd = """
        $(execpath :generate_micro_mutable_op_resolver_from_model) \
            --common_tflite_path=$$(dirname $(execpath //tensorflow/lite/micro/models:person_detect.tflite)) \
            --input_tflite_files=$$(basename $(execpath //tensorflow/lite/micro/models:person_detect.tflite)) \
            --output_dir=$$(dirname $@)
    """,
    tools = [
        ":generate_micro_mutable_op_resolver_from_model",
    ],
)

tflm_cc_library(
    name = "person_detect_op_resolver",
    hdrs = [
        ":generate_person_detect_op_resolver_hdr",
    ],
    deps = [
        "//tensorflow/lite/micro:op_resolvers",
    ],
)

tflm_cc_test(
    name = "micro_mutable_op_resolver_test",
    srcs = [
        "micro_mutable_op_resolver_test.cc",
    ],
    deps = [
        ":person_detect_op_resolver",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/micro/examples/person_detection:person_detect_model_data",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
