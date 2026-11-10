load(
    "//python:py_rules.bzl",
    "tflm_py_binary",
    "tflm_py_library",
    "tflm_py_test",
)
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "INCOMPATIBLE_WITH_WINDOWS",
    "tflm_cc_test",
)
load(
    "//tensorflow/lite/micro/tools/toolchains:platform_suites.bzl",
    "tflm_platform_target",
)
load(
    "//third_party/flatbuffers:build_defs.bzl",
    "flatbuffer_cc_library",
    "flatbuffer_py_library",
)
load("view_tests.bzl", "generate_view_tests")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

tflm_py_library(
    name = "compression",
    srcs = ["__init__.py"],
    deps = [
        ":compress_lib",
        ":spec",
        ":spec_builder",
    ],
)

flatbuffer_cc_library(
    # Generates the header-only library "metadata_generated.h", used to read
    # the metadata flatbuffer.
    name = "metadata_cc",
    srcs = ["metadata.fbs"],
)

tflm_cc_test(
    name = "metadata_test_cc",
    size = "small",
    srcs = ["metadata_test.cc"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":metadata_cc",
        "//tensorflow/lite/micro:hexdump",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:span",
        "//tensorflow/lite/micro/testing:micro_test",
        "//third_party/flatbuffers:runtime_cc",
    ],
)

tflm_platform_target(
    name = "metadata_test_cc_compression",
    size = "small",
    profile = "compression",
    target = ":metadata_test_cc",
)

flatbuffer_py_library(
    # Generates the Python module "metadata_py_generated", used to read the
    # metadata flatbuffer.
    name = "metadata_py",
    srcs = ["metadata.fbs"],
)

tflm_py_test(
    name = "metadata_test_py",
    size = "small",
    srcs = ["metadata_test.py"],
    main = "metadata_test.py",
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        "metadata_py",
        "//third_party/flatbuffers:runtime_py",
    ],
)

tflm_py_library(
    name = "compress_lib",
    srcs = [
        "compress.py",
    ],
    deps = [
        ":compressor",
        ":constant_inputs",
        ":decode_insert",
        ":huffman",
        ":lut",
        ":pruning",
        ":spec",
        "//tensorflow/lite/micro/tools:model_editor",
        "//tensorflow/lite/micro/tools:tflite_flatbuffer_align",
    ],
)

tflm_py_binary(
    name = "compress",
    srcs = [
        "compress.py",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":compress_lib",
    ],
)

tflm_py_test(
    name = "compress_test",
    size = "small",
    srcs = [
        "compress_test.py",
    ],
    tags = [
        "noasan",
        "nomsan",  # Sanitizer symbols don't work with Python extension libs
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":compress_lib",
        ":compressor",
        ":decode_insert",
        ":spec",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_library(
    name = "verify",
    srcs = ["verify.py"],
    deps = [
        "//python/tflite_micro:runtime",
        "//tensorflow/lite/micro/tools:model_editor",
        "//tensorflow/lite/micro/tools:tensor_type",
        "//third_party/py/numpy",
    ],
)

tflm_py_test(
    name = "verify_test",
    size = "small",
    srcs = ["verify_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":verify",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_test(
    name = "compression_integration_test",
    size = "small",
    srcs = ["compression_integration_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":compress_lib",
        ":decode_insert",
        ":spec",
        ":verify",
        "//python/tflite_micro:runtime",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_test(
    name = "proprietary_integration_test",
    size = "small",
    srcs = ["proprietary_integration_test.py"],
    tags = [
        "manual",
        "noasan",
        "nomsan",
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":compress_lib",
        ":spec",
        ":verify",
        "//third_party/py/yaml",
    ],
)

tflm_py_test(
    name = "proprietary_integration_smoke_test",
    size = "small",
    srcs = [
        "compression_integration_test.py",
        "proprietary_integration_smoke_test.py",
        "proprietary_integration_test.py",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notap",
        "noubsan",
    ],
    target_compatible_with = select({
        "//:with_compression_enabled": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":compress_lib",
        ":decode_insert",
        ":spec",
        ":verify",
        "//python/tflite_micro:runtime",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
        "//third_party/py/yaml",
    ],
)

tflm_py_library(
    name = "spec",
    srcs = ["spec.py"],
    deps = [
        "//third_party/py/yaml",
    ],
)

tflm_py_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.py"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":spec",
    ],
)

tflm_py_library(
    name = "spec_builder",
    srcs = ["spec_builder.py"],
    deps = [
        ":spec",
    ],
)

tflm_py_test(
    name = "spec_builder_test",
    size = "small",
    srcs = ["spec_builder_test.py"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":spec",
        ":spec_builder",
    ],
)

tflm_py_library(
    name = "decode",
    srcs = ["decode.py"],
)

tflm_py_test(
    name = "decode_test",
    size = "small",
    srcs = ["decode_test.py"],
    deps = [
        ":decode",
    ],
)

tflm_py_library(
    name = "compressor",
    srcs = ["compressor.py"],
    deps = [
        ":decode",
        ":spec",
        "//tensorflow/lite/micro/tools:model_editor",
    ],
)

tflm_py_library(
    name = "constant_inputs",
    srcs = ["constant_inputs.py"],
    deps = [
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
    ],
)

tflm_py_test(
    name = "constant_inputs_test",
    size = "small",
    srcs = ["constant_inputs_test.py"],
    tags = [
        "noasan",
        "nomsan",  # Sanitizer symbols don't work with Python extension libs
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":constant_inputs",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_library(
    name = "lut",
    srcs = ["lut.py"],
    deps = [
        ":compressor",
        ":decode",
        ":spec",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_test(
    name = "lut_test",
    size = "small",
    srcs = ["lut_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":compressor",
        ":decode",
        ":lut",
        ":spec",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_library(
    name = "huffman",
    srcs = ["huffman.py"],
    deps = [
        ":compressor",
        ":decode",
        ":spec",
        "//tensorflow/lite/micro/tools:model_editor",
    ],
)

tflm_py_library(
    name = "pruning",
    srcs = ["pruning.py"],
    deps = [
        ":compressor",
        ":decode",
        ":spec",
        "//tensorflow/lite/micro/tools:model_editor",
    ],
)

tflm_py_library(
    name = "decode_insert",
    srcs = ["decode_insert.py"],
    deps = [
        ":compressor",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
    ],
)

tflm_py_test(
    name = "decode_insert_test",
    size = "small",
    srcs = ["decode_insert_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":compressor",
        ":decode",
        ":decode_insert",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_test(
    name = "decode_insert_runtime_test",
    size = "small",
    srcs = ["decode_insert_runtime_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":decode_insert",
        ":lut",
        ":spec",
        "//python/tflite_micro:runtime",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_library(
    name = "propose_spec_lib",
    srcs = ["propose_spec.py"],
    deps = [
        ":compress_lib",
        ":constant_inputs",
        ":decode",
        ":lut",
        ":spec",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//tensorflow/lite/micro/tools:tensor_type",
        "//third_party/py/numpy",
    ],
)

tflm_py_binary(
    name = "propose_spec",
    srcs = ["propose_spec.py"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":propose_spec_lib",
    ],
)

tflm_py_test(
    name = "propose_spec_test",
    size = "small",
    srcs = ["propose_spec_test.py"],
    tags = [
        "noasan",
        "nomsan",  # Sanitizer symbols don't work with Python extension libs
        "noubsan",
    ],
    deps = [
        ":compress_lib",
        ":lut",
        ":propose_spec_lib",
        ":spec",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:model_editor",
        "//third_party/py/numpy",
    ],
)

tflm_py_library(
    name = "view_lib",
    srcs = [
        "view.py",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":metadata_py",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:tensor_type",
        "//third_party/py/numpy",
    ],
)

tflm_py_binary(
    name = "view",
    srcs = [
        "view.py",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":view_lib",
    ],
)

generate_view_tests(
    targets = [
        "//tensorflow/lite/micro/models:keyword_scrambled.tflite",
        "//tensorflow/lite/micro/models:keyword_scrambled_8bit.tflite",
        "//tensorflow/lite/micro/models:person_detect.tflite",
        "//tensorflow/lite/micro/models:person_detect_vela.tflite",
    ],
)
