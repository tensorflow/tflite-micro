# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_binary",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "model_settings",
    srcs = [
        "model_settings.cc",
    ],
    hdrs = [
        "model_settings.h",
    ],
)

generate_cc_arrays(
    name = "generated_no_person_bmp_cc",
    src = "testdata/no_person.bmp",
    out = "testdata/no_person_image_data.cc",
)

generate_cc_arrays(
    name = "generated_no_person_bmp_hdr",
    src = "testdata/no_person.bmp",
    out = "testdata/no_person_image_data.h",
)

generate_cc_arrays(
    name = "generated_person_bmp_cc",
    src = "testdata/person.bmp",
    out = "testdata/person_image_data.cc",
)

generate_cc_arrays(
    name = "generated_person_bmp_hdr",
    src = "testdata/person.bmp",
    out = "testdata/person_image_data.h",
)

tflm_cc_library(
    name = "person_detect_model_data",
    srcs = [
        "//tensorflow/lite/micro/models:generated_person_detect_model_cc",
    ],
    hdrs = [
        "//tensorflow/lite/micro/models:generated_person_detect_model_hdr",
    ],
)

tflm_cc_library(
    name = "simple_images_test_data",
    srcs = [
        ":generated_no_person_bmp_cc",
        ":generated_person_bmp_cc",
    ],
    hdrs = [
        ":generated_no_person_bmp_hdr",
        ":generated_person_bmp_hdr",
    ],
    deps = [
        ":model_settings",
    ],
)

tflm_cc_test(
    name = "person_detection_test",
    srcs = ["person_detection_test.cc"],
    deps = [
        ":model_settings",
        ":person_detect_model_data",
        ":simple_images_test_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

tflm_cc_library(
    name = "image_provider",
    srcs = [
        "image_provider.cc",
    ],
    hdrs = [
        "image_provider.h",
    ],
    deps = [
        ":model_settings",
        "//tensorflow/lite/c:common",
    ],
)

tflm_cc_test(
    name = "image_provider_test",
    srcs = [
        "image_provider_test.cc",
    ],
    deps = [
        ":image_provider",
        ":model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_library(
    name = "detection_responder",
    srcs = [
        "detection_responder.cc",
    ],
    hdrs = [
        "detection_responder.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_log",
    ],
)

tflm_cc_test(
    name = "detection_responder_test",
    srcs = [
        "detection_responder_test.cc",
    ],
    deps = [
        ":detection_responder",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

tflm_cc_binary(
    name = "person_detection",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    deps = [
        ":detection_responder",
        ":image_provider",
        ":model_settings",
        ":person_detect_model_data",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
