load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

cc_binary(
    name = "generate_detection_postprocess_flexbuffers_data",
    srcs = [
        "generate_detection_postprocess_flexbuffers_data.cc",
    ],
    deps = [
        "//third_party/flatbuffers",
    ],
)

cc_binary(
    name = "generate_circular_buffer_flexbuffers_data",
    srcs = [
        "generate_circular_buffer_flexbuffers_data.cc",
    ],
    deps = [
        "//third_party/flatbuffers",
    ],
)
