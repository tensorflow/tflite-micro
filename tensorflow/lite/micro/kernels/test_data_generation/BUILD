package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_binary(
    name = "generate_detection_postprocess_flexbuffers_data",
    srcs = [
        "generate_detection_postprocess_flexbuffers_data.cc",
    ],
    deps = [
        "@flatbuffers",
    ],
)

cc_binary(
    name = "generate_circular_buffer_flexbuffers_data",
    srcs = [
        "generate_circular_buffer_flexbuffers_data.cc",
    ],
    deps = [
        "@flatbuffers",
    ],
)
