cc_test(
    name = "cc_deps_link_test",
    size = "small",
    srcs = [
        "cc_deps_link_test.cc",
    ],
    tags = [
        "nomsan",  # avoid, because tensorflow library fails msan
    ],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)
