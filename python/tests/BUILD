cc_test(
    name = "cc_deps_link_test",
    size = "small",
    srcs = [
        "cc_deps_link_test.cc",
    ],
    copts = [
        "-std=c++14",
        # Compiling against upstream Tensorflow headers requires -std=c++14
        # due (at least) to its dependency on Eigen, which raises the error
        # "This compiler appears to be too told to be supported by Eigen"
        # via the preprocessor, as well as several compilation errors.
        #
        # TODO(b/271210933): Harmonize this with a global solution
        # throughout the repository. Presumably this same issue will be
        # encountered by other code compiling against upstream Tensorflow,
        # such as custom operator implementations.
    ],
    deps = [
        "@tensorflow_cc_deps//:cc_library",
    ],
)
