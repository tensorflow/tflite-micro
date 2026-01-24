def tflite_copts():
    """Defines common compile time flags for TFLite libraries."""
    return select({
        "@bazel_tools//src/conditions:windows": [
            "/DFARMHASH_NO_CXX_STRING",
            "/EHs-",  # -fno-exceptions
            "/GR-",
        ],
        "//conditions:default": [
            "-DFARMHASH_NO_CXX_STRING",
            "-Wno-sign-compare",
            "-Wno-unused-parameter",
            "-fno-exceptions",  # Exceptions are unused in TFLite.
        ],
    })
