def tflite_copts():
    """Defines common compile time flags for TFLite libraries."""
    copts = [
        "-DFARMHASH_NO_CXX_STRING",
        "-Wno-sign-compare",
        "-Wno-unused-parameter",
        "-fno-exceptions",  # Exceptions are unused in TFLite.
    ]
    return copts
