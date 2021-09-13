def micro_copts():
    return [
        "-Wall",
        "-Werror",
        "-DFLATBUFFERS_LOCALE_INDEPENDENT=0",
    ]

def generate_cc_arrays(name, src, out, visibility = None):
    native.genrule(
        name = name,
        srcs = [
            src,
        ],
        outs = [
            out,
        ],
        cmd = "$(location //tensorflow/lite/micro/tools:generate_cc_arrays) $@ $<",
        tools = ["//tensorflow/lite/micro/tools:generate_cc_arrays"],
        visibility = visibility,
    )

def tflm_kernel_accelerated_cc_library(
        name,
        default_srcs = [],
        accelerated_srcs = [],
        accelerated_hdrs = [],
        deps = [],
        **kwargs):
    """Creates a cc_library with the accelerated target sources.

    This resembles the behavior of the Makefile where specialize_files.py will
    filter out all reference ops sources in favor of their accelerated
    counterpart.

    Args:
      name: The name of the target.
      default_srcs: The non-accelerated TFLM kernel source files.
      accelerated_srcs: The platform accelerated TFLM kerenel source files.
      accelerated_hdrs: The platform accelerated TFLM kernel headers.
      deps: The library's dependencies.
      **kwargs: Arguments passed into the cc_library.
    """

    accelerated_srcs_filenames = [src.split("/")[-1] for src in accelerated_srcs]
    srcs = accelerated_srcs

    # Filter out all reference ops that have accelerated implementations.
    for src in default_srcs:
        if src not in accelerated_srcs_filenames:
            srcs.append(src)

    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = accelerated_hdrs,
        deps = deps,
        **kwargs
    )