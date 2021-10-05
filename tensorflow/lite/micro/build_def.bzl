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

def tflm_kernel_cc_library(
        name,
        srcs = [],
        hdrs = [],
        accelerated_srcs = {},
        deps = [],
        **kwargs):
    """Creates a cc_library with the optional accelerated target sources.

    Note:
      Bazel macros cannot evaluate a select() statement. Therefore, the accelerated_srcs and
      accelerated_hdrs are passed as a dictionary, and the select statement is generated from the
      supplied dictionary.

    Args:
      name: The name of the target.
      srcs: The non-accelerated TFLM kernel source files.
      hdrs: The non-accelerated TFLM kernel header files.
      accelerated_srcs: A dictionary organized as {target: accelerated tflm kernel sources}.
      deps: The library's dependencies.
      **kwargs: Arguments passed into the cc_library.
    """

    all_srcs = {
        "//conditions:default": srcs,
    }

    all_hdrs = {
        "//conditions:default": hdrs,
    }

    # Identify all of the sources for each target. This ends up creating a dictionary for both the
    # sources and headers that looks like the following:
    # {
    #   "target1" : [target1_srcs] + [reference_srcs that aren't accelerated],
    #   "target2" : [target2_srcs] + [reference_srcs that aren't accelerated],
    #   "//conditions:default": [reference_srcs]
    # }
    for target in accelerated_srcs:
        target_srcs = accelerated_srcs[target]
        target_src_filenames = [src.split("/")[-1] for src in target_srcs]
        all_target_srcs = target_srcs

        # Filter out all reference ops that have accelerated implementations.
        for src in srcs:
            if src not in target_src_filenames:
                all_target_srcs.append(src)

        all_srcs[target] = all_target_srcs

    native.cc_library(
        name = name,
        srcs = select(all_srcs),
        hdrs = hdrs,
        deps = deps,
        **kwargs
    )
