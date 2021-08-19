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
