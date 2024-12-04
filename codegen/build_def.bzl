""" Build rule for generating ML inference code from TFLite model. """

load("//tensorflow/lite/micro:build_def.bzl", "tflm_cc_library")

def tflm_inference_library(
        name,
        tflite_model,
        visibility = None):
    """Creates a C++ library capable of performing ML inference of the provided
    model.

    Args:
      name: Target name.
      tflite_model: TFLite Model to generate inference from.
      visibility: Visibility for the C++ library.
    """
    generated_target = name + "_gen"
    native.genrule(
        name = generated_target,
        srcs = [tflite_model],
        outs = [
            name + ".h",
            name + ".cc",
            name + ".log",
        ],
        tools = ["//codegen:code_generator"],
        cmd = """
            # code_generator (partially because it uses Tensorflow) outputs
            # much noise to the console. Intead, write output to a logfile to
            # prevent noise in the error-free bazel output.
            NAME=%s
            LOGFILE=$(RULEDIR)/$$NAME.log
            $(location //codegen:code_generator) \
                    --model=$< \
                    --output_dir=$(RULEDIR) \
                    --output_name=$$NAME \
                    >$$LOGFILE 2>&1
        """ % name,
        visibility = ["//visibility:private"],
    )

    tflm_cc_library(
        name = name,
        hdrs = [name + ".h"],
        srcs = [name + ".cc"],
        deps = [
            generated_target,
            "//codegen/runtime:micro_codegen_context",
            "//tensorflow/lite/c:common",
            "//tensorflow/lite/c:c_api_types",
            "//tensorflow/lite/kernels/internal:compatibility",
            "//tensorflow/lite/micro/kernels:micro_ops",
            "//tensorflow/lite/micro:micro_common",
            "//tensorflow/lite/micro:micro_context",
        ],
        target_compatible_with = select({
            "//conditions:default": [],
            "//:with_compression_enabled": ["@platforms//:incompatible"],
        }),
        visibility = visibility,
    )
