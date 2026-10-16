"""Build rules for TensorFlow custom op plugins using the TF C API."""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//python:py_rules.bzl", "tflm_py_library")

def py_tflm_signal_library(
        name,
        srcs = [],
        deps = [],
        visibility = None,
        cc_op_defs = [],
        cc_op_kernels = []):
    """Creates build rules for signal custom ops as shared libraries."""
    binary_path = "ops"
    if srcs:
        binary_path_end_pos = srcs[0].rfind("/")
        binary_path = srcs[0][0:binary_path_end_pos]

    binary_name = binary_path + "/_" + (cc_op_kernels[0][1:] if not cc_op_defs else name) + ".so"
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    })

    if cc_op_defs:
        library_name = name + "_cc"
        cc_library(
            name = library_name,
            alwayslink = 1,
            copts = ["-pthread"],
            target_compatible_with = target_compatible_with,
            deps = cc_op_defs + cc_op_kernels,
        )
        cc_binary(
            name = binary_name,
            copts = ["-pthread"],
            linkopts = select({
                "@platforms//os:macos": ["-Wl,-undefined,dynamic_lookup"],
                "//conditions:default": ["-Wl,--unresolved-symbols=ignore-in-object-files"],
            }),
            linkshared = 1,
            target_compatible_with = target_compatible_with,
            deps = [
                ":" + library_name,
            ],
        )

    tflm_py_library(
        name = name,
        srcs = srcs,
        data = [":" + binary_name] if cc_op_defs else [],
        target_compatible_with = target_compatible_with,
        visibility = visibility,
        deps = deps,
    )

def tflm_signal_kernel_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        alwayslink = 1):
    cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        alwayslink = alwayslink,
        copts = copts,
        deps = deps,
    )
