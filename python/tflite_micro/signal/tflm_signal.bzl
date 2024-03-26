"""Build rule for wrapping a custom TF OP from .cc to python."""

load("@rules_python//python:defs.bzl", "py_library")

# TODO(b/286890280): refactor to be more generic build target for any custom OP
def py_tflm_signal_library(
        name,
        srcs = [],
        deps = [],
        visibility = None,
        cc_op_defs = [],
        cc_op_kernels = []):
    """Creates build rules for signal ops as shared libraries.

    Defines three targets:
    <name>
        Python library that exposes all ops defined in `cc_op_defs` and `py_srcs`.
    <name>_cc
        C++ library that registers any c++ ops in `cc_op_defs`, and includes the
        kernels from `cc_op_kernels`.
    ops/_<name>.so
        Shared library exposing the <name>_cc library.
    Args:
      name: The name for the python library target build by this rule.
      srcs: Python source files for the Python library.
      deps: Dependencies for the Python library.
      visibility: Visibility for the Python library.
      cc_op_defs: A list of c++ libraries containing REGISTER_OP definitions.
      cc_op_kernels: A list of c++ targets containing kernels that are used
          by the Python library.
    """
    binary_path = "ops"
    if srcs:
        binary_path_end_pos = srcs[0].rfind("/")
        binary_path = srcs[0][0:binary_path_end_pos]
    binary_name = binary_path + "/_" + cc_op_kernels[0][1:] + ".so"
    if cc_op_defs:
        binary_name = "ops/_" + name + ".so"
        library_name = name + "_cc"
        native.cc_library(
            name = library_name,
            copts = select({
                "//conditions:default": ["-pthread"],
            }),
            alwayslink = 1,
            deps =
                cc_op_defs +
                cc_op_kernels +
                ["@tensorflow_cc_deps//:cc_library"] +
                select({"//conditions:default": []}),
        )

        native.cc_binary(
            name = binary_name,
            copts = select({
                "//conditions:default": ["-pthread"],
            }),
            linkshared = 1,
            linkopts = [],
            deps = [
                ":" + library_name,
                "@tensorflow_cc_deps//:cc_library",
            ] + select({"//conditions:default": []}),
        )

    py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        visibility = visibility,
        data = [":" + binary_name],
        deps = deps,
    )

# A rule to build a TensorFlow OpKernel.
def tflm_signal_kernel_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        alwayslink = 1):
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        alwayslink = alwayslink,
    )
