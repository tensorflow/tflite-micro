"""Bazel macros wrapping @rules_python's py_* rules for use in this repository.

The tflm_py_library, tflm_py_test, and tflm_py_binary wrappers inject a
dependency on //:tflite_micro_shim, which synthesizes the "tflite_micro"
top-level package namespace at import time (see //:tflite_micro.py).

The shim is necessary because, under Bzlmod, the main repository's runfiles root
is the fixed name "_main" rather than the module name, so the "tflite_micro"
import prefix used throughout this repository's Python sources (e.g. `from
tflite_micro.tensorflow.lite... import x`) no longer resolves on its own as it
did under the legacy WORKSPACE, where the runfiles root took the workspace name.

Any py_* target whose code imports "tflite_micro.*" must depend on the shim.
These wrappers add that dependency automatically so it need not be repeated in
every target. Prefer them over the native py_library/py_test/py_binary for any
target under the tflite_micro namespace.
"""

load("@pybind11_bazel//:build_defs.bzl", _pybind_extension = "pybind_extension", _pybind_library = "pybind_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@rules_python//python:packaging.bzl", _py_package = "py_package", _py_wheel = "py_wheel")
load("@rules_python//python:pip.bzl", _compile_pip_requirements = "compile_pip_requirements")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//tensorflow/lite/micro:build_def.bzl", "INCOMPATIBLE_WITH_WINDOWS")

compile_pip_requirements = _compile_pip_requirements
py_package = _py_package
py_wheel = _py_wheel
pybind_library = _pybind_library

def tflm_oss_manual_tags():
    """Returns ["manual"] in OSS builds for targets requiring external TF/LiteRT deps."""
    return ["manual"]

def tflm_tf_shape_inference_copts():
    """Returns copts for shape_inference_utils when building with full TF."""
    return []

def tflm_tf_shape_inference_deps():
    """Returns extra deps for shape_inference_utils when building with full TF."""
    return []

def pybind_extension(name, **kwargs):
    if "manual" in kwargs.get("tags", []):
        _pybind_library(
            name = name + "_pybind_lib",
            **kwargs
        )
        cc_binary(
            name = name + ".so",
            linkshared = True,
            tags = kwargs.get("tags", []),
            deps = [":" + name + "_pybind_lib"],
        )
    else:
        _pybind_extension(name = name, **kwargs)

def tflm_py_library(deps = [], **kwargs):
    py_library(
        deps = deps + [
            "//:tflite_micro_shim",
        ],
        **kwargs
    )

def tflm_py_binary(deps = [], **kwargs):
    py_binary(
        deps = deps + [
            "//:tflite_micro_shim",
        ],
        **kwargs
    )

def tflm_py_test(deps = [], **kwargs):
    py_test(
        deps = deps + [
            "//:tflite_micro_shim",
        ],
        **kwargs
    )

def tflm_whl_test(name, srcs, **kwargs):
    sh_test(
        name = name,
        srcs = srcs,
        args = [
            "$(rootpath :whl)",
        ] + select({
            "@rules_python//python/config_settings:is_python_3.10": ["$(rootpath @python_3_10//:bin/python3)"],
            "@rules_python//python/config_settings:is_python_3.11": ["$(rootpath @python_3_11//:bin/python3)"],
            "@rules_python//python/config_settings:is_python_3.12": ["$(rootpath @python_3_12//:bin/python3)"],
            "@rules_python//python/config_settings:is_python_3.13": ["$(rootpath @python_3_13//:bin/python3)"],
            "@rules_python//python/config_settings:is_python_3.14": ["$(rootpath @python_3_14//:bin/python3)"],
        }),
        data = [
            ":whl",
            "@python_3_10//:bin/python3",
            "@python_3_11//:bin/python3",
            "@python_3_12//:bin/python3",
            "@python_3_13//:bin/python3",
            "@python_3_14//:bin/python3",
        ],
        **kwargs
    )

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

    if cc_op_defs:
        library_name = name + "_cc"
        cc_library(
            name = library_name,
            alwayslink = 1,
            copts = ["-pthread"],
            target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
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
            target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
            deps = [
                ":" + library_name,
            ],
        )

    tflm_py_library(
        name = name,
        srcs = srcs,
        data = [":" + binary_name] if cc_op_defs else [],
        target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
        visibility = visibility,
        deps = deps,
    )

def tflm_signal_op_test(
        name,
        srcs,
        data = [],
        deps = [],
        tags = [],
        target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
        **kwargs):
    """Creates TFLM and TFLite test targets for a signal custom op."""
    common_tags = tags + tflm_oss_manual_tags() + [
        "noasan",
        "nomsan",
        "noubsan",
    ]
    tflm_py_test(
        name = name,
        srcs = srcs,
        data = data,
        tags = common_tags,
        target_compatible_with = target_compatible_with,
        deps = deps,
        **kwargs
    )
    tflm_py_test(
        name = name + "_tflite",
        srcs = srcs,
        main = srcs[0],
        data = data + [
            "//signal/lite/register:pywrap_signal_ops.so",
        ],
        env = {"TFLM_SIGNAL_USE_TFLITE": "1"},
        tags = common_tags,
        target_compatible_with = target_compatible_with,
        deps = deps + [
            requirement("ai-edge-litert"),
        ],
        **kwargs
    )
