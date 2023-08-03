load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("//python:py_namespace.bzl", "py_namespace")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)
load(
    "//tensorflow:extra_rules.bzl",
    "tflm_python_op_resolver_friends",
)

package(
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "op_resolver_friends",
    packages = tflm_python_op_resolver_friends(),
)

cc_library(
    name = "python_ops_resolver",
    srcs = [
        "python_ops_resolver.cc",
    ],
    hdrs = [
        "python_ops_resolver.h",
    ],
    copts = micro_copts(),
    visibility = [
        ":op_resolver_friends",
        "//tensorflow/lite/micro/integration_tests:__subpackages__",
        "//tensorflow/lite/micro/python/interpreter/src:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

pybind_extension(
    name = "_runtime",
    # target = _runtime.so because pybind_extension() appends suffix
    srcs = [
        "_runtime.cc",
        "interpreter_wrapper.cc",
        "interpreter_wrapper.h",
        "numpy_utils.cc",
        "numpy_utils.h",
        "pybind11_lib.h",
        "python_utils.cc",
        "python_utils.h",
        "shared_library.h",
    ],
    deps = [
        ":python_ops_resolver",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "@numpy_cc_deps//:cc_headers",
    ],
)

py_library(
    name = "runtime",
    srcs = [
        "runtime.py",
    ],
    data = [
        ":_runtime.so",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        "//tensorflow/lite/tools:flatbuffer_utils",
    ],
)

py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    python_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols in _runtime.so
        "noubsan",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        ":runtime",
        "//tensorflow/lite/micro/testing:generate_test_models_lib",
    ],
)

py_library(
    name = "postinstall_check",
    srcs = [
        "postinstall_check.py",
    ],
    data = [
        "sine_float.tflite",
    ],
)

# Collect the `deps` and their transitive dependences together into a set of
# files to package. The files retain their full path relative to the workspace
# root, which determines the subpackage path at which they're located within
# the Python distribution package.
py_package(
    name = "files_to_package",

    # Only Python subpackage paths matching the following prefixes are included
    # in the files to package. This avoids packaging, e.g., numpy, which is a
    # transitive dependency of the tflm runtime target.  This list may require
    # modification when adding, directly or indirectly, `deps` from other paths
    # in the tflm tree.
    packages = [
        "python.tflite_micro",
        "tensorflow.lite.python",
        "tensorflow.lite.tools.flatbuffer_utils",
    ],
    deps = [
        ":postinstall_check",
        ":runtime",
    ],
)

# Relocate `deps` underneath the given Python package namespace, otherwise
# maintaining their full paths relative to the workspace root.
#
# For example:
#   ${workspace_root}/example/hello.py
# becomes:
#   namespace.example.hello
#
# Place `init` at the root of the namespace, regardless of `init`'s path in the
# source tree.
py_namespace(
    name = "namespace",
    init = "__init__.py",
    namespace = "tflite_micro",
    deps = [
        ":files_to_package",
    ],
)

py_wheel(
    name = "whl",
    distribution = "tflite_micro",
    requires = [
        "flatbuffers",
        "numpy",
        "tensorflow",
    ],
    strip_path_prefixes = [package_name()],
    version = "0.1.0",
    deps = [
        ":namespace",
    ],
)

sh_test(
    name = "whl_test",
    srcs = [
        "whl_test.sh",
    ],
    args = [
        "$(location :whl)",
    ],
    data = [
        ":whl",
    ],
    tags = [
        "notap",  # See http://b/294278650#comment4 for more details.
    ],
)
