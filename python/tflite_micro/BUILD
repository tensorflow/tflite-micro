load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_namespace.bzl", "py_namespace")
load(
    "//python:py_rules.bzl",
    "py_package",
    "py_wheel",
    "tflm_py_library",
    "tflm_py_test",
    "tflm_whl_test",
)
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "INCOMPATIBLE_WITH_WINDOWS",
    "tflm_cc_library",
)
load(
    "//tensorflow/lite/micro:extra_rules.bzl",
    "tflm_python_op_resolver_friends",
)
load("//tools:expand_stamp_vars.bzl", "expand_stamp_vars")

package(
    licenses = ["notice"],
)

package_group(
    name = "op_resolver_friends",
    packages = tflm_python_op_resolver_friends(),
)

tflm_cc_library(
    name = "compression_utils",
    hdrs = ["compression_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/micro:compression",
        "//tensorflow/lite/micro/schema:schema_fbs",
    ],
)

tflm_cc_library(
    name = "shared_library",
    hdrs = ["shared_library.h"],
    visibility = ["//visibility:public"],
)

tflm_cc_library(
    name = "python_ops_resolver",
    srcs = [
        "python_ops_resolver.cc",
    ],
    hdrs = [
        "python_ops_resolver.h",
    ],
    visibility = [
        ":op_resolver_friends",
        "//tensorflow/lite/micro/integration_tests:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_compatibility",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:micro_ops",
    ],
)

tflm_cc_library(
    name = "runtime_c_api",
    srcs = ["runtime_c_api.cc"],
    hdrs = ["runtime_c_api.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":compression_utils",
        ":python_ops_resolver",
        ":shared_library",
        "//tensorflow/lite/micro:micro_allocator",
        "//tensorflow/lite/micro:micro_common",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_resource_variable",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/schema:schema_fbs",
        "@flatbuffers",
    ],
    alwayslink = True,
)

cc_binary(
    name = "_runtime.so",
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:private"],
    deps = [
        ":runtime_c_api",
    ],
)

tflm_py_library(
    name = "runtime",
    srcs = [
        "runtime.py",
    ],
    data = [
        ":_runtime.so",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        "//tensorflow/lite/micro/python:schema_py",
    ],
)

tflm_py_test(
    name = "runtime_test",
    srcs = ["runtime_test.py"],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        ":runtime",
        requirement("ai-edge-litert"),
        requirement("numpy"),
        "//tensorflow/lite/micro/examples/recipes:add_four_numbers",
        "//tensorflow/lite/micro/testing:generate_test_models_lib",
    ],
)

tflm_py_test(
    name = "test_compression_unsupported",
    srcs = ["test_compression_unsupported.py"],
    data = [
        "sine_float.tflite",
    ],
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    # Only compatible when compression is NOT enabled
    target_compatible_with = select({
        "//:with_compression_enabled": ["@platforms//:incompatible"],
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":runtime",
        "//tensorflow/lite/micro/compression:model_editor",
    ],
)

tflm_py_library(
    name = "postinstall_check",
    srcs = [
        "postinstall_check.py",
    ],
    data = [
        "sine_float.tflite",
    ],
)

# Generate a version attribute, imported as tflite_micro.__version__, using
# stamp (a.k.a. workspace status) variables.
expand_stamp_vars(
    name = "version",
    out = "_version.py",
    template = "_version.py.in",
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
        "tensorflow.lite.micro.compression",
        "tensorflow.lite.micro.python",
        "tensorflow.lite.micro.tools",
    ],
    deps = [
        ":postinstall_check",
        ":runtime",
        ":version",
        "//tensorflow/lite/micro/compression",
        "//tensorflow/lite/micro/tools:tflite_flatbuffer_align",
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
    tags = ["manual"],
    deps = [
        ":files_to_package",
    ],
)

expand_stamp_vars(
    name = "description_file",
    out = "README.pypi.md",
    template = "README.pypi.md.in",
)

# Building the :whl or its descendants requires the following build setting to
# supply the Python compatibility tags for the wheel metadata.
string_flag(
    name = "compatibility_tag",
    build_setting_default = "local",
    values = [
        "py3_none_manylinux_2_28_x86_64",
        "local",
    ],
)

config_setting(
    name = "py3_none_manylinux_2_28_x86_64",
    flag_values = {
        ":compatibility_tag": "py3_none_manylinux_2_28_x86_64",
    },
)

config_setting(
    name = "local",
    flag_values = {
        ":compatibility_tag": "local",
    },
)

py_wheel(
    name = "whl",
    # This macro yields additional targets:
    #
    # - whl.dist: build a properly named file under whl_dist/
    #
    abi = "none",
    description_file = ":description_file",
    distribution = "tflite_micro",
    platform = select({
        ":local": "any",
        ":py3_none_manylinux_2_28_x86_64": "manylinux_2_28_x86_64",
    }),
    python_tag = "py3",
    requires = [
        "flatbuffers",
        "numpy",
        "pyyaml",
    ],
    stamp = 1,  # 1 == always stamp
    strip_path_prefixes = [package_name()],
    summary = "TensorFlow Lite for Microcontrollers",
    version = "{BUILD_EMBED_LABEL}.dev{STABLE_GIT_COMMIT_TIME}",
    deps = [
        ":namespace",
    ],
)

tflm_whl_test(
    name = "whl_test",
    size = "large",
    srcs = [
        "whl_test.sh",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notap",  # See http://b/294278650#comment4 for more details.
        "noubsan",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)
