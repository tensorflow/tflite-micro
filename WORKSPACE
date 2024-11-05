# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

workspace(name = "tflite_micro")

load("//tensorflow:workspace.bzl", "workspace")

workspace()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# compile_commands.json generator
http_archive(
    name = "hedron_compile_commands",
    sha256 = "bacabfe758676fdc19e4bea7c4a3ac99c7e7378d259a9f1054d341c6a6b44ff6",
    strip_prefix = "bazel-compile-commands-extractor-1266d6a25314d165ca78d0061d3399e909b7920e",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/1266d6a25314d165ca78d0061d3399e909b7920e.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

_rules_python_version = "0.26.0"

http_archive(
    name = "rules_python",
    sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
    strip_prefix = "rules_python-{}".format(_rules_python_version),
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/{}.tar.gz".format(_rules_python_version),
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Read the Python package dependencies of the build environment. To modify
# them, see //third_party:python_requirements.in.
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "tflm_pip_deps",
    requirements_lock = "//third_party:python_requirements.txt",
)

# Create repositories for each Python package dependency.
load("@tflm_pip_deps//:requirements.bzl", "install_deps", "requirement")

install_deps()

http_archive(
    name = "pybind11_bazel",
    sha256 = "a185aa68c93b9f62c80fcb3aadc3c83c763854750dc3f38be1dadcb7be223837",
    strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec",
    strip_prefix = "pybind11-2.10.0",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(
    name = "local_config_python",
    python_version = "3",
)

load("//python:py_pkg_cc_deps.bzl", "py_pkg_cc_deps")

py_pkg_cc_deps(
    name = "numpy_cc_deps",
    includes = ["numpy/_core/include"],
    pkg = requirement("numpy"),
)

py_pkg_cc_deps(
    name = "tensorflow_cc_deps",
    includes = ["tensorflow/include"],
    libs = ["tensorflow/libtensorflow_framework.so.2"],
    pkg = requirement("tensorflow"),
)

# Optimized kernel deps
http_archive(
    name = "nnlib_hifi4",
    build_file = "@tflite_micro//third_party/xtensa/nnlib_hifi4:nnlib_hifi4.BUILD",
    integrity = "sha256-ulZ+uY4dRsbDUMZbZtD972eghclWQrqYRb0Y4Znfyyc=",
    strip_prefix = "nnlib-hifi4-34f5f995f28d298ae2b6e2ba6e76c32a5cb34989",
    urls = ["https://github.com/foss-xtensa/nnlib-hifi4/archive/34f5f995f28d298ae2b6e2ba6e76c32a5cb34989.zip"],
)
