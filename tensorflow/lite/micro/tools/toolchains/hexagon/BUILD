# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")
load(":hexagon_toolchain_config.bzl", "hexagon_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

hexagon_toolchain_config(
    name = "hexagon_v66_config",
    ar = "@hexagon_toolchain//:bin/ar",
    c_compiler = "@hexagon_toolchain//:bin/hexagon-clang",
    cpp_compiler = "@hexagon_toolchain//:bin/hexagon-clang++",
    cpu_version = "v66",
    strip = "@hexagon_toolchain//:bin/hexagon-strip",
)

cc_toolchain(
    name = "cc_toolchain_hexagon",
    all_files = "@hexagon_toolchain//:everything",
    ar_files = "@hexagon_toolchain//:everything",
    as_files = "@hexagon_toolchain//:compiler_files",
    compiler_files = "@hexagon_toolchain//:compiler_files",
    dwp_files = ":empty",
    linker_files = "@hexagon_toolchain//:linker_files",
    objcopy_files = ":empty",
    strip_files = "@hexagon_toolchain//:everything",
    supports_param_files = 0,
    toolchain_config = ":hexagon_v66_config",
    toolchain_identifier = "hexagon-v66-toolchain",
)

toolchain(
    name = "hexagon_toolchain",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:none",
        "@platforms//cpu:hexagon",
    ],
    toolchain = ":cc_toolchain_hexagon",
    toolchain_type = "@rules_cc//cc:toolchain_type",
)

platform(
    name = "hexagon_platform",
    constraint_values = [
        "@platforms//os:none",
        "@platforms//cpu:hexagon",
    ],
)

cc_test(
    name = "smoke_test",
    srcs = ["test_smoke.cc"],
)
