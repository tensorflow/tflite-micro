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
load(":xtensa_toolchain_config.bzl", "xtensa_toolchain_config")

package(default_visibility = ["//visibility:public"])

filegroup(name = "empty")

CORES = {
    "fusion_f1": {
        "arch_def": "-DHIFI3",
        "core": "P29A_FusionF1",
    },
    "hifi3z": {
        "arch_def": "-DHIFI3",
        "core": "P29A_HiFi3z",
    },
    "hifi5": {
        "arch_def": "-DHIFI5",
        "core": "PRD_H5_RDO_07_01_2022",
    },
    "vision_p6": {
        "arch_def": "-DVISION_P6",
        "core": "P29A_VP6",
    },
}

[
    xtensa_toolchain_config(
        name = "%s_config" % name,
        ar = "@xtensa_toolchain//:bin/ar",
        c_compiler = "@xtensa_toolchain//:bin/xt-clang",
        copts = [
            "--xtensa-core=%s" % info["core"],
            "-mcoproc",
            "-mlongcalls",
            info["arch_def"],
        ],
        core = info["core"],
        cpp_compiler = "@xtensa_toolchain//:bin/xt-clang++",
        name_alias = name,
        strip = "@xtensa_toolchain//:bin/xt-strip",
    )
    for name, info in CORES.items()
]

[
    cc_toolchain(
        name = "cc_toolchain_%s" % name,
        all_files = "@xtensa_toolchain//:everything",
        ar_files = "@xtensa_toolchain//:everything",
        as_files = "@xtensa_toolchain//:compiler_files",
        compiler_files = "@xtensa_toolchain//:compiler_files",
        dwp_files = ":empty",
        linker_files = "@xtensa_toolchain//:linker_files",
        objcopy_files = ":empty",
        strip_files = "@xtensa_toolchain//:everything",
        supports_param_files = 0,
        toolchain_config = ":%s_config" % name,
        toolchain_identifier = "xtensa-%s-toolchain" % name,
    )
    for name in CORES.keys()
]

constraint_setting(name = "xtensa_core")

[
    constraint_value(
        name = name,
        constraint_setting = ":xtensa_core",
    )
    for name in CORES.keys()
]

[
    toolchain(
        name = "xtensa_%s_toolchain" % name,
        exec_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ],
        target_compatible_with = [
            "@platforms//os:none",
            "@platforms//cpu:xtensa",
            ":%s" % name,
        ],
        toolchain = ":cc_toolchain_%s" % name,
        toolchain_type = "@rules_cc//cc:toolchain_type",
    )
    for name in CORES.keys()
]

[
    platform(
        name = "%s_platform" % name,
        constraint_values = [
            "@platforms//os:none",
            "@platforms//cpu:xtensa",
            ":%s" % name,
        ],
    )
    for name in CORES.keys()
]

cc_test(
    name = "smoke_test",
    srcs = ["test_smoke.cc"],
)
