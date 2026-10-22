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
"""Environment-specific toolchain and simulator runner bindings for TFLM."""

load("@rules_cc//cc/toolchains:cc_toolchain.bzl", "cc_toolchain")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load(
    "//tensorflow/lite/micro/tools/toolchains/hexagon:hexagon_toolchain_config.bzl",
    "hexagon_toolchain_config",
)
load(
    "//tensorflow/lite/micro/tools/toolchains/xtensa:xtensa_toolchain_config.bzl",
    "xtensa_toolchain_config",
)

ARM_TOOLCHAIN_BIN_PREFIX = "@arm_none_eabi//:bin/"
ARM_TOOLCHAIN_FILES_PREFIX = "@arm_none_eabi//:"

RISCV_TOOLCHAIN_BIN_PREFIX = "@riscv_toolchain//:bin/"
RISCV_TOOLCHAIN_FILES_PREFIX = "@riscv_toolchain//:"
RISCV_LINKER_FILES = "@riscv_toolchain//:linker_files"

QEMU_ARM_DATA = []
QEMU_RISCV32_DATA = []
CORSTONE300_FVP_DATA = []
HEXAGON_RUNNER_DATA = ["@hexagon_toolchain//:bin/hexagon-sim"]

def tflm_xtensa_runner_data(core):
    """Returns data dependencies for the Xtensa simulator runner."""
    _ = core
    return ["@xtensa_toolchain//:bin/xt-run"]

def tflm_sh_runner(name, srcs, data = None):
    """Declares a test-only simulator runner script."""
    sh_binary(
        name = name,
        testonly = True,
        srcs = srcs,
        data = data or [],
        tags = ["manual"],
        visibility = ["//visibility:public"],
    )

def tflm_cc_toolchain(name, **kwargs):
    """Declares a C/C++ toolchain target."""
    cc_toolchain(
        name = name,
        **kwargs
    )

def tflm_arm_cpu_constraint_setting():
    """Declares and returns the constraint_setting label for Cortex-M CPUs."""
    native.constraint_setting(name = "cortex_m_cpu")
    return ":cortex_m_cpu"

def tflm_xtensa_toolchains(cores):
    """Registers Xtensa cc_toolchain, constraint, toolchain, and platform targets."""
    native.filegroup(name = "empty")
    for name, info in cores.items():
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
        tflm_cc_toolchain(
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

    native.constraint_setting(name = "xtensa_core")
    for name in cores.keys():
        native.constraint_value(
            name = name,
            constraint_setting = ":xtensa_core",
        )
        native.toolchain(
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
        native.platform(
            name = "%s_platform" % name,
            constraint_values = [
                "@platforms//os:none",
                "@platforms//cpu:xtensa",
                ":%s" % name,
            ],
        )

def tflm_hexagon_toolchains():
    """Registers Hexagon cc_toolchain, toolchain, and platform targets."""
    native.filegroup(name = "empty")
    hexagon_toolchain_config(
        name = "hexagon_v66_config",
        ar = "@hexagon_toolchain//:bin/ar",
        c_compiler = "@hexagon_toolchain//:bin/hexagon-clang",
        cpp_compiler = "@hexagon_toolchain//:bin/hexagon-clang++",
        cpu_version = "v66",
        strip = "@hexagon_toolchain//:bin/hexagon-strip",
    )
    tflm_cc_toolchain(
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
    native.toolchain(
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
    native.platform(
        name = "hexagon_platform",
        constraint_values = [
            "@platforms//os:none",
            "@platforms//cpu:hexagon",
        ],
    )
