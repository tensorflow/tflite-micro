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
HEXAGON_CONSTRAINTS = [
    "@platforms//os:none",
    "@platforms//cpu:hexagon",
]
XTENSA_CONSTRAINTS = [
    "@platforms//os:none",
    "@platforms//cpu:xtensa",
]

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

TFLM_PLATFORM_CONFIGS = {
    "compression": {
        "baremetal": False,
        "flags": [
            "--//:with_compression=true",
        ],
        "mode": "test",
        "parents": ["@platforms//host"],
        "runner": None,
    },
    "corstone300": {
        "baremetal": True,
        "default_timeout": "long",
        "flags": [
            "--//tensorflow/lite/micro:bsp=//tensorflow/lite/micro/tools/toolchains/arm/corstonefvp:corstone_bsp",
            "--//tensorflow/lite/micro/kernels:optimized_kernels=cmsis_nn_ethos_u",
            "--//tensorflow/lite/micro/kernels:cmsis_nn_dep=@cmsis_nn//:cmsis_nn",
            "--//tensorflow/lite/micro/kernels:ethos_u_dep=@ethos_u_core_driver//:ethos_u_core_driver",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/arm:m55_baremetal_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/arm/corstonefvp:run_corstone300",
    },
    "cortex_m0": {
        "baremetal": True,
        "flags": [],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/arm:m0_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/arm:run_m0",
    },
    "cortex_m4": {
        "baremetal": True,
        "flags": [],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/arm:m4_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/arm:run_m4",
    },
    "cortex_m4_cmsis_nn": {
        "baremetal": True,
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=cmsis_nn",
            "--//tensorflow/lite/micro/kernels:cmsis_nn_dep=@cmsis_nn//:cmsis_nn",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/arm:m4_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/arm:run_m4",
    },
    "cortex_m55": {
        "baremetal": True,
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=cmsis_nn",
            "--//tensorflow/lite/micro/kernels:cmsis_nn_dep=@cmsis_nn//:cmsis_nn",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/arm:m55_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/arm:run_m55",
    },
    "cortex_m55_ref": {
        "baremetal": True,
        "flags": [],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/arm:m55_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/arm:run_m55",
    },
    "hexagon": {
        "baremetal": True,
        "default_timeout": "long",
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=hexagon",
            "--//tensorflow/lite/micro/kernels:hexagon_tflm_core_dep=@hexagon_toolchain//:hexagon_tflm_core",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/hexagon:hexagon_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/hexagon:run_hexagon",
    },
    "riscv32": {
        "baremetal": True,
        "flags": [
            "--//tensorflow/lite/micro:bsp=//tensorflow/lite/micro/tools/toolchains/riscv:riscv_bsp",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/riscv:riscv32_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/riscv:run_riscv32",
    },
    "xtensa_fusion_f1": {
        "baremetal": True,
        "default_timeout": "long",
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=xtensa_fusion_f1",
            "--//tensorflow/lite/micro/kernels:xtensa_nnlib_hifi4_dep=@nnlib_hifi4//:nnlib_hifi4",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/xtensa:fusion_f1_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/xtensa:run_fusion_f1",
    },
    "xtensa_hifi3z": {
        "baremetal": True,
        "default_timeout": "long",
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=xtensa_hifi_3z",
            "--//tensorflow/lite/micro/kernels:xtensa_nnlib_hifi4_dep=@nnlib_hifi4//:nnlib_hifi4",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/xtensa:hifi3z_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/xtensa:run_hifi3z",
    },
    "xtensa_hifi5": {
        "baremetal": True,
        "default_timeout": "long",
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=xtensa_hifi_5",
            "--//tensorflow/lite/micro/kernels:xtensa_nnlib_hifi5_dep=@nnlib_hifi5//:nnlib_hifi5",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/xtensa:hifi5_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/xtensa:run_hifi5",
    },
    "xtensa_vision_p6": {
        "baremetal": True,
        "default_timeout": "long",
        "flags": [
            "--//tensorflow/lite/micro/kernels:optimized_kernels=xtensa_vision_p6",
            "--//tensorflow/lite/micro/kernels:xtensa_vision_p6_dep=@xi_tflmlib_vision_p6//:xi_tflmlib_vision_p6",
        ],
        "mode": "test",
        "parents": ["//tensorflow/lite/micro/tools/toolchains/xtensa:vision_p6_platform"],
        "runner": "//tensorflow/lite/micro/tools/toolchains/xtensa:run_vision_p6",
    },
}

def tflm_declare_test_platforms(name = "test_platform"):
    """Instantiates `<profile>_<name>` targets for all active profiles.

    Args:
      name: Suffix for generated platform targets.
    """
    for profile, cfg in TFLM_PLATFORM_CONFIGS.items():
        if cfg.get("mode") == "off":
            continue
        native.platform(
            name = "%s_%s" % (profile, name),
            flags = cfg.get("flags", []),
            parents = cfg["parents"],
            visibility = ["//visibility:public"],
        )
