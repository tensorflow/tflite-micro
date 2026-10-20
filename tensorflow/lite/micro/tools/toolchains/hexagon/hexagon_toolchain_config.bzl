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
"""Qualcomm Hexagon toolchain configuration rule."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "flag_group",
    "flag_set",
    "tool",
    "variable_with_value",
)
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _hexagon_toolchain_config_impl(ctx):
    if ctx.attr.c_compiler_path:
        c_compiler_tool = [tool(path = ctx.attr.c_compiler_path)]
    elif ctx.file.c_compiler:
        c_compiler_tool = [tool(tool = ctx.file.c_compiler)]
    else:
        fail("c_compiler or c_compiler_path must be specified")

    if ctx.attr.cpp_compiler_path:
        cpp_compiler_tool = [tool(path = ctx.attr.cpp_compiler_path)]
    elif ctx.file.cpp_compiler:
        cpp_compiler_tool = [tool(tool = ctx.file.cpp_compiler)]
    else:
        fail("cpp_compiler or cpp_compiler_path must be specified")

    if ctx.attr.ar_path:
        ar_tool = [tool(path = ctx.attr.ar_path)]
    elif ctx.file.ar:
        ar_tool = [tool(tool = ctx.file.ar)]
    else:
        fail("ar or ar_path must be specified")

    strip_tool = []
    if ctx.attr.strip_path:
        strip_tool = [tool(path = ctx.attr.strip_path)]
    elif ctx.file.strip:
        strip_tool = [tool(tool = ctx.file.strip)]

    action_configs = [
        action_config(action_name = ACTION_NAMES.c_compile, tools = c_compiler_tool),
        action_config(action_name = ACTION_NAMES.cpp_compile, tools = cpp_compiler_tool),
        action_config(action_name = ACTION_NAMES.cpp_link_executable, tools = cpp_compiler_tool),
        action_config(action_name = ACTION_NAMES.cpp_link_static_library, tools = ar_tool),
        action_config(action_name = ACTION_NAMES.assemble, tools = c_compiler_tool),
        action_config(action_name = ACTION_NAMES.preprocess_assemble, tools = c_compiler_tool),
    ]
    if strip_tool:
        action_configs.append(action_config(action_name = ACTION_NAMES.strip, tools = strip_tool))

    static_link_flags = feature(
        name = "static_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = [
                            "rcsD",
                            "%{output_execpath}",
                        ],
                        expand_if_available = "output_execpath",
                    ),
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    base_compile_flags = [
        "-O2",
        "-mcpu=" + ctx.attr.cpu_version,
        "-m" + ctx.attr.cpu_version,
        "-DTF_LITE_MCU_DEBUG_LOG",
        "-DTF_LITE_USE_CTIME",
        "-DHEXAGON",
        "-DHEXAGON_ASM",
        "-DMALLOC_IN_STDLIB",
        "-DPTHREAD_STUBS",
        "-DUSE_PREALLOCATED_BUFFER",
        "-D_HAS_C9X",
        "-Wall",
        "-Wextra",
        "-Wno-missing-field-initializers",
        "-Wno-sign-compare",
        "-Wno-unused-parameter",
        "-Wno-write-strings",
        "-Wunused-function",
        "-Wno-unused-private-field",
        "-Wvla",
        "-fdata-sections",
        "-ffunction-sections",
        "-fmessage-length=0",
        "-fno-delete-null-pointer-checks",
        "-fno-exceptions",
        "-fno-register-global-dtors-with-atexit",
        "-fno-rtti",
        "-fno-short-enums",
        "-fno-threadsafe-statics",
        "-fno-aligned-allocation",
        "-fno-unwind-tables",
        "-fno-use-cxa-atexit",
        "-fomit-frame-pointer",
        "-fpermissive",
        "-funsigned-char",
        "-no-canonical-prefixes",
        "-isystem",
        "external/+hexagon_toolchain+hexagon_toolchain/sdk/posix",
        "-isystem",
        "external/+hexagon_toolchain+hexagon_toolchain/sdk/qurt",
        "-isystem",
        "external/hexagon_toolchain/sdk/posix",
        "-isystem",
        "external/hexagon_toolchain/sdk/qurt",
        "-I/root/Qualcomm/Hexagon_SDK/3.5.1/libs/common/qurt/computev66/include/posix",
        "-I/root/Qualcomm/Hexagon_SDK/3.5.1/libs/common/qurt/computev66/include/qurt",
    ]

    hexagon_flags = feature(
        name = "default_compiler_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                ],
                flag_groups = [
                    flag_group(
                        flags = base_compile_flags + ctx.attr.copts,
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-std=c++17",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-mcpu=" + ctx.attr.cpu_version,
                            "-m" + ctx.attr.cpu_version,
                            "-Lexternal/+hexagon_toolchain+hexagon_toolchain/stubs",
                            "-Wl,--whole-archive",
                            "-lpthread_stubs",
                            "-Wl,--no-whole-archive",
                            "-lstdc++",
                            "-lhexagon",
                            "-lm",
                            "-Wl,--gc-sections",
                            "-no-canonical-prefixes",
                        ] + ctx.attr.linkopts,
                    ),
                ],
            ),
        ],
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "hexagon-" + ctx.attr.cpu_version + "-toolchain",
        host_system_name = "local",
        target_system_name = "hexagon-unknown-none-elf",
        target_cpu = "hexagon",
        target_libc = "unknown",
        compiler = "hexagon-clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        action_configs = action_configs,
        tool_paths = [],
        cxx_builtin_include_directories = [
            "/root/Qualcomm",
            "/usr/include",
        ],
        features = [hexagon_flags, static_link_flags],
    )

hexagon_toolchain_config = rule(
    implementation = _hexagon_toolchain_config_impl,
    attrs = {
        "ar": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "ar_path": attr.string(),
        "c_compiler": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "c_compiler_path": attr.string(),
        "copts": attr.string_list(default = []),
        "cpp_compiler": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "cpp_compiler_path": attr.string(),
        "cpu_version": attr.string(default = "v66"),
        "linkopts": attr.string_list(default = []),
        "strip": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "strip_path": attr.string(),
    },
)
