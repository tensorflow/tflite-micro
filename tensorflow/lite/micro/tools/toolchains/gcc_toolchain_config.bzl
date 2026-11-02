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
"""Shared bare-metal GCC cc_toolchain_config rule for embedded targets."""

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

def _gcc_toolchain_config_impl(ctx):
    c_compiler_tool = [tool(tool = ctx.file.c_compiler)]
    tool_prefix = ctx.file.c_compiler.path
    if tool_prefix.endswith("gcc"):
        tool_prefix = tool_prefix[:-3]

    cpp_compiler_tool = [tool(tool = ctx.file.cpp_compiler)]
    ar_tool = [tool(tool = ctx.file.ar)]
    strip_tool = [tool(tool = ctx.file.strip)] if ctx.file.strip else []

    base_flags = [
        "-O2",
        "-fomit-frame-pointer",
        "-ffunction-sections",
        "-fdata-sections",
        "-no-canonical-prefixes",
        "-fno-canonical-system-headers",
    ]
    assemble_flags = ["-no-canonical-prefixes"]
    if tool_prefix:
        base_flags.append("-B" + tool_prefix)
        assemble_flags.append("-B" + tool_prefix)

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

    flag_sets = []
    if ctx.attr.arch_flags:
        flag_sets.append(
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(flags = ctx.attr.arch_flags),
                ],
            ),
        )
    flag_sets.extend([
        flag_set(
            actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
            flag_groups = [
                flag_group(flags = base_flags + ctx.attr.compile_flags),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.assemble, ACTION_NAMES.preprocess_assemble],
            flag_groups = [
                flag_group(flags = assemble_flags),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.preprocess_assemble],
            flag_groups = [
                flag_group(flags = ["-fno-canonical-system-headers"]),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.cpp_compile],
            flag_groups = [
                flag_group(
                    flags = [
                        "-std=c++17",
                        "-fno-exceptions",
                        "-fno-rtti",
                        "-fno-unwind-tables",
                        "-fno-threadsafe-statics",
                    ] + ctx.attr.cxxopts,
                ),
            ],
        ),
        flag_set(
            actions = [ACTION_NAMES.cpp_link_executable],
            flag_groups = [
                flag_group(
                    flags = [
                        "-static",
                        "-no-pie",
                        "-Wl,--gc-sections",
                        "-no-canonical-prefixes",
                    ] + ctx.attr.linkopts,
                ),
            ],
        ),
    ])

    default_compiler_flags = feature(
        name = "default_compiler_flags",
        enabled = True,
        flag_sets = flag_sets,
    )

    has_cxx17_headers = feature(
        name = "has_cxx17_headers",
        enabled = True,
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = ctx.attr.cpu + "-toolchain",
        host_system_name = "local",
        target_system_name = ctx.attr.target_system_name,
        target_cpu = ctx.attr.cpu,
        target_libc = "unknown",
        compiler = "gcc",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        action_configs = action_configs,
        tool_paths = [],
        cxx_builtin_include_directories = [],
        features = [
            default_compiler_flags,
            static_link_flags,
            has_cxx17_headers,
        ],
    )

gcc_toolchain_config = rule(
    implementation = _gcc_toolchain_config_impl,
    attrs = {
        "ar": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "arch_flags": attr.string_list(mandatory = True),
        "c_compiler": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "compile_flags": attr.string_list(default = []),
        "cpp_compiler": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "cpu": attr.string(mandatory = True),
        "cxxopts": attr.string_list(default = []),
        "linkopts": attr.string_list(default = []),
        "strip": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
        ),
        "target_system_name": attr.string(mandatory = True),
    },
)
