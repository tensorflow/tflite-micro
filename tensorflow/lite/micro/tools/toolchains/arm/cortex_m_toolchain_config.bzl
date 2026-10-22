"""Cortex-M toolchain configuration rule for ARM toolchains."""

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

def _cortex_m_toolchain_config_impl(ctx):
    tool_prefix = ""
    if ctx.attr.c_compiler_path:
        c_compiler_tool = [tool(path = ctx.attr.c_compiler_path)]
        if ctx.attr.c_compiler_path.endswith("gcc"):
            tool_prefix = ctx.attr.c_compiler_path[:-3]
    elif ctx.file.c_compiler:
        c_compiler_tool = [tool(tool = ctx.file.c_compiler)]
        tool_prefix = ctx.file.c_compiler.path
        if tool_prefix.endswith("gcc"):
            tool_prefix = tool_prefix[:-3]
    else:
        fail("c_compiler or c_compiler_path must be specified")

    base_flags = [
        "-O2",
        "-fomit-frame-pointer",
        "-ffunction-sections",
        "-fdata-sections",
        "-no-canonical-prefixes",
        "-fno-canonical-system-headers",
    ]
    if tool_prefix:
        base_flags.append("-B" + tool_prefix)

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

    qemu_semihosting = feature(
        name = "qemu_semihosting",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [
                    flag_group(
                        flags = [
                            "--specs=rdimon.specs",
                            "-Wl,-Ttext=0x10000",
                        ],
                    ),
                ],
            ),
        ],
    )

    cortex_m_flags = feature(
        name = "default_compiler_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile, ACTION_NAMES.cpp_link_executable],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.copts,
                    ),
                ],
            ),
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = base_flags,
                    ),
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
                        ],
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
                        ],
                    ),
                ],
            ),
        ],
    )

    has_cxx17_headers = feature(
        name = "has_cxx17_headers",
        enabled = True,
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = ctx.attr.cpu + "-toolchain",
        host_system_name = "local",
        target_system_name = "arm-none-eabi",
        target_cpu = ctx.attr.cpu,
        target_libc = "unknown",
        compiler = "gcc",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        action_configs = action_configs,
        tool_paths = [],
        cxx_builtin_include_directories = [],
        features = [
            cortex_m_flags,
            static_link_flags,
            qemu_semihosting,
            has_cxx17_headers,
        ],
    )

cortex_m_toolchain_config = rule(
    implementation = _cortex_m_toolchain_config_impl,
    attrs = {
        "ar": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "ar_path": attr.string(),
        "c_compiler": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "c_compiler_path": attr.string(),
        "copts": attr.string_list(mandatory = True),
        "cpp_compiler": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "cpp_compiler_path": attr.string(),
        "cpu": attr.string(mandatory = True),
        "strip": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "strip_path": attr.string(),
    },
)
