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
"""Repository rule for Xtensa Cadence Toolchain."""

def _xtensa_toolchain_impl(rctx):
    tools_dir = ""

    env_tools_path = rctx.os.environ.get("XTENSA_TOOLS_PATH", "")
    if env_tools_path and rctx.path(env_tools_path).exists:
        tools_dir = env_tools_path

    if not tools_dir:
        env_base = rctx.os.environ.get("XTENSA_BASE", "")
        tools_version = rctx.os.environ.get("XTENSA_TOOLS_VERSION", "RI-2022.9-linux")
        if env_base:
            candidate = env_base + "/tools/" + tools_version + "/XtensaTools"
            if rctx.path(candidate).exists:
                tools_dir = candidate

    if not tools_dir:
        clang = rctx.which("xt-clang")
        if clang:
            # xt-clang is in <XtensaTools>/bin/xt-clang
            candidate = str(clang.dirname.dirname)
            if rctx.path(candidate).exists:
                tools_dir = candidate

    if not tools_dir:
        # Xtensa tools not installed on host. Create dummy targets so
        # non-Xtensa builds don't fail when analyzing the workspace.
        rctx.file("bin/ar", "")
        rctx.file("bin/xt-clang", "")
        rctx.file("bin/xt-clang++", "")
        rctx.file("bin/xt-run", "")
        rctx.file("bin/xt-strip", "")
        rctx.file(
            "BUILD.bazel",
            content = """package(default_visibility = ["//visibility:public"])

filegroup(name = "everything")
filegroup(name = "compiler_files")
filegroup(name = "linker_files")
exports_files([
    "bin/ar",
    "bin/xt-clang",
    "bin/xt-clang++",
    "bin/xt-run",
    "bin/xt-strip",
])
""",
        )
        return

    # Symlink toolchain directory structure, explicitly excluding large source trees
    # (e.g. xtensa-elf/src and src/ which contain tens of thousands of unused source files
    # that cause sandbox inode exhaustion during parallel builds).
    for item in ["config", "include", "lib", "libexec", "llvm", "share"]:
        item_path = tools_dir + "/" + item
        if rctx.path(item_path).exists:
            rctx.symlink(item_path, item)

    # Only symlink xtensa-elf/include, never xtensa-elf/src
    if rctx.path(tools_dir + "/xtensa-elf/include").exists:
        rctx.symlink(tools_dir + "/xtensa-elf/include", "xtensa-elf/include")

    # In bin/, symlink xtensa tools and link host ar (to avoid RI-2022.9 xt-ar bug)
    bin_tools = [
        "xt-clang",
        "xt-clang++",
        "xt-run",
        "xt-objcopy",
        "xt-strip",
        "xt-size",
        "xt-nm",
    ]
    for b in bin_tools:
        src = tools_dir + "/bin/" + b
        if rctx.path(src).exists:
            rctx.symlink(src, "bin/" + b)

    host_ar = rctx.which("ar")
    if host_ar:
        rctx.symlink(host_ar, "bin/ar")
    elif rctx.path(tools_dir + "/bin/xt-ar").exists:
        rctx.symlink(tools_dir + "/bin/xt-ar", "bin/ar")

    rctx.file(
        "BUILD.bazel",
        content = """package(default_visibility = ["//visibility:public"])

filegroup(
    name = "everything",
    srcs = glob(["**/*"], allow_empty = True),
)

alias(
    name = "all_files",
    actual = ":everything",
)

filegroup(
    name = "compiler_files",
    srcs = glob([
        "bin/xt-clang*",
        "config/**",
        "lib/**",
        "llvm/**",
        "xtensa-elf/**",
    ], allow_empty = True),
)

alias(
    name = "as_files",
    actual = ":compiler_files",
)

alias(
    name = "ar_files",
    actual = ":everything",
)

filegroup(
    name = "linker_files",
    srcs = glob([
        "bin/xt-clang*",
        "bin/ar",
        "config/**",
        "lib/**",
        "llvm/**",
        "xtensa-elf/**",
    ], allow_empty = True),
)

exports_files(glob(["bin/*"], allow_empty = True))
""",
    )

xtensa_toolchain = repository_rule(
    implementation = _xtensa_toolchain_impl,
    environ = [
        "XTENSA_BASE",
        "XTENSA_TOOLS_PATH",
        "XTENSA_TOOLS_VERSION",
    ],
)
