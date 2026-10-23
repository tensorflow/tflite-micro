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
"""Repository rule for Qualcomm Hexagon Toolchain."""

PTHREAD_STUBS_C = """typedef int pthread_once_t;
typedef unsigned int pthread_key_t;
typedef void* pthread_mutex_t;

int pthread_once(pthread_once_t *control, void (*init_routine)(void)) {
  if (control && *control == 0) {
    *control = 1;
    if (init_routine) init_routine();
  }
  return 0;
}
int pthread_key_create(pthread_key_t *key, void (*destructor)(void*)) {
  (void)destructor;
  if (key) *key = 1;
  return 0;
}
static void* s_val = 0;
void *pthread_getspecific(pthread_key_t key) { (void)key; return s_val; }
int pthread_setspecific(pthread_key_t key, const void *value) { (void)key; s_val = (void*)value; return 0; }
int pthread_mutex_lock(pthread_mutex_t *mutex) { (void)mutex; return 0; }
int pthread_mutex_unlock(pthread_mutex_t *mutex) { (void)mutex; return 0; }
void free(void*);
void _ZdlPvSt11align_val_t(void* ptr, unsigned long align) { (void)align; free(ptr); }
"""

def _hexagon_toolchain_impl(rctx):
    sdk_root = rctx.os.environ.get("HEXAGON_SDK_ROOT", "")
    tools_root = rctx.os.environ.get("HEXAGON_ROOT", "")
    tool_ver = rctx.os.environ.get("HEXAGON_TOOL_VER", "8.3.07")
    cpu_ver = rctx.os.environ.get("HEXAGON_CPU_VER", "v66")
    tflm_core = rctx.os.environ.get("HEXAGON_TFLM_CORE", "")
    if not tflm_core:
        tflm_core = rctx.os.environ.get("HEXAGON_TFLM_LIB", "")

    tools_dir = ""
    if tools_root and tool_ver:
        candidate = tools_root.rstrip("/") + "/" + tool_ver + "/Tools"
        if rctx.path(candidate).exists:
            tools_dir = candidate

    if not tools_dir:
        clang = rctx.which("hexagon-clang++")
        if clang:
            candidate = str(clang.dirname.dirname)
            if rctx.path(candidate).exists:
                tools_dir = candidate
                if not tools_root:
                    tools_root = str(clang.dirname.dirname.dirname.dirname)
                    tool_ver = clang.dirname.dirname.dirname.basename

    if not sdk_root:
        if tools_root:
            candidate_sdk = str(rctx.path(tools_root).dirname)
            if rctx.path(candidate_sdk + "/libs").exists:
                sdk_root = candidate_sdk
        if not sdk_root and rctx.path("/root/Qualcomm/Hexagon_SDK/3.5.1").exists:
            sdk_root = "/root/Qualcomm/Hexagon_SDK/3.5.1"

    if not tflm_core:
        if rctx.path("/root/Qualcomm/hexagon_tflm_core.a").exists:
            tflm_core = "/root/Qualcomm/hexagon_tflm_core.a"
        elif sdk_root and rctx.path(sdk_root + "/../hexagon_tflm_core.a").exists:
            tflm_core = str(rctx.path(sdk_root + "/../hexagon_tflm_core.a"))

    if not tools_dir:
        # Hexagon tools not installed on host. Create dummy targets so
        # non-Hexagon builds and queries don't fail when analyzing the workspace.
        rctx.file("bin/ar", "")
        rctx.file("bin/hexagon-clang", "")
        rctx.file("bin/hexagon-clang++", "")
        rctx.file("bin/hexagon-sim", "")
        rctx.file("bin/hexagon-strip", "")
        rctx.file("qualcomm/hexagon_tflm_core.a", "")
        rctx.file("stubs/libpthread_stubs.a", "")
        rctx.file("sdk/lib/libposix.a", "")
        rctx.file("sdk/lib/libqurt.a", "")
        rctx.file("target/hexagon/lib/v66/libstdc++.a", "")
        rctx.file("target/hexagon/lib/v66/libhexagon.a", "")
        rctx.file(
            "BUILD.bazel",
            content = """load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(name = "everything")
filegroup(name = "compiler_files")
filegroup(name = "linker_files")
filegroup(name = "as_files")
filegroup(name = "ar_files")

cc_library(name = "libstdcxx")
cc_library(name = "hexagon_tflm_core")
cc_library(name = "qurt_headers")

exports_files([
    "bin/ar",
    "bin/hexagon-clang",
    "bin/hexagon-clang++",
    "bin/hexagon-sim",
    "bin/hexagon-strip",
])
""",
        )
        return

    # Symlink toolchain binaries
    bin_tools = [
        "hexagon-clang",
        "hexagon-clang++",
        "hexagon-sim",
        "hexagon-strip",
        "hexagon-nm",
        "hexagon-size",
    ]
    for b in bin_tools:
        src = tools_dir + "/bin/" + b
        if rctx.path(src).exists:
            rctx.symlink(src, "bin/" + b)

    if rctx.path(tools_dir + "/bin/hexagon-ar").exists:
        rctx.symlink(tools_dir + "/bin/hexagon-ar", "bin/ar")
    elif rctx.which("ar"):
        rctx.symlink(rctx.which("ar"), "bin/ar")

    # Symlink toolchain target and library trees
    for item in ["target", "lib", "include"]:
        src = tools_dir + "/" + item
        if rctx.path(src).exists:
            rctx.symlink(src, item)

    # Symlink SDK headers (QURT & POSIX)
    if sdk_root:
        qurt_include = sdk_root + "/libs/common/qurt/compute" + cpu_ver + "/include"
        if rctx.path(qurt_include + "/posix").exists:
            rctx.symlink(qurt_include + "/posix", "sdk/posix")
        if rctx.path(qurt_include + "/qurt").exists:
            rctx.symlink(qurt_include + "/qurt", "sdk/qurt")
        qurt_lib = sdk_root + "/libs/common/qurt/compute" + cpu_ver + "/lib"
        if rctx.path(qurt_lib).exists:
            rctx.symlink(qurt_lib, "sdk/lib")

    # Symlink hexagon_tflm_core.a
    if tflm_core and rctx.path(tflm_core).exists:
        rctx.symlink(tflm_core, "qualcomm/hexagon_tflm_core.a")
    else:
        rctx.file("qualcomm/hexagon_tflm_core.a", "")

    # Build pthread_stubs static library to satisfy libc++abi in standalone simulator
    rctx.file("stubs/pthread_stubs.c", PTHREAD_STUBS_C)
    ar_cmd = tools_dir + "/bin/hexagon-ar" if rctx.path(tools_dir + "/bin/hexagon-ar").exists else "ar"
    rctx.execute([tools_dir + "/bin/hexagon-clang", "-c", "-mcpu=" + cpu_ver, "-m" + cpu_ver, "-O2", "stubs/pthread_stubs.c", "-o", "stubs/pthread_stubs.o"])
    rctx.execute([ar_cmd, "rcs", "stubs/libpthread_stubs.a", "stubs/pthread_stubs.o"])

    rctx.file(
        "BUILD.bazel",
        content = """load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

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
        "bin/hexagon-clang*",
        "lib/clang/**",
        "target/**",
        "sdk/**",
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
        "bin/hexagon-clang*",
        "bin/ar",
        "lib/**",
        "target/**",
        "sdk/**",
        "stubs/**",
        "qualcomm/**",
    ], allow_empty = True),
)

cc_library(
    name = "libstdcxx",
    srcs = glob([
        "target/hexagon/lib/{cpu}/libstdc++.a",
        "target/hexagon/lib/{cpu}/libhexagon.a",
    ], allow_empty = True),
)

cc_library(
    name = "hexagon_tflm_core",
    srcs = glob(["qualcomm/hexagon_tflm_core.a"], allow_empty = True),
)

cc_library(
    name = "qurt_headers",
    hdrs = glob([
        "sdk/**",
        "target/hexagon/include/**",
    ], allow_empty = True),
    includes = [
        "sdk/posix",
        "sdk/qurt",
        "target/hexagon/include",
    ],
)

exports_files(glob(["bin/*"], allow_empty = True))
""".format(cpu = cpu_ver),
    )

hexagon_toolchain = repository_rule(
    implementation = _hexagon_toolchain_impl,
    environ = [
        "HEXAGON_CPU_VER",
        "HEXAGON_ROOT",
        "HEXAGON_SDK_ROOT",
        "HEXAGON_TFLM_CORE",
        "HEXAGON_TFLM_LIB",
        "HEXAGON_TOOL_VER",
    ],
)
