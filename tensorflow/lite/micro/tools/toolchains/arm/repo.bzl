"""Repository rule for Arm GNU Toolchain."""

def _arm_toolchain_impl(rctx):
    env_path = rctx.os.environ.get("ARM_TOOLCHAIN_PATH", "")

    local_path = ""
    if env_path:
        local_path = env_path
    else:
        candidate = str(rctx.workspace_root) + "/tensorflow/lite/micro/tools/make/downloads/gcc_embedded"
        if rctx.path(candidate).exists:
            local_path = candidate

    if local_path:
        for item in ["arm-none-eabi", "bin", "include", "lib", "libexec", "share"]:
            item_path = local_path + "/" + item
            if rctx.path(item_path).exists:
                rctx.symlink(item_path, item)
    else:
        rctx.download_and_extract(
            url = [
                "https://developer.arm.com/-/media/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi.tar.xz",
                "https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/14.3.rel1/binrel/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi.tar.xz",
            ],
            sha256 = "8f6903f8ceb084d9227b9ef991490413014d991874a1e34074443c2a72b14dbd",
            stripPrefix = "arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi",
        )

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
        "bin/arm-none-eabi-gcc*",
        "bin/arm-none-eabi-g++*",
        "bin/arm-none-eabi-cpp*",
        "bin/arm-none-eabi-as*",
        "arm-none-eabi/**",
        "lib/gcc/arm-none-eabi/**",
        "libexec/**",
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
        "bin/arm-none-eabi-ld*",
        "bin/arm-none-eabi-g++*",
        "bin/arm-none-eabi-gcc*",
        "arm-none-eabi/**",
        "lib/gcc/arm-none-eabi/**",
        "libexec/**",
    ], allow_empty = True),
)

exports_files(glob(["bin/*"], allow_empty = True))
""",
    )

arm_toolchain = repository_rule(
    implementation = _arm_toolchain_impl,
    environ = ["ARM_TOOLCHAIN_PATH"],
)
