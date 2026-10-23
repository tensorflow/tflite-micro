"""Repository rule for RISC-V GNU Toolchain."""

def _riscv_toolchain_impl(rctx):
    env_path = rctx.os.environ.get("RISCV_TOOLCHAIN_PATH", "")

    local_path = ""
    if env_path:
        local_path = env_path
    else:
        candidate = str(rctx.workspace_root) + "/tensorflow/lite/micro/tools/make/downloads/riscv_toolchain"
        if rctx.path(candidate).exists:
            local_path = candidate

    if local_path:
        for item in ["riscv64-unknown-elf", "bin", "include", "lib", "libexec", "share"]:
            item_path = local_path + "/" + item
            if rctx.path(item_path).exists:
                rctx.symlink(item_path, item)
    else:
        rctx.download_and_extract(
            url = [
                "http://mirror.tensorflow.org/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14.tar.gz",
            ],
            sha256 = "a4965d855d4e06eecc17b0305e0cb60c4fc7c4fc4c89a3514e28eccfc5b06d07",
            strip_prefix = "riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14",
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
        "bin/riscv64-unknown-elf-gcc*",
        "bin/riscv64-unknown-elf-g++*",
        "bin/riscv64-unknown-elf-cpp*",
        "bin/riscv64-unknown-elf-as*",
        "riscv64-unknown-elf/**",
        "lib/gcc/riscv64-unknown-elf/**",
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
        "bin/riscv64-unknown-elf-ld*",
        "bin/riscv64-unknown-elf-g++*",
        "bin/riscv64-unknown-elf-gcc*",
        "riscv64-unknown-elf/**",
        "lib/gcc/riscv64-unknown-elf/**",
        "libexec/**",
    ], allow_empty = True),
)

exports_files(glob(["bin/*"], allow_empty = True))
""",
    )

riscv_toolchain = repository_rule(
    implementation = _riscv_toolchain_impl,
    environ = ["RISCV_TOOLCHAIN_PATH"],
)
