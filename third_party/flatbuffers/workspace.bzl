"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-a66de58af9565586832c276fbb4251fc416bf07f",
        sha256 = "da06ac2fc6fed8e38b6392f5a20fa24a4290cecaadd87aef16b6b84960408680",
        urls = [
            "https://github.com/google/flatbuffers/archive/a66de58af9565586832c276fbb4251fc416bf07f.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
