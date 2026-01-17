"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-25.9.23",
        sha256 = "9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada",
        urls = [
            "https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.oss",
        system_build_file = "//third_party/flatbuffers:BUILD.system",
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
