"""Loads the kissfft library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "kissfft",
        patch_file = "//third_party/kissfft:kissfft.patch",
        strip_prefix = "kissfft-130",
        sha256 = "ac2259f84e372a582270ed7c7b709d02e6ca9c7206e40bb58de6ef77f6474872",
        urls = [
            "https://github.com/mborgerding/kissfft/archive/refs/tags/v130.zip",
        ],
        build_file = "//third_party/kissfft:BUILD.bazel",
    )
