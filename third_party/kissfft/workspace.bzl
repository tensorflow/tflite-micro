"""Loads the kissfft library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "kissfft",
        strip_prefix = "kissfft-33d9ad3bad3fe8f1fb43a4634f61ea9a40240534",
        sha256 = "53576140e94c947e31f208e3ccd1913df422439158f12c0dd39e8ef247552b94",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/mborgerding/kissfft/archive/33d9ad3bad3fe8f1fb43a4634f61ea9a40240534.tar.gz",
            "https://github.com/mborgerding/kissfft/archive/33d9ad3bad3fe8f1fb43a4634f61ea9a40240534.tar.gz",
        ],
        build_file = "//third_party/kissfft:BUILD.bazel",
    )
