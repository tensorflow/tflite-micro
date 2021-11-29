load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party:repo.bzl", "tf_http_archive")

def initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    flatbuffers()
    kissfft()
    ruy()

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def tf_repositories(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds."""

    # https://github.com/bazelbuild/bazel-skylib/releases
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "gemmlowp",
        sha256 = "43146e6f56cb5218a8caaab6b5d1601a083f1f31c06ff474a4378a7d35be9cfb",  # SHARED_GEMMLOWP_SHA
        strip_prefix = "gemmlowp-fda83bdc38b118cc6b56753bd540caa49e570745",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
            "https://github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
        ],
    )

    tf_http_archive(
        name = "absl_py",
        sha256 = "516e83df99fe7c365727ef09c9e1f83b55985afaf23dd1ca572b3e160057f5f8",
        strip_prefix = "abseil-py-b188d9080c8e5628bb52a93a04ad930abb1717eb",
        urls = [
            "https://github.com/abseil/abseil-py/archive/b188d9080c8e5628bb52a93a04ad930abb1717eb.zip",
        ],
    )

    tf_http_archive(
        name = "six_archive",
        urls = [
            "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        build_file = "@//third_party:six.BUILD",
    )

    initialize_third_party()

def workspace():
    tf_repositories()
