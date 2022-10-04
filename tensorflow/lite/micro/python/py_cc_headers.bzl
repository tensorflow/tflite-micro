"Repository rule for creating a repository with the C headers of a Python package."

# This extends the standard @rules_python rules to expose the C headers
# contained in some Python packages like NumPy. It extends @rules_python rather
# than duplicating the download mechanism, and to ensure the versions of Python
# packages used throughout the WWORKSPACE are consistent. Exposing the headers
# could conceivably become a build-in feature of @rules_python at some point.

def _find_headers(ctx, py_library, include_prefix):
    # Find the path to the headers within the @rules_python repository for the Python
    # package `py_library`.

    # WARNING: To get a flesystem path via ctx.path(), its argument must be a
    # label to a non-generated file. ctx.path() does not work on non-file
    # A standard technique for finding the path to a repository (see,
    # e.g., rules_go) is to use the repository's BUILD file; however, the exact
    # name of the build file is an implementation detail of @rules.python.
    build = py_library.relative(":BUILD.bazel")
    path = ctx.path(build).dirname

    # Append the include_prefix, if any. get_child() needs one component at a
    # time.
    if include_prefix:
        for c in include_prefix.split("/"):
            path = path.get_child(c)

    return path

def _tflm_py_cc_headers_impl(ctx):
    # Create a repository with the directory tree:
    #     repository/
    #               |- _include --> @rules_python//package/[include_prefix]
    #               \_ BUILD

    # Symlink to the headers within the @rules_python repository
    src = _find_headers(ctx, ctx.attr.py_library, ctx.attr.include_prefix)
    destdir = "_include"
    ctx.symlink(src, destdir)

    # Write a BUILD file publishing a cc_library() target for the headers.
    BUILD = """\
cc_library(
    name = "%s",
    hdrs = glob(["%s/**/*.h"], allow_empty=False),
    includes = ["%s"],
    visibility = ["@//tensorflow/lite/micro:__subpackages__"],
)
""" % (ctx.attr.name, destdir, destdir)
    ctx.file("BUILD", content = BUILD, executable = False)

tflm_py_cc_headers = repository_rule(
    implementation = _tflm_py_cc_headers_impl,
    local = True,
    attrs = {
        "py_library": attr.label(mandatory = True),
        "include_prefix": attr.string(mandatory = False),
    },
)
"""Repository rule for creating an external repository `name` with the C
headers from a Python package published as a `py_library` target by
@rules_python. The top-level Bazel package in the repository provides a
cc_library target named `name` (the default target of the package) for use
as a dependency in targets building against the headers.

Use the optional `include_prefix` to base headers at a subdirectory of the
Python package rather than its root. E.g., numpy's headers are in
`numpy/core/include/numpy/header.h` relative to the root of the numpy package.
Use `include_prefix = "numpy/core/include"` to include headers as `#include
<numpy/header.h>` instead of `#include <numpy/core/include/numpy/header.h>`

For example, to use the headers from NumPy:

1. Add Python dependencies (numpy is named in requirements.txt) to an external
repository named `tflm_pip_deps` via @rules_python in the WORKSPACE:
```
    load("@rules_python//python:pip.bzl", "pip_install")
    pip_install(
       name = "tflm_pip_deps",
       requirements = "//third_party:requirements.txt",
    )
```

2. Use the repository rule `tflm_py_cc_headers` in the WORKSPACE to create an
external repository with a target `@numpy_headers:numpy_headers`, passing the
py_library target from @tflm_pip_deps obtained via requirement() and an
`include_prefix` based on an examination of the package and the desired #include
paths in our C code:
```
    load("@tflm_pip_deps//:requirements.bzl", "requirement")
    load("@//tensorflow/lite/micro/python:py_cc_headers.bzl", "tflm_py_cc_headers")
    tflm_py_cc_headers(
        name = "numpy_headers",
        py_library = requirement("numpy"),
        include_prefix = "numpy/core/include",
    )
```

3. Use the cc_library target `@numpy_headers:numpy_headers` in a BUILD file as
a dependency to a rule that needs the headers, e.g., the cc_library()-based
pybind_library(). Note the target can be spelled as the default target of the
repository's top-level package:
```
    pybind_library(
        name = "your_extension_lib",
        srcs = [...],
        deps = ["@numpy_headers", ...],
    )
```
"""
