"""Repository rule for creating an external repository `name` with C-language
dependencies from the Python package `pkg`, published by rules_python. Set
`pkg` using the `requirement` function from rules_python.

The top-level Bazel package in the created repository provides two targets for
use as dependencies in C-language targets elsewhere.

    * `:cc_headers`--for including headers
    * `:cc_library`--for including headers and linking against a library

The mandatory `includes` attribute should be set to a list of
include dirs to be added to the compile command line.

The optional `libs` attribute should be set to a list of libraries to link with
the binary target.

Specify all paths relative to the parent directory in which the package is
extracted (e.g., site-packages/). Thus paths will begin with the package's
Python namespace or module name. Note this name may differ from the Python
distribution package name---e.g., the distribution package `tensorflow-gpu`
distributes the Python namespace package `tensorflow`. To see what paths are
available, it might help to examine the directory tree in the external
repository created for the package by rules_python. The external repository is
created in the bazel cache; in the example below, in a subdirectory
`external/tflm_pip_deps_numpy`.

For example, to use the headers from NumPy:

1. Add Python dependencies (numpy is named in python_requirements.txt), via the
usual method, to an external repository named `tflm_pip_deps` via @rules_python
in the WORKSPACE:
```
    load("@rules_python//python:pip.bzl", "pip_parse")
    pip_parse(
       name = "tflm_pip_deps",
       requirements_lock = "@//third_party:python_requirements.txt",
    )
    load("@tflm_pip_deps//:requirements.bzl", "install_deps")
    install_deps()
```

2. Use the repository rule `py_pkg_cc_deps` in the WORKSPACE to create an
external repository with a target `@numpy_cc_deps//:cc_headers`, passing the
`:pkg` target from @tflm_pip_deps, obtained via requirement(), and an
`includes` path based on an examination of the package and the desired #include
paths in the C code:
```
    load("@tflm_pip_deps//:requirements.bzl", "requirement")
    load("@//python:py_pkg_cc_deps.bzl", "py_pkg_cc_deps")
    py_pkg_cc_deps(
        name = "numpy_cc_deps",
        pkg = requirement("numpy"),
        includes = ["numpy/core/include"],
    )
```

3. Use the cc_library target `@numpy_cc_deps//:cc_headers` in a BUILD file as
a dependency to a rule that needs the headers, e.g., the cc_library()-based
pybind_library():
```
    pybind_library(
        name = "your_extension_lib",
        srcs = [...],
        deps = ["@numpy_cc_deps//:cc_headers", ...],
    )
```

See the test target //python/tests:cc_dep_link_test elsewhere for an example
which links against a library shipped in a Python package.
"""

# This extends the standard rules_python rules to expose C-language dependences
# contained in some Python packages like NumPy. It extends rules_python to
# avoid duplicating the download mechanism, and to ensure the Python package
# versions used throughout the WORKSPACE are consistent.

def _rules_python_path(ctx, pkg):
    # Make an absolute path to the rules_python repository for the Python
    # package `pkg`.

    # WARNING: To get a filesystem path via ctx.path(), its argument must be a
    # label to a non-generated file. ctx.path() does not work on non-file
    # A standard technique for finding the path to a repository (see,
    # e.g., rules_go) is to use the repository's BUILD file; however, the exact
    # name of the build file is an implementation detail of rules_python.
    build_file = pkg.relative(":BUILD.bazel")
    abspath = ctx.path(build_file).dirname
    return abspath

def _join_paths(a, b):
    result = ""
    if type(a) == "string" and type(b) == "string":
        result = "/".join((a, b))

    elif type(a) == "path" and type(b) == "string":
        # Append components of string b to path a, because path.get_child()
        # requires one component at a time.
        result = a
        for x in b.split("/"):
            result = result.get_child(x)

    return result

def _make_build_file(basedir, include_paths, libs):
    template = """\
package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cc_headers",
    hdrs = glob(%s, allow_empty=False, exclude_directories=1),
    includes = %s,
)

cc_library(
    name = "cc_library",
    srcs = %s,
    deps = [":cc_headers"],
)
"""
    hdrs = [(_join_paths(basedir, inc) + "/**") for inc in include_paths]
    includes = [_join_paths(basedir, inc) for inc in include_paths]
    srcs = [_join_paths(basedir, lib) for lib in libs]

    return template % (hdrs, includes, srcs)

def _py_pkg_cc_deps(ctx):
    # Create a repository with the directory tree:
    #     repository/
    #               |- _site --> @specific_rules_python_pkg/site-packages
    #               \_ BUILD
    #
    # When debugging, it might help to examine the tree and BUILD file of this
    # repository, created in the bazel cache.

    # Symlink to the rules_python repository of pkg
    srcdir = _join_paths(_rules_python_path(ctx, ctx.attr.pkg), "site-packages")
    destdir = "_site"
    ctx.symlink(srcdir, destdir)

    # Write a BUILD file publishing targets
    ctx.file(
        "BUILD",
        content = _make_build_file(destdir, ctx.attr.includes, ctx.attr.libs),
        executable = False,
    )

py_pkg_cc_deps = repository_rule(
    implementation = _py_pkg_cc_deps,
    local = True,
    attrs = {
        "pkg": attr.label(
            doc = "Python package target via rules_python's requirement()",
            mandatory = True,
        ),
        "includes": attr.string_list(
            doc = "list of include dirs",
            mandatory = True,
            allow_empty = False,
        ),
        "libs": attr.string_list(
            doc = "list of libraries against which to link",
            mandatory = False,
        ),
    },
)
