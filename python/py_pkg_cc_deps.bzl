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
    template = """\npackage(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cc_headers",
    hdrs = glob(%s, allow_empty=True, exclude_directories=1),
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

def _cc_deps_impl(ctx):
    # Dictionary to store group of configs by name
    # { name: [config, ...] }
    groups = {}
    for mod in ctx.modules:
        for config in mod.tags.config:
            if config.name not in groups:
                groups[config.name] = []
            groups[config.name].append(config)

    for name, configs in groups.items():
        # Create a hub repository that selects between versions
        name_version_map = {}

        for config in configs:
            for version in config.versions:
                vname = "%s_%s" % (name, version.replace(".", ""))

                # Construct package label from template
                # e.g. @tflm_pip_deps_310_numpy//:pkg
                pip_repo = config.pip_repo_template.format(
                    version = version.replace(".", ""),
                    pkg = config.pkg_name,
                )
                pkg_label = Label("@%s//:pkg" % pip_repo)

                py_pkg_cc_deps(
                    name = vname,
                    pkg = pkg_label,
                    includes = config.includes,
                    libs = config.libs,
                )
                name_version_map[version] = vname

        _py_pkg_cc_deps_hub(
            name = name,
            name_version_map = name_version_map,
        )

def _py_pkg_cc_deps_hub_impl(ctx):
    # Map from python version string to repo name
    # e.g. {"3.10": "numpy_cc_deps_310"}

    content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cc_headers",
    deps = select({
%s
    }),
)

cc_library(
    name = "cc_library",
    deps = select({
%s
    }),
)
"""
    headers_lines = []
    library_lines = []

    # Sort versions to ensure deterministic output
    sorted_versions = sorted(ctx.attr.name_version_map.keys())

    for version in sorted_versions:
        repo = ctx.attr.name_version_map[version]
        condition = "@rules_python//python/config_settings:is_python_%s" % version
        headers_lines.append('        "%s": ["@%s//:cc_headers"],' % (condition, repo))
        library_lines.append('        "%s": ["@%s//:cc_library"],' % (condition, repo))

    # Add a default condition using the first sorted version
    if sorted_versions:
        first_repo = ctx.attr.name_version_map[sorted_versions[0]]
        headers_lines.append('        "//conditions:default": ["@%s//:cc_headers"],' % first_repo)
        library_lines.append('        "//conditions:default": ["@%s//:cc_library"],' % first_repo)

    ctx.file("BUILD", content % ("\n".join(headers_lines), "\n".join(library_lines)))

_py_pkg_cc_deps_hub = repository_rule(
    implementation = _py_pkg_cc_deps_hub_impl,
    attrs = {
        "name_version_map": attr.string_dict(mandatory = True),
    },
)

cc_deps = module_extension(
    implementation = _cc_deps_impl,
    tag_classes = {
        "config": tag_class(
            attrs = {
                "name": attr.string(mandatory = True),
                "includes": attr.string_list(mandatory = True),
                "libs": attr.string_list(),
                "versions": attr.string_list(mandatory = True),
                "pkg_name": attr.string(mandatory = True),
                "pip_repo_template": attr.string(mandatory = True),
            },
        ),
    },
)
