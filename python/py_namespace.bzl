"""Repository rule py_workspace(), augmenting @rules_python, for relocating
py_library() and py_package() files underneath a given Python namespace.

The stock @rules_python py_library() -> py_package() -> py_wheel() BUILD file
workflow packages files at Python package paths set to the source paths of the
files relative to the workspace root. This has a several problems. Firstly, it
implies that files must be located underneath a source directory with the same
name as the desired Python namespace package. ( py_wheel.strip_path_prefixes
can remove path components, but cannot add them.) This is not always feasible
or desirable.

Secondly, this path naming is incompatible with the PYTHONPATH set by
@rules_python when executing Python programs in the source tree via
py_binary(). PYTHONPATH is set such that imports should begin with the
WORKSPACE name, followed by the path from the workspace root. py_wheel(),
however, packages files such that imports use only the path from the workspace
root.

For example, the source file:
    example/hello.py
is imported by a py_binary() running in the source tree as:
    `import workspace_name.example.hello`
but must be imported from within the package created by py_wheel() as:
    `import example.hello`

The end result is that code cannot be written to work both in the source tree
and installed in a Python environment via a package.

py_namespace() fixes these problems by providing the means to package files
within a Python package namespace without adding a corresponding directory in
the source tree. The BUILD workflow changes to py_libary() -> py_package() ->
**py_namespace()** -> py_wheel(). For example:

```
    # in example/BUILD

    py_library(
        name = "library",
        srcs = ["hello.py"],
        deps = ...,
    )

    py_package(
        name = "package",
        deps = [":library"],
    )

    py_namespace(
        name = "namespace",
        deps = [":package"],
        namespace = "foo",
    )

    py_wheel(
        ....
        deps = [":namespace"],
    )
```

In this case, the source file:
    example/hello.py
which is imported by a py_binary() running in the source tree as:
    `import workspace_name.example.hello`
is imported from the package created by py_wheel() as:
    `import foo.example.hello`

If the namespace and the WORKSPACE name match, the import paths used when
running in the source tree will match the import paths used when installed in
the Python environment.

Furthermore, the Python package can be given an __init__.py file via the
attribute `init`. The given file is relocated directly under the namespace as
__init__.py, regardless of its path in the source tree. This __init__.py can be
used for, among other things, providing a user-friendly public API: providing
aliases for modules otherwise deeply nested in subpackages due to their
location in the source tree.
"""

def _relocate_init(ctx):
    # Copy the init file directly underneath the namespace directory.
    outfile = ctx.actions.declare_file(ctx.attr.namespace + "/__init__.py")
    ctx.actions.run_shell(
        inputs = [ctx.file.init],
        outputs = [outfile],
        arguments = [ctx.file.init.path, outfile.path],
        command = "cp $1 $2",
    )
    return outfile

def _relocate_deps(ctx):
    # Copy all transitive deps underneath the namespace directory. E.g.,
    #     example/hello.py
    # becomes:
    #     namespace/example/hello.py
    outfiles = []
    inputs = depset(transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps])

    for infile in sorted(inputs.to_list()):
        outfile = ctx.actions.declare_file(ctx.attr.namespace + "/" + infile.short_path)
        ctx.actions.run_shell(
            inputs = [infile],
            outputs = [outfile],
            arguments = [infile.path, outfile.path],
            command = "cp $1 $2",
        )
        outfiles.append(outfile)

    return outfiles

def _py_namespace(ctx):
    # Copy all input files underneath the namesapce directory and return a
    # Provider with the new file locations.
    outfiles = []

    if ctx.file.init:
        outfiles.append(_relocate_init(ctx))

    outfiles.extend(_relocate_deps(ctx))

    return [
        DefaultInfo(files = depset(outfiles)),
    ]

py_namespace = rule(
    implementation = _py_namespace,
    attrs = {
        "init": attr.label(
            doc = "optional file for __init__.py",
            allow_single_file = [".py"],
            mandatory = False,
        ),
        "namespace": attr.string(
            doc = "name for Python namespace",
            mandatory = True,
        ),
        "deps": attr.label_list(
            doc = "list of py_library() and py_package()s to include",
            mandatory = True,
        ),
    },
)
