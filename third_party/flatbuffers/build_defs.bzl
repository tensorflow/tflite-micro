"""BUILD rules for generating flatbuffer files."""

load(
    "@flatbuffers//:build_defs.bzl",
    _upstream_flatbuffer_cc_library = "flatbuffer_cc_library",
)
load("@rules_python//python:defs.bzl", "py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

DEFAULT_FLATC_ARGS = [
    "--no-union-value-namespacing",
    "--gen-object-api",
]

def flatbuffer_cc_library(name, flatc_args = DEFAULT_FLATC_ARGS, cc_include_paths = [], **kwargs):
    """Wrapper around upstream flatbuffer_cc_library with TFLM-specific defaults."""
    _upstream_flatbuffer_cc_library(
        name = name,
        flatc_args = flatc_args,
        cc_include_paths = cc_include_paths + ["."],
        **kwargs
    )

def flatbuffer_py_library(
        name,
        srcs,
        deps = [],
        include_paths = []):
    """A py_library with the generated reader/writers for the given schema."""

    # Generate the python source file using flatc --python --gen-onefile.
    # The output will be [name]_generated.py to match existing imports.
    srcs_lib = "%s_srcs" % (name)
    out_file = "%s_generated.py" % name

    # We use a genrule instead of upstream flatbuffer_library_public
    # because we need to rename the output file to [name]_generated.py
    # to maintain backward compatibility with existing imports.
    include_paths_cmd = ["-I %s" % (s) for s in include_paths]

    native.genrule(
        name = srcs_lib,
        srcs = srcs,
        outs = [out_file],
        tools = ["@flatbuffers//:flatc"],
        cmd = " ".join([
            "for f in $(SRCS); do",
            "$(location @flatbuffers//:flatc)",
            "--python --gen-onefile",
            " ".join(DEFAULT_FLATC_ARGS),
            " ".join(include_paths_cmd),
            "-o $(@D)",
            "$$f;",
            # flatc --gen-onefile produces [fbs_basename]_generated.py
            # we move it to the expected out_file.
            "fbs_base=$$(basename $$f .fbs);",
            "if [ \"$$fbs_base\"_generated.py != \"" + out_file + "\" ]; then",
            "mv $(@D)/\"$$fbs_base\"_generated.py $(@D)/" + out_file + ";",
            "fi;",
            "done",
        ]),
        message = "Generating flatbuffer python files for %s:" % (name),
    )

    py_library(
        name = name,
        srcs = [out_file],
        deps = deps + [
            requirement("flatbuffers"),
        ],
        visibility = ["//visibility:public"],
    )
