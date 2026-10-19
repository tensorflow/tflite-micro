"""BUILD rules for generating flatbuffer files."""

load(
    "@flatbuffers//:build_defs.bzl",
    "flatbuffer_library_public",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_library")

DEFAULT_FLATC_ARGS = [
    "--no-union-value-namespacing",
    "--gen-object-api",
]

def flatbuffer_cc_library(
        name,
        srcs,
        flatc_args = DEFAULT_FLATC_ARGS,
        cc_include_paths = [],
        deps = [],
        includes = [],
        include_paths = None,
        out_prefix = "",
        visibility = None,
        **kwargs):
    """Generates C++ code for reading/writing flatbuffers using only runtime_cc."""
    output_headers = [
        (out_prefix + "%s_generated.h") % (s.replace(".fbs", "").split("/")[-1].split(":")[-1])
        for s in srcs
    ]
    srcs_lib = "%s_srcs" % name
    flatbuffer_library_public(
        name = srcs_lib,
        srcs = srcs,
        outs = output_headers,
        language_flag = "-c",
        out_prefix = out_prefix,
        includes = includes,
        include_paths = include_paths,
        flatc_args = flatc_args,
        visibility = visibility,
        **kwargs
    )
    cc_library(
        name = name,
        hdrs = [":" + srcs_lib],
        srcs = [":" + srcs_lib],
        features = ["-parse_headers"],
        deps = ["@flatbuffers//:runtime_cc"] + deps,
        includes = cc_include_paths + ["."],
        linkstatic = 1,
        visibility = visibility,
    )
    native.filegroup(
        name = "%s_includes" % name,
        srcs = srcs + includes,
        visibility = visibility,
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

    tflm_py_library(
        name = name,
        srcs = [out_file],
        deps = deps + [
            requirement("flatbuffers"),
            requirement("numpy"),
        ],
        visibility = ["//visibility:public"],
    )
