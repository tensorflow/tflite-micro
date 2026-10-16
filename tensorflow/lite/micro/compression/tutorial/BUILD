load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("//python:py_rules.bzl", "tflm_py_test")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "INCOMPATIBLE_WITH_WINDOWS",
)

py_library(
    name = "md_to_ipynb_lib",
    srcs = ["md_to_ipynb.py"],
    deps = [requirement("pyyaml")],
)

py_binary(
    # Converts a jupytext markdown notebook to .ipynb with deterministic cell
    # ids, so that generated output can be diffed against a saved copy.
    name = "md_to_ipynb",
    srcs = ["md_to_ipynb.py"],
    deps = [":md_to_ipynb_lib"],
)

tflm_py_test(
    name = "md_to_ipynb_test",
    size = "small",
    srcs = ["md_to_ipynb_test.py"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [":md_to_ipynb_lib"],
)

genrule(
    # The tutorial is authored and edited as markdown in
    # "mnist_compression_tutorial.md". This rule generates the equivalent
    # .ipynb notebook, which is saved to the source tree and committed to git
    # as "mnist_compression_tutorial.ipynb" so that Google Colab, which cannot
    # open markdown notebooks, can open it straight from GitHub.
    #
    # When the markdown source changes, update the saved notebook by running
    # the target ":ipynb.update".
    #
    # Committing the generated file risks inconsistency between the markdown
    # source and the saved notebook, so consistency is ensured by the test
    # ":ipynb_test".
    name = "ipynb",
    srcs = ["mnist_compression_tutorial.md"],
    outs = ["mnist_compression_tutorial_generated.ipynb"],
    cmd = "$(execpath :md_to_ipynb) " +
          "$(execpath mnist_compression_tutorial.md) $@",
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    tools = [":md_to_ipynb"],
)

sh_binary(
    # Copies the generated notebook into the source tree as the saved
    # notebook. `bazel run` this target after editing the markdown source.
    # See :ipynb above.
    name = "ipynb.update",
    srcs = ["ipynb_update.sh"],
    args = ["$(rootpath :ipynb)"],
    data = [":ipynb"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)

sh_test(
    # Ensures consistency between the tutorial's markdown source and the
    # saved generated notebook. Fails if they mismatch, in which case
    # ":ipynb.update" should be run. See :ipynb above.
    name = "ipynb_test",
    size = "small",
    srcs = ["ipynb_test.sh"],
    args = [
        "$(rootpath mnist_compression_tutorial.ipynb)",
        "$(rootpath :ipynb)",
    ],
    data = [
        "mnist_compression_tutorial.ipynb",
        ":ipynb",
    ],
    tags = ["notap"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)

py_binary(
    # Starts JupyterLab on the tutorial.
    name = "lab",
    srcs = ["notebook_tool.py"],
    args = ["lab"],
    main = "notebook_tool.py",
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)

py_binary(
    # Runs the tutorial end to end.
    name = "run",
    srcs = ["notebook_tool.py"],
    args = ["execute"],
    data = ["notebook_execute.py"],
    main = "notebook_tool.py",
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)

py_binary(
    # Builds a small model with model_editor, compresses one of its tensors
    # with the compression tools, and writes both the original and the
    # compressed model as .tflite files; see model_editor_example.py.
    name = "model_editor_example",
    srcs = ["model_editor_example.py"],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    deps = [
        "//tensorflow/lite/micro/compression:compress_lib",
        "//tensorflow/lite/micro/compression:decode_insert",
        "//tensorflow/lite/micro/compression:lut",
        "//tensorflow/lite/micro/compression:model_editor",
        "//tensorflow/lite/micro/compression:spec",
        "//tensorflow/lite/micro/compression:spec_builder",
        "//tensorflow/lite/micro/python:schema_py",
        "//tensorflow/lite/micro/tools:tflite_flatbuffer_align",
        requirement("numpy"),
    ],
)
