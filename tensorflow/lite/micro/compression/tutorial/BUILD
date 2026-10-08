load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "INCOMPATIBLE_WITH_WINDOWS",
)

py_binary(
    # Converts a jupytext markdown notebook to .ipynb with deterministic cell
    # ids, so that generated output can be diffed against a saved copy.
    name = "md_to_ipynb",
    srcs = ["md_to_ipynb.py"],
    deps = [requirement("jupytext")],
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
    cmd = "$(location :md_to_ipynb) " +
          "$(location mnist_compression_tutorial.md) $@",
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
    tools = [":md_to_ipynb"],
)

sh_binary(
    # Copies the generated notebook into the source tree as the saved
    # notebook. `bazel run` this target after editing the markdown source.
    # See :ipynb above.
    name = "ipynb.update",
    srcs = ["ipynb_update.sh"],
    args = ["$(location :ipynb)"],
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
        "$(location mnist_compression_tutorial.ipynb)",
        "$(location :ipynb)",
    ],
    data = [
        "mnist_compression_tutorial.ipynb",
        ":ipynb",
    ],
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)

py_binary(
    # Opens the tutorial's markdown source in JupyterLab for interactive
    # editing and execution, in a self-managed Python environment that
    # mirrors Google Colab; see notebook_tool.py.
    name = "lab",
    srcs = ["notebook_tool.py"],
    args = ["lab"],
    main = "notebook_tool.py",
    target_compatible_with = INCOMPATIBLE_WITH_WINDOWS,
)

py_binary(
    # Executes the tutorial notebook end to end in the same environment as
    # :lab and writes the executed notebook, with outputs, to the invocation
    # directory; see notebook_tool.py.
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
        "//tensorflow/lite/micro/tools:tflite_flatbuffer_align",
        "//tensorflow/lite/python:schema_py",
        requirement("numpy"),
    ],
)
