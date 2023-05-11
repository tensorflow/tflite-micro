load("@rules_python//python:pip.bzl", "compile_pip_requirements")

licenses(["notice"])

compile_pip_requirements(
    # Defines targets which use pip-compile to keep the Python locked
    # requirements up-to-date:
    #
    # :python_requirements.update  bazel run this target to update
    #                              ./python_requirements.txt by recursively following
    #                              and locking the dependencies seeded by
    #                              ./python_requirements.in
    #
    # :python_requirements_test    bazel test target which fails if
    #                              ./python_requirements.txt does not match
    #                              that generated from ./python_requirements.in
    name = "python_requirements",
    extra_args = [
        "--allow-unsafe",
        # ^ lets pip-compile include setuptools, recommended by
        # `pip-compile -h` as future default behavior
    ],
    requirements_in = "python_requirements.in",
    requirements_txt = "python_requirements.txt",
    tags = [
        "manual",
        # ^ exclude .update and _test targets from wildcards in,
        # e.g., `bazel test ...`
    ],
)
