# tflm:github-begin
# This line of code is only required for github. This part is not required
# for google3
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
# tflm:github-end

licenses(["notice"])

# tflm:github-begin
# This block of code is only required for github. This part is not required
# for google3
compile_pip_requirements(
    # Defines targets which use pip-compile to keep the Python locked
    # requirements up-to-date:
    #
    # :python_requirements.update  bazel run this target to update
    #                              ./requirements.txt by recursively following
    #                              and locking the dependencies seeded by
    #                              ./requirements.in
    #
    # :python_requirements_test    bazel test target which fails if
    #                              python_requirements.txt falls behind
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
# tflm:github-end
