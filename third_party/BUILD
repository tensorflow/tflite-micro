licenses(["notice"])

load("@rules_python//python/pip_install:requirements.bzl", "compile_pip_requirements")

compile_pip_requirements(
    # Defines targets which use pip-compile to keep the Python locked
    # requirements.txt up-to-date.
    #
    # :requirements.update  bazel run this target to update ./requirements.txt by
    #                       recursively following and locking the dependencies
    #                       seeded by ./requirements.in
    #
    # :requirements_test    bazel test target which fails if requirements.txt
    #                       falls behind
    name = "requirements",
    requirements_in = "requirements.in",
    extra_args = [
        "--allow-unsafe", # lets pip-compile include setuptools, recommended by
                          # `pip-compile -h` as future default behavior
    ],
)
