load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

# To update requirements for a specific version, run:
# bazel run //python:requirements_3_10.update --@rules_python//python/config_settings:python_version=3.10
compile_pip_requirements(
    name = "requirements_3_10",
    extra_args = ["--allow-unsafe"],
    requirements_in = "python_requirements.in",
    requirements_txt = "python_requirements_3_10.txt",
    tags = [
        "manual",
        "notap",
    ],
)

# bazel run //python:requirements_3_11.update --@rules_python//python/config_settings:python_version=3.11
compile_pip_requirements(
    name = "requirements_3_11",
    extra_args = ["--allow-unsafe"],
    requirements_in = "python_requirements.in",
    requirements_txt = "python_requirements_3_11.txt",
    tags = [
        "manual",
        "notap",
    ],
)

# bazel run //python:requirements_3_12.update --@rules_python//python/config_settings:python_version=3.12
compile_pip_requirements(
    name = "requirements_3_12",
    extra_args = ["--allow-unsafe"],
    requirements_in = "python_requirements.in",
    requirements_txt = "python_requirements_3_12.txt",
    tags = [
        "manual",
        "notap",
    ],
)

# bazel run //python:requirements_3_13.update --@rules_python//python/config_settings:python_version=3.13
compile_pip_requirements(
    name = "requirements_3_13",
    extra_args = ["--allow-unsafe"],
    requirements_in = "python_requirements.in",
    requirements_txt = "python_requirements_3_13.txt",
    tags = [
        "manual",
        "notap",
    ],
)
