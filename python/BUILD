load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

# To update requirements for a specific version, run:
# bazel run //python:requirements_3_10.update --@rules_python//python/config_settings:python_version=3.10
[
    compile_pip_requirements(
        name = "requirements_" + version.replace(".", "_"),
        extra_args = ["--allow-unsafe"],
        requirements_in = "python_requirements.in",
        requirements_txt = "python_requirements_" + version.replace(".", "_") + ".txt",
        tags = [
            "manual",
            "notap",
        ],
    )
    for version in [
        "3.10",
        "3.11",
        "3.12",
        "3.13",
    ]
]
