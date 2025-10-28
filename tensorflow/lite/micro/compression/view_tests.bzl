load("@rules_python//python:defs.bzl", "py_test")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

def generate_view_tests(targets):
    """Generates py_test targets for each target's path and a test_suite to
    group them.

    Args:
        targets: List of target labels to .tflite models with which to test.
    """
    test_names = []
    for target in targets:
        # Create a test name from the last component of the target name
        short_name = target.split(":")[-1] if ":" in target else target.split("/")[-1]
        test_name = "view_test_{}".format(short_name.replace(".", "_"))
        py_test(
            name = test_name,
            srcs = ["view_test.py"],
            args = ["$(location {})".format(target)],
            main = "view_test.py",
            data = [target],
            deps = [
                ":view",
                requirement("absl_py"),
            ],
            size = "small",
        )
        test_names.append(test_name)

    # Create a test suite for all generated tests
    native.test_suite(
        name = "view_tests",
        tests = test_names,
    )
