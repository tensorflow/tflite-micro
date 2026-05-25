"""Bazel macros wrapping @rules_python's py_* rules for use in this repository.

The tflm_py_library, tflm_py_test, and tflm_py_binary wrappers inject a
dependency on //:tflite_micro_shim, which synthesizes the "tflite_micro"
top-level package namespace at import time (see //:tflite_micro.py).

The shim is necessary because, under Bzlmod, the main repository's runfiles root
is the fixed name "_main" rather than the module name, so the "tflite_micro"
import prefix used throughout this repository's Python sources (e.g. `from
tflite_micro.tensorflow.lite... import x`) no longer resolves on its own as it
did under the legacy WORKSPACE, where the runfiles root took the workspace name.

Any py_* target whose code imports "tflite_micro.*" must depend on the shim.
These wrappers add that dependency automatically so it need not be repeated in
every target. Prefer them over the native py_library/py_test/py_binary for any
target under the tflite_micro namespace.
"""

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

def tflm_py_library(deps = [], **kwargs):
    py_library(
        deps = deps + ["//:tflite_micro_shim"],
        **kwargs
    )

def tflm_py_binary(deps = [], **kwargs):
    py_binary(
        deps = deps + ["//:tflite_micro_shim"],
        **kwargs
    )

def tflm_py_test(deps = [], **kwargs):
    py_test(
        deps = deps + ["//:tflite_micro_shim"],
        **kwargs
    )
