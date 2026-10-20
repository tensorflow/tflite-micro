"""Build rules for TensorFlow custom op plugins and signal tests."""

load(
    "//python:py_rules.bzl",
    _py_tflm_signal_library = "py_tflm_signal_library",
    _tflm_signal_kernel_library = "tflm_signal_kernel_library",
    _tflm_signal_op_test = "tflm_signal_op_test",
)

py_tflm_signal_library = _py_tflm_signal_library
tflm_signal_kernel_library = _tflm_signal_kernel_library
tflm_signal_op_test = _tflm_signal_op_test
