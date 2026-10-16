# TensorFlow C API Headers (`tf_c_headers`)

This directory contains the minimal set of pure C headers from TensorFlow
needed to compile TensorFlow custom-op shared library plugins (`.so`) without
depending on TensorFlow or its C++ framework headers at build time.

## Contents

- `tensorflow/c/ops.h`: C API for registering custom ops and shape inference.
- `tensorflow/c/kernels.h`: C API for registering custom op kernels.
- `tensorflow/c/c_api.h`, `c_api_macros.h`, `tf_attrtype.h`, `tf_buffer.h`,
  `tf_datatype.h`, `tf_file_statistics.h`, `tf_status.h`, `tf_tensor.h`,
  `tf_tstring.h`: Supporting C API types.
- `tensorflow/c/experimental/stream_executor/stream_executor.h`,
  `tensorflow/compiler/xla/tsl/c/tsl_status.h`,
  `tensorflow/tsl/platform/ctstring.h`, `ctstring_internal.h`, `platform.h`:
  Transitive pure-C headers included by `ops.h` and `kernels.h`.

## Build & Runtime Model

- **Compile time**: Custom ops include `tensorflow/c/ops.h` and
  `tensorflow/c/kernels.h` with `-DIS_SLIM_BUILD=1`.
- **Link time**: Custom op `.so` libraries link with
  `-Wl,--unresolved-symbols=ignore-in-object-files` (Linux) or
  `-Wl,-undefined,dynamic_lookup` (macOS) so pure C `TF_*` symbols resolve
  dynamically when loaded by Python `tf.load_op_library()`.
- **Google3 mapping**: Targets in `BUILD` (`:ops`, `:kernels`, `:tf_datatype`,
  `:tf_status`, `:tf_tensor`) match `//third_party/tensorflow/c:*` 1-to-1 for
  Copybara sync.
