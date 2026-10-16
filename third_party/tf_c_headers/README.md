# TensorFlow C API Headers (`tf_c_headers`)

This directory contains the minimal set of pure C headers from TensorFlow
needed to compile TensorFlow custom-op shared library plugins (`.so`) without
depending on TensorFlow or its C++ framework headers at build time.

Currently synced to TensorFlow **v2.20.0**
(`72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa`).

## Updating Headers

Use `sync.sh` to update the headers from upstream
`https://github.com/tensorflow/tensorflow`. All files are copied verbatim
without patching `#include` paths.

```bash
# Sync to the default pinned commit/tag in sync.sh (downloads from GitHub):
./third_party/tf_c_headers/sync.sh

# Sync to a newer TensorFlow release tag or commit SHA from GitHub:
./third_party/tf_c_headers/sync.sh --commit v2.21.0

# Sync from a local TensorFlow git checkout (at a specific tag or working tree):
./third_party/tf_c_headers/sync.sh --tf-dir /path/to/tensorflow --commit v2.20.0
```

When bumping the default version permanently, update `DEFAULT_TF_COMMIT` in
`sync.sh` and the version note above.

## Contents

- `tensorflow/c/ops.h`: C API for registering custom ops and shape inference.
- `tensorflow/c/kernels.h`: C API for registering custom op kernels.
- `tensorflow/c/c_api.h`, `c_api_macros.h`, `tf_attrtype.h`, `tf_buffer.h`,
  `tf_datatype.h`, `tf_status.h`, `tf_tensor.h`, `tf_tstring.h`: Supporting C
  API types.
- `tensorflow/c/experimental/stream_executor/stream_executor.h`,
  `tensorflow/core/platform/ctstring.h`, `xla/tsl/c/tsl_status.h`,
  `tsl/platform/ctstring.h`, `tsl/platform/ctstring_internal.h`,
  `tsl/platform/platform.h`: Transitive pure-C headers included by `ops.h` and
  `kernels.h`.

## Build & Runtime Model

- **Compile time**: Custom ops include `tensorflow/c/ops.h` and
  `tensorflow/c/kernels.h` with `-DIS_SLIM_BUILD=1`.
- **Link time**: Custom op `.so` libraries link with
  `-Wl,--unresolved-symbols=ignore-in-object-files` (Linux) or
  `-Wl,-undefined,dynamic_lookup` (macOS) so pure C `TF_*` symbols resolve
  dynamically when loaded by Python `tf.load_op_library()`.
- **Google3 mapping**: Targets in `BUILD` (`:ops`, `:kernels`, `:tf_datatype`,
  `:tf_status`, `:tf_tensor`) match `//third_party/tensorflow/c:*` 1-to-1 for
  Copybara sync. This directory is excluded from Google3 import because Google3
  builds directly against `//third_party/tensorflow/c:*`.
