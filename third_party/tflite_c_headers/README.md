# TFLite Opaque C API Headers (`tflite_c_headers`)

This directory contains the minimal set of pure C headers from TensorFlow Lite
needed to compile TFLite and LiteRT custom-op kernels (`TfLiteOperator` and
`TfLiteRegistration`) without depending on the full TensorFlow Lite C++ runtime
at build time.

Currently synced to TensorFlow **v2.20.0**
(`72fbba3d20f4616d7312b5e2b7f79daf6e82f2fa`).

## Updating Headers

Use `sync.sh` to update the headers from upstream
`https://github.com/tensorflow/tensorflow`. All files keep their upstream paths
and are copied verbatim without patching `#include` lines.

```bash
# Sync to the default pinned commit/tag in sync.sh (downloads from GitHub):
./third_party/tflite_c_headers/sync.sh

# Sync to a newer TensorFlow release tag or commit SHA from GitHub:
./third_party/tflite_c_headers/sync.sh --commit v2.21.0

# Sync from a local TensorFlow git checkout (at a specific tag or working tree):
./third_party/tflite_c_headers/sync.sh --tf-dir /path/to/tensorflow --commit v2.20.0
```

When bumping the default version permanently, update `DEFAULT_TF_COMMIT` in
`sync.sh` and the version note above.

## Contents

- `tensorflow/lite/c/c_api_opaque.h` and `tensorflow/lite/core/c/c_api_opaque.h`:
  Opaque C API for accessing tensors, nodes, and contexts in custom ops.
- `tensorflow/lite/core/c/operator.h`: `TfLiteOperator` API
  (`TfLiteOperatorCreate`, `TfLiteOperatorSetInitWithData`, `SetFreeWithData`,
  `SetPrepareWithData`, `SetInvokeWithData`).
- `tensorflow/lite/c/common.h` and `tensorflow/lite/core/c/common.h`:
  `TfLiteRegistration` and common C types.
- `tensorflow/lite/c/c_api.h`, `tensorflow/lite/c/c_api_types.h`,
  `tensorflow/lite/core/c/c_api.h`, `tensorflow/lite/core/c/c_api_types.h`,
  `tensorflow/lite/builtin_ops.h`, `tensorflow/lite/core/async/c/types.h`,
  `tensorflow/compiler/mlir/lite/core/c/tflite_types.h`: Transitive pure-C
  headers included by `c_api_opaque.h` and `common.h`.

## Build & Runtime Model

- **Compile time**: Custom ops include `tensorflow/lite/c/c_api_opaque.h` and
  `tensorflow/lite/c/common.h`.
- **Link time**: Custom op targets link against TFLite (`libtensorflowlite_c`)
  or LiteRT (`LiteRtOpOptions` / `AddCustomOp`) in the host application.
- **Google3 mapping**: Targets in `BUILD` (`:c_api_opaque`, `:common`) match
  `//third_party/tensorflow/lite/c:*` 1-to-1 for Copybara sync. This directory
  is excluded from Google3 import because Google3 builds directly against
  `//third_party/tensorflow/lite/c:*`.
