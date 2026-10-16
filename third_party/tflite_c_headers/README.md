# TFLite Opaque C API Headers (`tflite_c_headers`)

This directory contains the minimal set of pure C headers from TensorFlow Lite
needed to compile TFLite and LiteRT custom-op kernels (`TfLiteOperator` and
`TfLiteRegistration`) without depending on the full TensorFlow Lite C++ runtime
at build time.

## Contents

- `tensorflow/lite/c/c_api_opaque.h` and `tensorflow/lite/core/c/c_api_opaque.h`:
  Opaque C API for accessing tensors, nodes, and contexts in custom ops.
- `tensorflow/lite/core/c/operator.h`: `TfLiteOperator` API (`TfLiteOperatorCreate`,
  `TfLiteOperatorSetInit`, `SetFree`, `SetPrepare`, `SetInvoke`).
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
  `//third_party/tensorflow/lite/c:*` 1-to-1 for Copybara sync.
