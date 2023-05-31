/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SIGNAL_MICRO_KERNELS_SIGNAL_OPS_REGISTERER_PYTHON_H_
#define SIGNAL_MICRO_KERNELS_SIGNAL_OPS_REGISTERER_PYTHON_H_

#include "signal/micro/kernels/register_signal_ops.h"
#include "tensorflow/lite/micro/python/interpreter/src/python_ops_resolver.h"

namespace tflite {


// This is a wrapper registerer function that adds the individual ops
// needed for the audio frontend as custom ops via their Register_*() API. This
// is specifically used for the TFLM interpreter only, and should not be
// confused with TFLite's registerer `AudioFrontendOpsRegisterer()`. The symbol
// name should be unique, because the registerer is passed as string to the TFLM
// Python interpreter. It takes in an AllOpsResolver parameter because that's
// the current default op resolver used by the TFLM Python API. Extern "C" is
// used to prevent symbol name mangling by the C++ compiler to preseve the
// correct symbol name needed for dynamic symbol lookup.
extern "C" bool SignalOpsRegistererMicro(
    ::tflite::PythonOpsResolver *resolver);

}  // namespace tflite

#endif  // SIGNAL_MICRO_KERNELS_SIGNAL_OPS_REGISTERER_PYTHON_H_