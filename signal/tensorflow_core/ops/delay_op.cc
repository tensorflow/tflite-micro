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

#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void DelayShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, in0, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(in0);
}

// Delay the innermost dimension of input signal by delay_length samples.
//
// For example, assuming an input signal of 10 samples,
// [1 2 3 4 5 6 7 8 9 0]
// If we input the signal to a delay op configured with delay_length=3, the op
// will produce the following output:
// [0 0 0 1 2 3 4 5 6 7]
// To retrieve the remainder of the input signal, call the delay op again with
// zeros as input:
// [0 0 0 0 0 0 0 0 0 0]
// to get the output:
// [8 9 0 0 0 0 0 0 0 0]
//
// input: A multidimensional input signal.
// output: An output signal of the same shape as the input signal. The innermost
//         dimension is delayed by delay_length samples.
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalDelayOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SignalDelay");
  TF_OpDefinitionBuilderAddAttr(builder, "delay_length: int >= 0");
  TF_OpDefinitionBuilderAddInput(builder, "input: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output: int16");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &DelayShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
