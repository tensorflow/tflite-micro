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

#include <cstddef>

#include "signal/tensorflow_core/ops/shape_inference_utils.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void StackerShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  int num_channels = 0;
  int stacker_left_context = 0;
  int stacker_right_context = 0;
  if (!GetAttrInt(ctx, "num_channels", &num_channels, status) ||
      !GetAttrInt(ctx, "stacker_left_context", &stacker_left_context, status) ||
      !GetAttrInt(ctx, "stacker_right_context", &stacker_right_context,
                  status)) {
    return;
  }

  int output_frames = stacker_left_context + 1 + stacker_right_context;
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* ranked = TF_NewShapeHandle();
  TF_ShapeHandle* scalar = TF_ShapeInferenceContextScalar(ctx);

  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in0, 1, ranked, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    CheckDimValue(ctx, ranked, 0, num_channels, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeHandle* out = TF_ShapeInferenceContextVectorFromSize(
        ctx, static_cast<size_t>(num_channels) * output_frames);
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
    TF_DeleteShapeHandle(out);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 1, scalar, status);
  }

  TF_DeleteShapeHandle(scalar);
  TF_DeleteShapeHandle(ranked);
  TF_DeleteShapeHandle(in0);
}

// Stack several input frames into a single stacked frame. On each invocation,
// it generates a stacked frame that contains: (stacker_left_context + 1 +
// stacker_right_context) consecutive unstacked frames. The stacked frame
// becomes the input to the neural network. The stacker then moves forward in a
// step of one or more input frames.
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalStackerOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SignalStacker");
  TF_OpDefinitionBuilderAddAttr(builder, "num_channels: int >= 1");
  TF_OpDefinitionBuilderAddAttr(builder, "stacker_left_context: int >= 0");
  TF_OpDefinitionBuilderAddAttr(builder, "stacker_right_context: int >= 0");
  TF_OpDefinitionBuilderAddAttr(builder, "stacker_step: int >= 1");
  TF_OpDefinitionBuilderAddInput(builder, "input: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output_valid: bool");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &StackerShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
