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

#include <cstdint>

#include "signal/tensorflow_core/ops/shape_inference_utils.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void OverlapAddShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  int frame_step = 0;
  if (!GetAttrInt(ctx, "frame_step", &frame_step, status)) {
    return;
  }

  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* in = TF_NewShapeHandle();
  TF_ShapeHandle* sub = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();

  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, in0, 2, in, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSubshape(ctx, in, 0, -1, sub, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_DimensionHandle* frames_dim = TF_NewDimensionHandle();
    TF_ShapeInferenceContextDim(ctx, sub, -1, frames_dim);
    if (TF_DimensionHandleValueKnown(frames_dim)) {
      int64_t n_frames = TF_DimensionHandleValue(frames_dim);
      if (ReplaceLastDim(ctx, sub, n_frames * frame_step, out, status)) {
        TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
      }
    } else {
      // `sub` already has an unknown last dimension (or unknown rank).
      TF_ShapeInferenceContextSetOutput(ctx, 0, sub, status);
    }
    TF_DeleteDimensionHandle(frames_dim);
  }

  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(sub);
  TF_DeleteShapeHandle(in);
  TF_DeleteShapeHandle(in0);
}

// Transform an input signal made of overlapping frames of size frame_size into
// an output signal made of frames of size frame_step.
// The overlapping input frames are spaced frame_step apart in time.
// The op adds the overlapping frames into the output frame.
//
// input: A [..., frames, frame_length] Tensor. Rank must be at least 2.
// output: A Tensor with shape [..., output_size] containing the overlap-added
// frames of signal's inner-most two dimensions.
//
// frame_step: The number of output samples to progress the 'current'
// sample on each invocation. Also the number of samples in each output frame.
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalOverlapAddOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder =
      TF_NewOpDefinitionBuilder("SignalOverlapAdd");
  TF_OpDefinitionBuilderAddAttr(builder, "T: {float, int16}");
  TF_OpDefinitionBuilderAddAttr(builder, "frame_step: int >= 1");
  TF_OpDefinitionBuilderAddInput(builder, "input: T");
  TF_OpDefinitionBuilderAddOutput(builder, "output: T");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &OverlapAddShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
