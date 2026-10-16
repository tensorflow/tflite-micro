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
#include <cstdint>

#include "signal/tensorflow_core/ops/shape_inference_utils.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void FramerShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  int frame_step = 0;
  int frame_size = 0;
  if (!GetAttrInt(ctx, "frame_step", &frame_step, status) ||
      !GetAttrInt(ctx, "frame_size", &frame_size, status)) {
    return;
  }

  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* in = TF_NewShapeHandle();
  TF_ShapeHandle* prefix_with_frames = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeHandle* scalar = TF_ShapeInferenceContextScalar(ctx);

  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, in0, 1, in, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_DimensionHandle* last_dim = TF_NewDimensionHandle();
    TF_ShapeInferenceContextDim(ctx, in, -1, last_dim);
    TF_ShapeHandle* frame_size_vec = TF_ShapeInferenceContextVectorFromSize(
        ctx, static_cast<size_t>(frame_size));
    if (TF_DimensionHandleValueKnown(last_dim)) {
      int64_t n_frames = TF_DimensionHandleValue(last_dim) / frame_step;
      if (ReplaceLastDim(ctx, in, n_frames, prefix_with_frames, status)) {
        TF_ShapeInferenceContextConcatenateShapes(ctx, prefix_with_frames,
                                                  frame_size_vec, out, status);
      }
    } else {
      // `in` already has an unknown last dimension (or unknown rank).
      TF_ShapeInferenceContextConcatenateShapes(ctx, in, frame_size_vec, out,
                                                status);
    }
    TF_DeleteShapeHandle(frame_size_vec);
    TF_DeleteDimensionHandle(last_dim);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 1, scalar, status);
  }

  TF_DeleteShapeHandle(scalar);
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(prefix_with_frames);
  TF_DeleteShapeHandle(in);
  TF_DeleteShapeHandle(in0);
}

// Transform an input signal into a series of overlapping frames, each of
// size frame_size. The frame_step determines how many samples the framer
// progresses on each invocation. When the framer has enough samples to produce
// a frame, it writes it to the output tensor and sets the output_valid to True.
// If the framer doesn't have enough samples to produce a frame, it doesn't
// initialize the contents of the output tensor and sets the output_valid
// boolean to False.
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFramerOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SignalFramer");
  TF_OpDefinitionBuilderAddAttr(builder, "frame_size: int >= 1");
  TF_OpDefinitionBuilderAddAttr(builder, "frame_step: int >= 1");
  TF_OpDefinitionBuilderAddAttr(builder, "prefill: bool");
  TF_OpDefinitionBuilderAddInput(builder, "input: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output_valid: bool");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &FramerShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
