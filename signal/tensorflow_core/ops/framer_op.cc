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

void FramerShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  int frame_step = 0;
  int frame_size = 0;
  if (!ctx.GetAttr("frame_step", &frame_step) ||
      !ctx.GetAttr("frame_size", &frame_size)) {
    return;
  }

  ShapeHandle in = ctx.WithRankAtLeast(ctx.Input(0), 1);
  DimensionHandle last_dim = ctx.Dim(in, -1);
  ShapeHandle frame_size_vec = ctx.VectorFromSize(frame_size);
  ShapeHandle out;
  if (ctx.ValueKnown(last_dim)) {
    int64_t n_frames = ctx.Value(last_dim) / frame_step;
    out = ctx.Concatenate(ctx.ReplaceLastDim(in, n_frames), frame_size_vec);
  } else {
    out = ctx.Concatenate(in, frame_size_vec);
  }
  ctx.SetOutput(0, out);
  ctx.SetOutput(1, ctx.Scalar());
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
  OpDefBuilder("SignalFramer")
      .Attr("frame_size: int >= 1")
      .Attr("frame_step: int >= 1")
      .Attr("prefill: bool")
      .Input("input: int16")
      .Output("output: int16")
      .Output("output_valid: bool")
      .SetShapeFn(&FramerShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
