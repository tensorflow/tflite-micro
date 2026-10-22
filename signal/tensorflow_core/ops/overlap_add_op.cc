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

void OverlapAddShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  int frame_step = 0;
  if (!ctx.GetAttr("frame_step", &frame_step)) {
    return;
  }

  ShapeHandle in = ctx.WithRankAtLeast(ctx.Input(0), 2);
  ShapeHandle sub = ctx.Subshape(in, 0, -1);
  DimensionHandle frames_dim = ctx.Dim(sub, -1);
  if (ctx.ValueKnown(frames_dim)) {
    int64_t n_frames = ctx.Value(frames_dim);
    ctx.SetOutput(0, ctx.ReplaceLastDim(sub, n_frames * frame_step));
  } else {
    ctx.SetOutput(0, sub);
  }
}

// Registers the SignalOverlapAdd custom op (documented in overlap_add_op.py).
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalOverlapAddOp() {
  SignalOpDefBuilder("SignalOverlapAdd")
      .Attr("T: {float, int16}")
      .Attr("frame_step: int >= 1")
      .Input("input: T")
      .Output("output: T")
      .SetShapeFn(&OverlapAddShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
