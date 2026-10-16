/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

void WindowShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  ShapeHandle weights = ctx.WithRank(ctx.Input(1), 1);
  ShapeHandle out = ctx.WithRankAtLeast(ctx.Input(0), 1);
  DimensionHandle dim_in = ctx.Dim(out, -1);
  if (ctx.ValueKnown(ctx.Dim(weights, 0))) {
    int64_t expected = ctx.ValueKnown(dim_in) ? ctx.Value(dim_in) : -1;
    ctx.CheckDim(weights, 0, expected);
  }
  ctx.SetOutput(0, out);
}

// TODO(b/286250473): change back name to "Window" after name clash resolved
__attribute__((constructor)) void RegisterSignalWindowOp() {
  OpDefBuilder("SignalWindow")
      .Attr("shift: int")
      .Input("input: int16")
      .Input("weights: int16")
      .Output("output: int16")
      .SetShapeFn(&WindowShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
