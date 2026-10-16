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

void StackerShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  int num_channels = 0;
  int stacker_left_context = 0;
  int stacker_right_context = 0;
  if (!ctx.GetAttr("num_channels", &num_channels) ||
      !ctx.GetAttr("stacker_left_context", &stacker_left_context) ||
      !ctx.GetAttr("stacker_right_context", &stacker_right_context)) {
    return;
  }

  int64_t output_frames = stacker_left_context + 1 + stacker_right_context;
  ShapeHandle in = ctx.WithRank(ctx.Input(0), 1);
  ctx.CheckDim(in, 0, num_channels);
  ctx.SetOutput(0, ctx.VectorFromSize(num_channels * output_frames));
  ctx.SetOutput(1, ctx.Scalar());
}

// Stack several input frames into a single stacked frame. On each invocation,
// it generates a stacked frame that contains: (stacker_left_context + 1 +
// stacker_right_context) consecutive unstacked frames. The stacked frame
// becomes the input to the neural network. The stacker then moves forward in a
// step of one or more input frames.
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalStackerOp() {
  OpDefBuilder("SignalStacker")
      .Attr("num_channels: int >= 1")
      .Attr("stacker_left_context: int >= 0")
      .Attr("stacker_right_context: int >= 0")
      .Attr("stacker_step: int >= 1")
      .Input("input: int16")
      .Output("output: int16")
      .Output("output_valid: bool")
      .SetShapeFn(&StackerShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
