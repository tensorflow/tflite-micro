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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {
namespace signal {

absl::Status StackerShape(InferenceContext* c) {
  int num_channels, stacker_left_context, stacker_right_context;
  TF_RETURN_IF_ERROR(c->GetAttr<int>("num_channels", &num_channels));
  TF_RETURN_IF_ERROR(
      c->GetAttr<int>("stacker_left_context", &stacker_left_context));
  TF_RETURN_IF_ERROR(
      c->GetAttr<int>("stacker_right_context", &stacker_right_context));

  int output_frames = stacker_left_context + 1 + stacker_right_context;

  ShapeHandle out;
  shape_inference::DimensionHandle dim_in;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(c->input(0), 0), num_channels, &dim_in));
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(out, 0, c->MakeDim(num_channels * output_frames), &out));
  c->set_output(0, out);
  c->set_output(1, c->Scalar());
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalStacker")
    .Attr("num_channels: int >= 1")
    .Attr("stacker_left_context: int >= 0")
    .Attr("stacker_right_context: int >= 0")
    .Attr("stacker_step: int >= 1")
    .Input("input: int16")
    .Output("output: int16")
    .Output("output_valid: bool")
    .SetShapeFn([](InferenceContext* c) { return StackerShape(c); })
    .Doc(R"doc(
Stack several input frames into a single stacked frame. On each invocation, it
generates a stacked frame that contains:
(stacker_left_context + 1 + stacker_right_context)
consecutive unstacked frames. The stacked frame which becomes the input to the
neural network. The stacker then moves forward in a step of one or more input
frames.
For example, assuming a squence of 10 input frames, where each input frame
is itself a vector of size num_channels:
[1 2 3 4 5 6 7 8 9 0], and the current frame is 9, the following configuration:
stacker_left_context=0
stacker_right_context=1
stacker_step=2
will produce 5 stacked frames:
[1,2] [3,4] [5,6] [7,8] [9,0].

input: A 1-D input frame of size num_channels
output: A 1-D output frame of size
       (stacker_left_context + 1 + stacker_right_context) * num_channels
output_valid: A boolean scalar.
              If true, the output is a valid output frame
              If false, the output is an invalid output frame of all zeros
              Once the stacker produces its first output frame, its output will
              be valid every stacker_step input frames.
num_channels: the number of filter bank channels in each stacker input frame
stacker_left_context: The number of input frames to the left of the current
                      frame to include in the output frame.
stacker_right_context: The number of input frames to the right of the current
                       frame to include in the output frame.
stacker_step: The number of input frames to increment the stacker's 'current'
              frame on each invocation.
)doc");

}  // namespace signal
}  // namespace tensorflow
