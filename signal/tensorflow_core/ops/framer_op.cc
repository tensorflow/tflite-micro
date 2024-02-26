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

absl::Status FramerShape(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle in;
  int frame_step, frame_size;

  TF_RETURN_IF_ERROR(c->GetAttr<int>("frame_step", &frame_step));
  TF_RETURN_IF_ERROR(c->GetAttr<int>("frame_size", &frame_size));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &in));
  int n_frames = c->Value(c->Dim(in, -1)) / frame_step;

  shape_inference::DimensionHandle extra_dim = c->MakeDim({frame_size});
  ShapeHandle extra_dim_shape =
      c->MakeShape(std::vector<shape_inference::DimensionHandle>({extra_dim}));
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Concatenate(in, extra_dim_shape, &out));
  TF_RETURN_IF_ERROR(c->ReplaceDim(out, -2, c->MakeDim(n_frames), &out));
  TF_RETURN_IF_ERROR(c->ReplaceDim(out, -1, c->MakeDim(frame_size), &out));
  c->set_output(0, out);
  c->set_output(1, c->Scalar());
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalFramer")
    .Attr("frame_size: int >= 1")
    .Attr("frame_step: int >= 1")
    .Attr("prefill: bool")
    .Input("input: int16")
    .Output("output: int16")
    .Output("output_valid: bool")
    .SetShapeFn([](InferenceContext* c) { return FramerShape(c); })
    .Doc(R"doc(
Transform an input signal into a series of overlapping frames, each of
size frame_size. The frame_step determines how many samples the framer
progresses on each invocation. When the framer has enough samples to produce
a frame, it writes it to the output tensor and sets the output_valid to True.
If the framer doesn't have enough samples to produce a frame, it doesn't
initialize the contents of the output tensor and sets the output_valid boolean
to False.
For example, assuming an input signal of 10 samples,
[1 2 3 4 5 6 7 8 9 0]
and the framer is invoked 5 times, each time with two samples.
For a frame with the configuration:
frame_size=3
frame_step=2
The framer will produce the following output:
input: [1, 2], output: [undefined, undefined, undefined], output_valid: False
input: [3, 4], output: [1, 2, 3], output_valid: True
input: [5, 6], output: [3, 4, 5], output_valid: True
input: [7, 8], output: [5, 6, 7], output_valid: True
input: [9, 0], output: [7, 6, 8], output_valid: True

input: A 1-D input signal frame. Size must be a multiple of frame_step
output: A 2-D output frame with innermost dimension frame_size
output_valid: A boolean scalar.
If True, the output is a valid output frame
If False, the output is an invalid output frame of all zeros

frame_size: The number of samples in each output frame
frame_step: The number of input samples to progress the framer's 'current'
sample on each invocation.
prefill: If true, initialize the framer with (frame_size - frame_step) zeros.
Can be used to guarantee a valid output starting with the first input.
)doc");

}  // namespace signal
}  // namespace tensorflow
