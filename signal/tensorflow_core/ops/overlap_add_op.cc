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

absl::Status OverlapAddShape(InferenceContext* c) {
  shape_inference::DimensionHandle unused;
  ShapeHandle in;
  ShapeHandle out;
  int frame_step;

  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &in));
  TF_RETURN_IF_ERROR(c->GetAttr<int>("frame_step", &frame_step));
  TF_RETURN_IF_ERROR(c->Subshape(in, 0, -1, &out));
  if (!c->ValueKnown(c->Dim(out, -1))) {
    TF_RETURN_IF_ERROR(c->ReplaceDim(out, -1, c->UnknownDim(), &out));
  } else {
    int n_frames = c->Value(c->Dim(out, -1));
    TF_RETURN_IF_ERROR(
        c->ReplaceDim(out, -1, c->MakeDim(n_frames * frame_step), &out));
  }
  c->set_output(0, out);
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalOverlapAdd")
    .Attr("T: {float, int16}")
    .Attr("frame_step: int >= 1")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) { return OverlapAddShape(c); })
    .Doc(R"doc(
Transform a an input signal made of overlaping frames of size frame_size into
an output signal made of frames of size frame_step.
The overalpping input frames are spaced frame_step apart in time.
The The op adds the overlapping frames into the output frame.
For example, for a series of 5 input frames, with frame_size=3, frame_step=2:
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]
[0, 1, 2]
[3, 4, 5]
The op will overlap the frames as follows:
[1, 2, 3]
      [4, 5, 6]
            [7, 8, 9]
                  [0, 1, 2]
                        [3, 4, 5]
Then add the samples that are aligned vertically to produce output frames of
size frame_step=2:
[1, 2]
[7, 5]
[13, 8]
[9, 1]
[5, 4]

input: A [..., frames, frame_length] Tensor. Rank must be at least 2.
output: A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.

frame_step: The number of output samples to progress the 'current'
sample on each invocation. Also the number of samples in each output frame.
)doc");

}  // namespace signal
}  // namespace tensorflow
