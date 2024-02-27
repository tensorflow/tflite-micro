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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {
namespace signal {

absl::Status WindowShape(InferenceContext* c) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &out));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &out));

  shape_inference::DimensionHandle dim_in;
  dim_in = c->Dim(c->input(0), -1);

  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(c->input(1), 0),
                                  InferenceContext::Value(dim_in), &dim_in));
  c->set_output(0, out);
  return absl::OkStatus();
}

// TODO(b/286250473): change back name to "Window" after name clash resolved
REGISTER_OP("SignalWindow")
    .Attr("shift: int")
    .Input("input: int16")
    .Input("weights: int16")
    .Output("output: int16")
    .SetShapeFn([](InferenceContext* c) { return WindowShape(c); })
    .Doc(R"doc(
Apply a window to an input signal with a right shift to each element

input: An N-D time domain input signal
weights: Constant 1-D window weights. Size must match innermost input dimension.
output: An N-D time domain output signal. Size must match input.

shift: An amount of right shifts to perform on each element before writing
to the output
)doc");

}  // namespace signal
}  // namespace tensorflow
