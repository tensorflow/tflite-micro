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

absl::Status DelayShape(InferenceContext* c) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalDelay")
    .Attr("delay_length: int >= 0")
    .Input("input: int16")
    .Output("output: int16")
    .SetShapeFn(DelayShape)
    .Doc(R"doc(
Delay the innermost dimension of input signal by delay_length samples.

For example, assuming an input signal of 10 samples,
[1 2 3 4 5 6 7 8 9 0]
If we input the signal to a delay op configured with delay_length=3, the op
will produce the following output:
[0 0 0 1 2 3 4 5 6 7]
To retrieve the remainder of the input signal, call the delay op again with
zeros as input:
[0 0 0 0 0 0 0 0 0 0]
to get the output:
[8 9 0 0 0 0 0 0 0 0]

input: A multidimensional input signal.
output: An output signal of the same shape as the input signal. The innermost
        dimension is delayed by delay_length samples.
)doc");

}  // namespace signal
}  // namespace tensorflow
