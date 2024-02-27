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

absl::Status EnergyShape(InferenceContext* c) {
  ShapeHandle out;

  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
  int64_t length = InferenceContext::Value(c->Dim(out, 0)) / 2;

  TF_RETURN_IF_ERROR(c->ReplaceDim(out, 0, c->MakeDim(length), &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalEnergy")
    .Attr("start_index: int")
    .Attr("end_index: int")
    .Input("input: int16")
    .Output("output: uint32")
    .SetShapeFn([](InferenceContext* c) { return EnergyShape(c); })
    .Doc(R"doc(
Calculate the energy of a spectral array. Only elements in the index range
[start_index, end_index] are calculated, and the rest are set to zero.

input: A 1-D frequency domain signal of complex int16 elements
output: A 1-D array of uint32 elements containing the square of the absolute
        value of each element in input.

start_index: index in input to start calculating from. Default: 0
end_index: last index in the input to calculate. Default: last element of input
)doc");

}  // namespace signal
}  // namespace tensorflow
