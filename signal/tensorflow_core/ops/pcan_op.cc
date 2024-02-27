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

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {
namespace signal {

namespace {

absl::Status PcanShape(InferenceContext* c) {
  ShapeHandle out, lut;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &out));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &lut));

  c->set_output(0, out);
  return absl::OkStatus();
}

}  // namespace

REGISTER_OP("SignalPCAN")
    .Attr("snr_shift: int")
    .Input("input: uint32")
    .Input("noise_estimate: uint32")
    .Input("gain_lut: int16")
    .Output("output: uint32")
    .SetShapeFn(PcanShape)
    .Doc(R"doc(
Determines whether per-channel amplitude-normalized (PCAN) auto gain control is
applied, using either floating-point or fixed-point computation. If enabled,
the dynamic range of the filterbank output is compressed by dividing by a power
of the noise estimate.

input: A 1-D array of mel-spectrum subband filter bank outputs.
noise_estimate: A 1-D array of mel-spectrun subbabd noise estimates.
gain_lut: A 1-D lookup table for gain calculation.
output: A 1-D array of processed subband filter bank.
snr_shift: Amount of right shift when calculcating the SNR.
)doc");

}  // namespace signal
}  // namespace tensorflow
