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

absl::Status FilterBankShape(InferenceContext* c) {
  ShapeHandle out;
  shape_inference::DimensionHandle unused;
  int num_channels;
  TF_RETURN_IF_ERROR(c->GetAttr<int>("num_channels", &num_channels));

  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));

  // Inputs 1,2 must have the same shape
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &out));
  auto num_weights = InferenceContext::Value(c->Dim(out, 0));

  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &out));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(c->input(2), 0), num_weights, &unused));
  // Inputs 3,4,5 must have the same shape
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &out));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(c->input(3), 0), num_channels + 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &out));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(c->input(4), 0), num_channels + 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &out));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(c->input(5), 0), num_channels + 1, &unused));
  TF_RETURN_IF_ERROR(c->ReplaceDim(out, 0, c->MakeDim(num_channels), &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalFilterBank")
    .Attr("num_channels: int >= 0")
    .Input("input: uint32")
    .Input("weights: int16")
    .Input("unweights: int16")
    .Input("channel_frequency_starts: int16")
    .Input("channel_weight_starts: int16")
    .Input("channel_widths: int16")
    .Output("output: uint64")
    .SetShapeFn([](InferenceContext* c) { return FilterBankShape(c); })
    .Doc(R"doc(
Applies a mel filter bank of size num_channels to an input spectral energy array
See filter_bank_ops.py for how weights, unweights, channel_frequency_start,
channel_weight_start, channel_widths are pre-calculated.

input: A 1-D spectral energy array.
weights: A 1-D filter weight array of num_channels + 1 elements.
unweights: A 1-D filter unweights array of num_channels + 1 elements.
channel_frequency_starts: A 1-D array of size num_channels + 1 elements.
                          Start index in input for each channel.
channel_weight_starts:  A 1-D array of size num_channels + 1 elements.
                        Start index in (un)weights array for each channel.
channel_widths: A 1-D array of num_channels + 1 elements.
                Number of bins for each channels.
output: A 1-D array of num_channels elements. Each elements
        contains the output of a single channel/filter in the bank.

num_channels: Number of channels in filter bank
)doc");

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalFilterBankSquareRoot")
    .Input("input: uint64")
    .Input("scale_bits: int32")
    .Output("output: uint32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    })
    .Doc(R"doc(
Applies a square root to each element in the input then shift right by
scale_bits before writing the result to output

input: A 1-D array of filter bank channels.
scale_bits: A scaler. Number of bits to shift right
output: A 1-D array of num_channels elements.
)doc");

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalFilterBankSpectralSubtraction")
    .Attr("num_channels: int >= 0")
    .Attr("smoothing: int >= 0")
    .Attr("one_minus_smoothing: int >= 0")
    .Attr("alternate_smoothing: int >= 0")
    .Attr("alternate_one_minus_smoothing: int >= 0")
    .Attr("smoothing_bits: int >= 0")
    .Attr("min_signal_remaining: int >= 0")
    .Attr("clamping: bool")
    .Attr("spectral_subtraction_bits: int")
    .Input("input: uint32")
    .Output("output: uint32")
    .Output("noise_estimate: uint32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
      c->set_output(0, out);
      c->set_output(1, out);
      return absl::OkStatus();
    })
    .Doc(R"doc(
Applies spectral subtraction to a filter bank output of size num_channels
Outputs the clean filter bank channels and the noise estimate for each channel.

input: A 1-D array of filter bank channels
output: A 1-D array of clean filter bank channels
noise_estimate: A 1-D array per-channel noise estimate

num_channels: Number of filter bank channels in input, output, noise_estimate
smoothing: Smoothing constant for noise LPF
one_minus_smoothing: (1 - smoothing) for noise LPF
min_signal_remaining: minimum amount of signal after subtraction
alternate_smoothing: if positive, noise LPF for odd-index channels, else ignored
alternate_one_minus_smoothing: (1 - alernate_smoothing), if in use
smoothing_bits: extra fractional bits for the noise_estimate smoothing filter
spectral_subtraction_bits: scaling bits for smoothing and min_signal_remaining
)doc");

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalFilterBankLog")
    .Attr("output_scale: int >= 1")
    .Attr("input_correction_bits: int >= 0")
    .Input("input: uint32")
    .Output("output: int16")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
      c->set_output(0, out);
      return absl::OkStatus();
    })
    .Doc(R"doc(
Applies natural log to each element in input with pre-shift and post scaling.
The operation is roughly equivalent to:
output = min(Log(input << input_correction_bits) * output_scale, int16max)
         If (input << input_correction_bits) is 1 or 0, the function returns 0

input: A 1-D array of filter bank channels.
output: A 1-D array of filter bank channels.

output_scale: A scaler.
input_correction_bits: A scalar
)doc");

}  // namespace signal
}  // namespace tensorflow
