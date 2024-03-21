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

absl::Status RfftShape(InferenceContext* c) {
  ShapeHandle out;
  int fft_length;
  TF_RETURN_IF_ERROR(c->GetAttr<int>("fft_length", &fft_length));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &out));
  auto dim = ((fft_length / 2) + 1) * 2;  // * 2 for complex
  TF_RETURN_IF_ERROR(c->ReplaceDim(out, -1, c->MakeDim(dim), &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

absl::Status IrfftShape(InferenceContext* c) {
  ShapeHandle out;
  int fft_length;
  TF_RETURN_IF_ERROR(c->GetAttr<int>("fft_length", &fft_length));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &out));
  TF_RETURN_IF_ERROR(c->ReplaceDim(out, -1, c->MakeDim(fft_length), &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalRfft")
    .Attr("T: {float, int16, int32}")
    .Attr("fft_length: int >= 2")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(RfftShape)
    .Doc(R"doc(
Computes the 1-dimensional discrete Fourier transform of a real-valued signal
over the inner-most dimension of input. Since the DFT of a real signal is
Hermitian-symmetric, RFFT only returns the fft_length / 2 + 1 unique complex
components of the FFT: the zero-frequency term, followed by the fft_length / 2
positive-frequency terms. Along the axis RFFT is computed on, if fft_length is
larger than the corresponding dimension of input, the dimension is padded with
zeros.

input: A Tensor. Must be one of the following types: float32, int16, int32
output: A tensor containing ((fft_length / 2) + 1) complex spectral
        components along its innermost dimension.
        Since there's no TF integer complex type, the array is represented using
        ((fft_length / 2) + 1) * 2 real elements.
        For integer input (int16, int32), the output is scaled by 1 / fft_length
        relative to the theoretical DFT, to avoid overflowing.
        For floating point (float32) input, the output isn't scaled.

fft_length: The length of the FFT operation. An input signal that's shorter
            will be zero padded to fft_length.
)doc");

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalIrfft")
    .Attr("T: {float, int16, int32}")
    .Attr("fft_length: int >= 2")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(IrfftShape)
    .Doc(R"doc(
Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
signal over the inner-most dimension of input.

The inner-most dimension of input is assumed to be the result of RFFT:
the fft_length / 2 + 1 unique components of the DFT of a real-valued signal.
fft_length must be provided.

input: A tensor containing ((fft_length / 2) + 1) complex spectral
       components along its innermost dimension.
       Since there's no TF integer complex type, the array is represented using
       ((fft_length / 2) + 1) * 2 real elements.
output: A tensor containing fft_length time domain elements along its innermost
        dimension.

fft_length: The length of the IFFT operation.
)doc");

// TODO(b/286250473): change back name after name clash resolved
REGISTER_OP("SignalFftAutoScale")
    .Input("input: int16")
    .Output("output: int16")
    .Output("scale_bits: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &out));
      c->set_output(0, out);
      c->set_output(1, c->Scalar());
      return absl::OkStatus();
    })
    .Doc(R"doc(
Shifts the input left until the amplitude is maximized without clipping. Returns
the amount of left shift for compensation later. This op can be used to maximize
precision of integer FFT implementations, especially 16-bit.

input: A 1-D time domain signal.
output: A 1-D time domain signal after auto scaling.
scale_bits: Scalar. The number of left shifts applied to the input signal.
)doc");

}  // namespace signal
}  // namespace tensorflow
