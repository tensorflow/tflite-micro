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

void RfftShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  int fft_length = 0;
  if (!GetAttrInt(ctx, "fft_length", &fft_length, status)) {
    return;
  }
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* ranked = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, in0, 1, ranked, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    int64_t dim =
        static_cast<int64_t>((fft_length / 2) + 1) * 2;  // * 2 for complex
    if (ReplaceLastDim(ctx, ranked, dim, out, status)) {
      TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
    }
  }
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(ranked);
  TF_DeleteShapeHandle(in0);
}

void IrfftShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  int fft_length = 0;
  if (!GetAttrInt(ctx, "fft_length", &fft_length, status)) {
    return;
  }
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* ranked = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, in0, 1, ranked, status);
  }
  if (TF_GetCode(status) == TF_OK &&
      ReplaceLastDim(ctx, ranked, fft_length, out, status)) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(ranked);
  TF_DeleteShapeHandle(in0);
}

void FftAutoScaleShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeHandle* scalar = TF_ShapeInferenceContextScalar(ctx);

  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in0, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 1, scalar, status);
  }

  TF_DeleteShapeHandle(scalar);
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(in0);
}

// Computes the 1-dimensional discrete Fourier transform of a real-valued signal
// over the inner-most dimension of input. Since the DFT of a real signal is
// Hermitian-symmetric, RFFT only returns the fft_length / 2 + 1 unique complex
// components of the FFT: the zero-frequency term, followed by the fft_length /
// 2 positive-frequency terms. Along the axis RFFT is computed on, if fft_length
// is larger than the corresponding dimension of input, the dimension is padded
// with zeros.
//
// input: A Tensor. Must be one of the following types: float32, int16, int32
// output: A tensor containing ((fft_length / 2) + 1) complex spectral
//         components along its innermost dimension.
//         Since there's no TF integer complex type, the array is represented
//         using
//         ((fft_length / 2) + 1) * 2 real elements.
//         For integer input (int16, int32), the output is scaled by 1 /
//         fft_length relative to the theoretical DFT, to avoid overflowing. For
//         floating point (float32) input, the output isn't scaled.
//
// fft_length: The length of the FFT operation. An input signal that's shorter
//             will be zero padded to fft_length.
//
// Computes the inverse 1-dimensional discrete Fourier transform of a
// real-valued signal over the inner-most dimension of input.
//
// The inner-most dimension of input is assumed to be the result of RFFT:
// the fft_length / 2 + 1 unique components of the DFT of a real-valued signal.
// fft_length must be provided.
//
// input: A tensor containing ((fft_length / 2) + 1) complex spectral
//        components along its innermost dimension.
//        Since there's no TF integer complex type, the array is represented
//        using
//        ((fft_length / 2) + 1) * 2 real elements.
// output: A tensor containing fft_length time domain elements along its
// innermost
//         dimension.
//
// fft_length: The length of the IFFT operation.
//
// Shifts the input left until the amplitude is maximized without clipping.
// Returns the amount of left shift for compensation later. This op can be used
// to maximize precision of integer FFT implementations, especially 16-bit.
//
// input: A 1-D time domain signal.
// output: A 1-D time domain signal after auto scaling.
// scale_bits: Scalar. The number of left shifts applied to the input signal.
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFftOps() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* rfft_builder =
      TF_NewOpDefinitionBuilder("SignalRfft");
  TF_OpDefinitionBuilderAddAttr(rfft_builder, "T: {float, int16, int32}");
  TF_OpDefinitionBuilderAddAttr(rfft_builder, "fft_length: int >= 2");
  TF_OpDefinitionBuilderAddInput(rfft_builder, "input: T");
  TF_OpDefinitionBuilderAddOutput(rfft_builder, "output: T");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(rfft_builder, &RfftShape);
  TF_RegisterOpDefinition(rfft_builder, status);

  TF_OpDefinitionBuilder* irfft_builder =
      TF_NewOpDefinitionBuilder("SignalIrfft");
  TF_OpDefinitionBuilderAddAttr(irfft_builder, "T: {float, int16, int32}");
  TF_OpDefinitionBuilderAddAttr(irfft_builder, "fft_length: int >= 2");
  TF_OpDefinitionBuilderAddInput(irfft_builder, "input: T");
  TF_OpDefinitionBuilderAddOutput(irfft_builder, "output: T");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(irfft_builder, &IrfftShape);
  TF_RegisterOpDefinition(irfft_builder, status);

  TF_OpDefinitionBuilder* auto_scale_builder =
      TF_NewOpDefinitionBuilder("SignalFftAutoScale");
  TF_OpDefinitionBuilderAddInput(auto_scale_builder, "input: int16");
  TF_OpDefinitionBuilderAddOutput(auto_scale_builder, "output: int16");
  TF_OpDefinitionBuilderAddOutput(auto_scale_builder, "scale_bits: int32");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(auto_scale_builder,
                                                  &FftAutoScaleShape);
  TF_RegisterOpDefinition(auto_scale_builder, status);

  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
