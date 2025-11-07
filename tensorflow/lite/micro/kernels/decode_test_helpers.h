/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_TEST_HELPERS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_TEST_HELPERS_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

struct TensorInDatum {
  const void* const data;
  const TfLiteIntArray& dims;
};

struct TensorOutDatum {
  void* const data;
  const TfLiteIntArray& dims;
  const TfLiteType type;
  const TfLiteFloatArray& scales;
  const TfLiteIntArray& zero_points;
  const int quantized_dimension;

  // initialized by CreatePerChannelQuantizedTensor
  const TfLiteAffineQuantization affine_quantization;
};

template <typename T, size_t N>
struct AncillaryData {
  AncillaryData() = delete;
  AncillaryData(const uint8_t (&dcm)[tflite::DecodeState::kDcmSizeInBytes],
                const T (&values)[N]) {
    std::copy(std::begin(dcm), std::end(dcm), std::begin(dcm_));
    std::copy(std::begin(values), std::end(values), std::begin(value_table_));
  }

 private:
  uint8_t dcm_[tflite::DecodeState::kDcmSizeInBytes];
  T value_table_[N > 0 ? N : 1];  // assure not zero length
};

template <typename T>
TfLiteStatus CheckOutput(const TfLiteTensor& output,
                         const void* const expected) {
  const T* const expected_data = reinterpret_cast<const T*>(expected);
  const T* const output_data = tflite::GetTensorData<T>(&output);

  constexpr float kTolerance = 1e-5;
  const size_t kOutputCount = tflite::NumElements(&output);
  for (size_t i = 0; i < kOutputCount; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
    TF_LITE_MICRO_CHECK_FAIL();
  }

  return kTfLiteOk;
}

template <size_t kNumInputs, size_t kNumOutputs>
TfLiteStatus ExecuteDecodeTest(
    TfLiteTensor* tensors, const TFLMRegistration& registration,
    const std::initializer_list<const void*>& expected) {
  int kInputArrayData[kNumInputs + 1] = {kNumInputs};
  for (size_t i = 0; i < kNumInputs; i++) {
    kInputArrayData[i + 1] = i;
  }
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);

  int kOutputArrayData[kNumOutputs + 1] = {kNumOutputs};
  for (size_t i = 0; i < kNumOutputs; i++) {
    kOutputArrayData[i + 1] = i + kNumInputs;
  }
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  micro::KernelRunner runner(registration, tensors, kNumInputs + kNumOutputs,
                             inputs_array, outputs_array, nullptr);

  if (runner.InitAndPrepare() != kTfLiteOk || runner.Invoke() != kTfLiteOk) {
    return kTfLiteError;
  }

  const TfLiteTensor* const output_tensors = &tensors[kNumInputs];
  TfLiteStatus status = kTfLiteError;
  for (size_t i = 0; i < kNumOutputs; i++) {
    switch (output_tensors[i].type) {
      case kTfLiteBool:
        status = CheckOutput<bool>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteInt8:
        status = CheckOutput<int8_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteInt16:
        status = CheckOutput<int16_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteFloat32:
        status = CheckOutput<float>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteInt32:
        status = CheckOutput<int32_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteInt64:
        status = CheckOutput<int64_t>(output_tensors[i], expected.begin()[i]);
        break;
      default:
        TF_LITE_MICRO_FAIL("unsupported tensor type in test");
        break;
    }
  }

  return status;
}

template <size_t kNumInputs, size_t kNumOutputs>
void TestDecode(const std::initializer_list<const TensorInDatum*>& encodes,
                const std::initializer_list<const TensorInDatum*>& ancillaries,
                const std::initializer_list<const TensorOutDatum*>& outputs,
                const std::initializer_list<const void*>& expected,
                const TFLMRegistration& registration,
                const TfLiteStatus expected_status = kTfLiteOk) {
  TfLiteTensor tensors[kNumInputs + kNumOutputs] = {};

  for (size_t i = 0; i < kNumInputs; i += 2) {
    const TensorInDatum& tid_encode = *encodes.begin()[i / 2];
    tensors[i] = CreateTensor(tid_encode.data,
                              const_cast<TfLiteIntArray*>(&tid_encode.dims),
                              false, kTfLiteUInt8);
    // must be a const tensor
    tensors[i].allocation_type = kTfLiteMmapRo;
    const TensorInDatum& tid_ancillary = *ancillaries.begin()[i / 2];
    tensors[i + 1] = CreateTensor(
        tid_ancillary.data, const_cast<TfLiteIntArray*>(&tid_ancillary.dims),
        false, kTfLiteUInt8);
    // must be a const tensor
    tensors[i + 1].allocation_type = kTfLiteMmapRo;
  }
  for (size_t i = 0; i < kNumOutputs; i++) {
    const TensorOutDatum& tod = *outputs.begin()[i];
    if (tod.scales.size == 0) {
      tensors[i + kNumInputs] = CreateTensor(
          tod.data, const_cast<TfLiteIntArray*>(&tod.dims), false, tod.type);
    } else {
      tensors[i + kNumInputs] = CreatePerChannelQuantizedTensor(
          tod.data, const_cast<TfLiteIntArray*>(&tod.dims),
          const_cast<TfLiteFloatArray*>(&tod.scales),
          const_cast<TfLiteIntArray*>(&tod.zero_points),
          const_cast<TfLiteAffineQuantization*>(&tod.affine_quantization),
          tod.quantized_dimension, false, tod.type);
    }
  }

  TfLiteStatus s = ExecuteDecodeTest<kNumInputs, kNumOutputs>(
      tensors, registration, expected);
  TF_LITE_MICRO_EXPECT_EQ(s, expected_status);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_TEST_HELPERS_H_
