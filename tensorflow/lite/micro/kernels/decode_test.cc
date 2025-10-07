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

#include <algorithm>
#include <array>
#include <initializer_list>
#include <type_traits>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/decode_state_huffman.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
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

//
// LUT test data
//
constexpr int kBitWidthLUT = 2;

constexpr int8_t kAncillaryDataLUT0[] = {1, 2, 3, 4};
constexpr int16_t kAncillaryDataLUT1[] = {5, 6, 7, 8};

constexpr uint8_t kDcmLUT0[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeLUT,  // type: LUT
    1,                                 // DCM version: 1
    0,                                 // reserved
    0,                                 // reserved
    1,                                 // LUT version: 1
    kBitWidthLUT,                      // Parameters: bit-width 2
    std::size(kAncillaryDataLUT0),     // channel stride
};

constexpr uint8_t kDcmLUT1[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeLUT,  // type: LUT
    1,                                 // DCM version: 1
    0,                                 // reserved
    0,                                 // reserved
    1,                                 // LUT version: 1
    kBitWidthLUT,                      // Parameters: bit-width 2
    std::size(kAncillaryDataLUT1),     // channel stride
};

// Align the tensor data the same as a Buffer in the TfLite schema
alignas(16) const uint8_t kEncodedLUT[] = {0x1B, 0xE4};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapeLUT[] = {3, 1, 2, 4};
constexpr int kEncodedShapeLUT[] = {1, sizeof(kEncodedLUT)};

constexpr int8_t kExpectLUT0[] = {1, 2, 3, 4, 4, 3, 2, 1};
constexpr int16_t kExpectLUT1[] = {5, 6, 7, 8, 8, 7, 6, 5};

//
// Prune test data
//
constexpr int8_t kAncillaryDataPrune0[] = {
    1, 2, 3, 4,  // 0
    1, 2, 3, 4,  // 1
    1, 2, 3, 4,  // 2
    1, 2, 3, 4,  // 3
    1, 2, 3, 4   // 4
};
constexpr int16_t kAncillaryDataPrune1[] = {
    5, 6, 7, 8,  // 0
    5, 6, 7, 8,  // 1
    5, 6, 7, 8,  // 2
    5, 6, 7, 8,  // 3
    5, 6, 7, 8   // 4
};
constexpr float kAncillaryDataPrune2[] = {
    9.0f, 10.0f, 11.0f, 12.0f,  // 0
    9.0f, 10.0f, 11.0f, 12.0f,  // 1
    9.0f, 10.0f, 11.0f, 12.0f,  // 2
    9.0f, 10.0f, 11.0f, 12.0f,  // 3
    9.0f, 10.0f, 11.0f, 12.0f   // 4
};
constexpr int8_t kAncillaryDataPrune3[] = {
    13, 14, 15, 16,  // 0
    13, 14, 15, 16,  // 1
    13, 14, 15, 16,  // 2
    13, 14, 15, 16,  // 3
    13, 14, 15, 16   // 4
};
constexpr int8_t kAncillaryDataPrune4[] = {
    17, 18, 19, 20,  // 0
    17, 18, 19, 20,  // 1
    17, 18, 19, 20,  // 2
    17, 18, 19, 20,  // 3
    17, 18, 19, 20   // 4
};

constexpr uint8_t kDcmPrune[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypePrune,  // type: Prune
    1,                                   // DCM version: 1
    0,                                   // reserved
    0,                                   // reserved
    1,                                   // Prune version: 1
};

// Align the tensor data the same as a Buffer in the TfLite schema
alignas(16) const uint8_t kEncodedPrune[] = {0xA5, 0xA5, 0xA5, 0xA5, 0xA5};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapePrune[] = {3, 2, 5, 4};
constexpr int kEncodedShapePrune[] = {1, sizeof(kEncodedPrune)};

// Quantization datum as TfLiteIntArray.
// Scales are modified by FloatArrayFromFloats. As globals they cannot be
// <const> without causing a processor exception.
float kScalesPrune0[] = {2, 1.0f, 1.0f};
constexpr int kZeroPointsPrune0[] = {2, -128, -64};
float kScalesPrune1[] = {4, 1.0f, 1.0f, 1.0f, 1.0f};
constexpr int kZeroPointsPrune1[] = {4, 0, 0, 0, 0};
float kScalesPrune4[] = {4, 1.0f, 1.0f, 1.0f, 1.0f};
constexpr int kZeroPointsPrune4[] = {4, -126, -62, -30, -14};

constexpr int8_t kExpectPrune0[] = {
    1,   -128, 2,    -128, -128, 3,   -128, 4,    1,   -128,  // 0
    2,   -128, -128, 3,    -128, 4,   1,    -128, 2,   -128,  // 0
    -64, 3,    -64,  4,    1,    -64, 2,    -64,  -64, 3,     // 1
    -64, 4,    1,    -64,  2,    -64, -64,  3,    -64, 4      // 1
};
constexpr int16_t kExpectPrune1[] = {
    5, 0, 6, 0,  // 0
    0, 7, 0, 8,  // 1
    5, 0, 6, 0,  // 2
    0, 7, 0, 8,  // 3
    5, 0, 6, 0,  // 4
    0, 7, 0, 8,  // 5
    5, 0, 6, 0,  // 6
    0, 7, 0, 8,  // 7
    5, 0, 6, 0,  // 8
    0, 7, 0, 8   // 9
};
constexpr float kExpectPrune2[] = {
    9.0f, 0.0f, 10.0f, 0.0f, 0.0f, 11.0f, 0.0f, 12.0f,  // 0
    9.0f, 0.0f, 10.0f, 0.0f, 0.0f, 11.0f, 0.0f, 12.0f,  // 1
    9.0f, 0.0f, 10.0f, 0.0f, 0.0f, 11.0f, 0.0f, 12.0f,  // 2
    9.0f, 0.0f, 10.0f, 0.0f, 0.0f, 11.0f, 0.0f, 12.0f,  // 3
    9.0f, 0.0f, 10.0f, 0.0f, 0.0f, 11.0f, 0.0f, 12.0f   // 4
};
constexpr int8_t kExpectPrune3[] = {
    13, 0, 14, 0, 0, 15, 0, 16,  // 0
    13, 0, 14, 0, 0, 15, 0, 16,  // 1
    13, 0, 14, 0, 0, 15, 0, 16,  // 2
    13, 0, 14, 0, 0, 15, 0, 16,  // 3
    13, 0, 14, 0, 0, 15, 0, 16   // 4
};
constexpr int8_t kExpectPrune4[] = {
    17,   -62, 18,  -14,  // 0
    -126, 19,  -30, 20,   // 1
    17,   -62, 18,  -14,  // 2
    -126, 19,  -30, 20,   // 3
    17,   -62, 18,  -14,  // 4
    -126, 19,  -30, 20,   // 5
    17,   -62, 18,  -14,  // 6
    -126, 19,  -30, 20,   // 7
    17,   -62, 18,  -14,  // 8
    -126, 19,  -30, 20    // 9
};

//
// Huffman test data
//
// Test data is based on an intial table size of 8 elements with the following
// codeword map:
//   'm' : '000',
//   'a' : '001',
//   'j' : '010',
//   'o' : '0110',
//   'h' : '0111',
//   'd' : '1000',
//   'i' : '10010',
//   'f' : '10011',
//   'c' : '1010',
//   'g' : '101100',
//   'l' : '101101',
//   'k' : '10111',
//   'e' : '1100',
//   'b' : '1101',
//   'p' : '1110',
//   'n' : '1111'
//

constexpr int kHuffmanInitial = 2;  // log2(8) - 1
constexpr int kHuffmanShift =
    tflite::DecodeStateHuffman::kDcmTableSizeInitialShift;
constexpr int kHuffman32 = tflite::DecodeStateHuffman::kDcmTableSize32BitsMask;

constexpr uint8_t kDcmHuffman16[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeHuffman,  // type: Huffman
    1,                                     // DCM version: 1
    0,                                     // reserved
    0,                                     // reserved
    1,                                     // Huffman version: 1
    kHuffmanInitial << kHuffmanShift,      // Table size: 8 16-bit elements
};

constexpr uint8_t kDcmHuffman32[tflite::DecodeState::kDcmSizeInBytes] = {
    tflite::DecodeState::kDcmTypeHuffman,  // type: Huffman
    1,                                     // DCM version: 1
    0,                                     // reserved
    0,                                     // reserved
    1,                                     // Huffman version: 1
    (kHuffmanInitial << kHuffmanShift) |
        kHuffman32,  // Table size: 8 32-bit elements
};

// Align the tensor data the same as a Buffer in the TfLite schema
alignas(16) const uint8_t kEncodedHuffman[] = {
    0xF0, 0x0D, 0x79, 0xFC, 0x9C, 0x1A, 0x6E, 0x4C, 0x32, 0xAF, 0x29,
    0xB3, 0x5D, 0xF6, 0x36, 0x02, 0x50, 0x15, 0x8C, 0xA7, 0x95, 0xDB,
    0x29, 0x68, 0x3F, 0xBA, 0xB7, 0xED, 0xB1, 0x19, 0xE2, 0xE0,
};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapeHuffman[] = {1, 64};
constexpr int kEncodedShapeHuffman[] = {1, sizeof(kEncodedHuffman)};

constexpr int8_t kExpectHuffmanInt8[] = {
    'n', 'm', 'm', 'a', 'c', 'n', 'a', 'n', 'e', 'f', 'd', 'a', 'c',
    'o', 'p', 'j', 'o', 'm', 'e', 'c', 'k', 'i', 'f', 'o', 'o', 'k',
    'h', 'b', 'd', 'b', 'd', 'm', 'j', 'j', 'd', 'm', 'j', 'g', 'o',
    'j', 'f', 'e', 'c', 'p', 'b', 'i', 'i', 'b', 'm', 'a', 'n', 'b',
    'b', 'j', 'b', 'n', 'l', 'g', 'j', 'a', 'f', 'e', 'j', 'p',
};
constexpr int16_t kExpectHuffmanInt16[] = {
    'n', 'm', 'm', 'a', 'c', 'n', 'a', 'n', 'e', 'f', 'd', 'a', 'c',
    'o', 'p', 'j', 'o', 'm', 'e', 'c', 'k', 'i', 'f', 'o', 'o', 'k',
    'h', 'b', 'd', 'b', 'd', 'm', 'j', 'j', 'd', 'm', 'j', 'g', 'o',
    'j', 'f', 'e', 'c', 'p', 'b', 'i', 'i', 'b', 'm', 'a', 'n', 'b',
    'b', 'j', 'b', 'n', 'l', 'g', 'j', 'a', 'f', 'e', 'j', 'p',
};

constexpr uint16_t kAncillaryDataHuffman16[] = {
    // Table 0:
    0x986D,  // [0]: size= 3 symbol=m
    0x9861,  // [1]: size= 3 symbol=a
    0x986A,  // [2]: size= 3 symbol=j
    0x0005,  // [3]: size= 0 offset=    5 (@8)
    0x0806,  // [4]: size= 1 offset=    6 (@10)
    0x0809,  // [5]: size= 1 offset=    9 (@14)
    0x000E,  // [6]: size= 0 offset=   14 (@20)
    0x000F,  // [7]: size= 0 offset=   15 (@22)
    // Table 1:
    0x886F,  // [8]: size= 1 symbol=o
    0x8868,  // [9]: size= 1 symbol=h
    // Table 2:
    0x8864,  // [10]: size= 1 symbol=d
    0x8864,  // [11]: size= 1 symbol=d
    0x9069,  // [12]: size= 2 symbol=i
    0x9066,  // [13]: size= 2 symbol=f
    // Table 3:
    0x8863,  // [14]: size= 1 symbol=c
    0x8863,  // [15]: size= 1 symbol=c
    0x0002,  // [16]: size= 0 offset=    2 (@18)
    0x906B,  // [17]: size= 2 symbol=k
    // Table 4:
    0x8867,  // [18]: size= 1 symbol=g
    0x886C,  // [19]: size= 1 symbol=l
    // Table 5:
    0x8865,  // [20]: size= 1 symbol=e
    0x8862,  // [21]: size= 1 symbol=b
    // Table 6:
    0x8870,  // [22]: size= 1 symbol=p
    0x886E,  // [23]: size= 1 symbol=n
};
constexpr uint32_t kAncillaryDataHuffman32[] = {
    // Table 0:
    0x9800006D,  // [0]: size= 3 symbol=m
    0x98000061,  // [1]: size= 3 symbol=a
    0x9800006A,  // [2]: size= 3 symbol=j
    0x00000005,  // [3]: size= 0 offset=    5 (@8)
    0x08000006,  // [4]: size= 1 offset=    6 (@10)
    0x08000009,  // [5]: size= 1 offset=    9 (@14)
    0x0000000E,  // [6]: size= 0 offset=   14 (@20)
    0x0000000F,  // [7]: size= 0 offset=   15 (@22)
    // Table 1:
    0x8800006F,  // [8]: size= 1 symbol=o
    0x88000068,  // [9]: size= 1 symbol=h
    // Table 2:
    0x88000064,  // [10]: size= 1 symbol=d
    0x88000064,  // [11]: size= 1 symbol=d
    0x90000069,  // [12]: size= 2 symbol=i
    0x90000066,  // [13]: size= 2 symbol=f
    // Table 3:
    0x88000063,  // [14]: size= 1 symbol=c
    0x88000063,  // [15]: size= 1 symbol=c
    0x00000002,  // [16]: size= 0 offset=    2 (@18)
    0x9000006B,  // [17]: size= 2 symbol=k
    // Table 4:
    0x88000067,  // [18]: size= 1 symbol=g
    0x8800006C,  // [19]: size= 1 symbol=l
    // Table 5:
    0x88000065,  // [20]: size= 1 symbol=e
    0x88000062,  // [21]: size= 1 symbol=b
    // Table 6:
    0x88000070,  // [22]: size= 1 symbol=p
    0x8800006E,  // [23]: size= 1 symbol=n
};

//
// Custom DECODE test data
//
constexpr int kDecodeTypeCustom = 200;

constexpr int8_t kAncillaryDataCustom[] = {0x42};

constexpr uint8_t kDcmCustom[tflite::DecodeState::kDcmSizeInBytes] = {
    kDecodeTypeCustom,  // type: custom
    1,                  // DCM version: 1
};

// Align the tensor data the same as a Buffer in the TfLite schema
alignas(16) const uint8_t kEncodedCustom[] = {0x42, 0x43, 0x40, 0x46,
                                              0x4A, 0x52, 0x62, 0x02};

// Tensor shapes as TfLiteIntArray
constexpr int kOutputShapeCustom[] = {1, 8};
constexpr int kEncodedShapeCustom[] = {1, sizeof(kEncodedCustom)};

constexpr int8_t kExpectCustom[] = {0x00, 0x01, 0x02, 0x04,
                                    0x08, 0x10, 0x20, 0x40};

struct DecodeStateCustom : public DecodeState {
  DecodeStateCustom() = delete;

  DecodeStateCustom(const TfLiteContext* context,
                    MicroProfilerInterface* profiler)
      : DecodeState(context, profiler) {}

  virtual TfLiteStatus Setup(const TfLiteTensor& input,
                             const TfLiteTensor& ancillary,
                             const TfLiteTensor& output) override {
    return kTfLiteOk;
  }
  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override {
    const uint8_t* inp = micro::GetTensorData<uint8_t>(&input);
    TF_LITE_ENSURE(const_cast<TfLiteContext*>(context_), inp != nullptr);
    uint8_t* outp =
        micro::GetTensorData<uint8_t>(const_cast<TfLiteEvalTensor*>(&output));
    TF_LITE_ENSURE(const_cast<TfLiteContext*>(context_), outp != nullptr);
    const uint8_t* vp = micro::GetTensorData<uint8_t>(&ancillary);
    TF_LITE_ENSURE(const_cast<TfLiteContext*>(context_), vp != nullptr);
    vp += kDcmSizeInBytes;

    std::transform(inp, inp + input.dims->data[0], outp,
                   [vp](uint8_t i) { return i ^ *vp; });

    return kTfLiteOk;
  }

  static DecodeState* CreateDecodeStateCustom(
      const TfLiteContext* context, MicroProfilerInterface* profiler) {
    alignas(4) static uint8_t buffer[sizeof(DecodeStateCustom)];
    DecodeState* instance = new (buffer) DecodeStateCustom(context, profiler);
    return instance;
  }

 protected:
  virtual ~DecodeStateCustom() = default;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
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
    const std::initializer_list<const void*>& expected,
    const std::initializer_list<MicroContext::AlternateMemoryRegion>* amr =
        nullptr,
    const std::initializer_list<tflite::MicroContext::CustomDecodeRegistration>*
        cdr = nullptr) {
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

  if (amr != nullptr) {
    runner.GetFakeMicroContext()->SetDecompressionMemory(*amr);
  }
  if (cdr != nullptr) {
    runner.GetFakeMicroContext()->SetCustomDecodeRegistrations(*cdr);
  }

  if (runner.InitAndPrepare() != kTfLiteOk || runner.Invoke() != kTfLiteOk) {
    return kTfLiteError;
  }

  const TfLiteTensor* const output_tensors = &tensors[kNumInputs];
  TfLiteStatus status = kTfLiteError;
  for (size_t i = 0; i < kNumOutputs; i++) {
    switch (output_tensors[i].type) {
      case kTfLiteInt8:
        status = CheckOutput<int8_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteInt16:
        status = CheckOutput<int16_t>(output_tensors[i], expected.begin()[i]);
        break;
      case kTfLiteFloat32:
        status = CheckOutput<float>(output_tensors[i], expected.begin()[i]);
        break;
      default:
        TF_LITE_MICRO_FAIL("unsupported tensor type in test");
        break;
    }
  }

  return status;
}

template <size_t kNumInputs, size_t kNumOutputs>
void TestDecode(
    const std::initializer_list<const TensorInDatum*>& encodes,
    const std::initializer_list<const TensorInDatum*>& ancillaries,
    const std::initializer_list<const TensorOutDatum*>& outputs,
    const std::initializer_list<const void*>& expected,
    const TFLMRegistration& registration,
    const std::initializer_list<MicroContext::AlternateMemoryRegion>* amr =
        nullptr,
    const std::initializer_list<tflite::MicroContext::CustomDecodeRegistration>*
        cdr = nullptr,
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
      tensors, registration, expected, amr, cdr);
  TF_LITE_MICRO_EXPECT_EQ(s, expected_status);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

using tflite::testing::AncillaryData;
using tflite::testing::kAncillaryDataCustom;
using tflite::testing::kAncillaryDataHuffman16;
using tflite::testing::kAncillaryDataHuffman32;
using tflite::testing::kAncillaryDataLUT0;
using tflite::testing::kAncillaryDataLUT1;
using tflite::testing::kAncillaryDataPrune0;
using tflite::testing::kAncillaryDataPrune1;
using tflite::testing::kAncillaryDataPrune2;
using tflite::testing::kAncillaryDataPrune3;
using tflite::testing::kAncillaryDataPrune4;
using tflite::testing::kDcmCustom;
using tflite::testing::kDcmHuffman16;
using tflite::testing::kDcmHuffman32;
using tflite::testing::kDcmLUT0;
using tflite::testing::kDcmLUT1;
using tflite::testing::kDcmPrune;
using tflite::testing::kEncodedCustom;
using tflite::testing::kEncodedHuffman;
using tflite::testing::kEncodedLUT;
using tflite::testing::kEncodedPrune;
using tflite::testing::kEncodedShapeCustom;
using tflite::testing::kEncodedShapeHuffman;
using tflite::testing::kEncodedShapeLUT;
using tflite::testing::kEncodedShapePrune;
using tflite::testing::kExpectCustom;
using tflite::testing::kExpectHuffmanInt16;
using tflite::testing::kExpectHuffmanInt8;
using tflite::testing::kExpectLUT0;
using tflite::testing::kExpectLUT1;
using tflite::testing::kExpectPrune0;
using tflite::testing::kExpectPrune1;
using tflite::testing::kExpectPrune2;
using tflite::testing::kExpectPrune3;
using tflite::testing::kExpectPrune4;
using tflite::testing::kOutputShapeCustom;
using tflite::testing::kOutputShapeHuffman;
using tflite::testing::kOutputShapeLUT;
using tflite::testing::kOutputShapePrune;
using tflite::testing::kScalesPrune0;
using tflite::testing::kScalesPrune1;
using tflite::testing::kScalesPrune4;
using tflite::testing::kZeroPointsPrune0;
using tflite::testing::kZeroPointsPrune1;
using tflite::testing::kZeroPointsPrune4;
using tflite::testing::TensorInDatum;
using tflite::testing::TensorOutDatum;

TF_LITE_MICRO_TEST(DecodeSingleTensor) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectLUT0)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataLUT0)>
      kAncillaryData = {{kDcmLUT0}, {kAncillaryDataLUT0}};

  constexpr int kAncillaryShapeLUT[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeLUT);
  static const TensorInDatum tid_encode = {
      kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeLUT);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectLUT0};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodeTwoTensors) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data0[std::size(kExpectLUT0)] = {};
  alignas(16) int16_t output_data1[std::size(kExpectLUT1)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataLUT0)>
      kAncillaryData0 = {{kDcmLUT0}, {kAncillaryDataLUT0}};
  alignas(16) const AncillaryData<int16_t, std::size(kAncillaryDataLUT1)>
      kAncillaryData1 = {{kDcmLUT1}, {kAncillaryDataLUT1}};

  constexpr int kAncillaryShapeLUT0[] = {1, sizeof(kAncillaryData0)};
  constexpr int kAncillaryShapeLUT1[] = {1, sizeof(kAncillaryData1)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeLUT);
  static const TensorInDatum tid_encode0 = {
      kEncodedLUT,
      *encoded_dims,
  };
  static const TensorInDatum tid_encode1 = {
      kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode0, &tid_encode1};

  const TfLiteIntArray* const ancillary_dims0 =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT0);
  static const TensorInDatum tid_ancillary0 = {
      &kAncillaryData0,
      *ancillary_dims0,
  };
  const TfLiteIntArray* const ancillary_dims1 =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT1);
  static const TensorInDatum tid_ancillary1 = {
      &kAncillaryData1,
      *ancillary_dims1,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary0, &tid_ancillary1};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeLUT);
  constexpr float output_scales_data[] = {1, 1.0f};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {1, 0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod0 = {
      output_data0,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static const TensorOutDatum tod1 = {
      output_data1,
      *output_dims,
      kTfLiteInt16,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod0, &tod1};

  const std::initializer_list<const void*> expected = {kExpectLUT0,
                                                       kExpectLUT1};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneFloat) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) float output_data[std::size(kExpectPrune2)] = {};
  alignas(16) const AncillaryData<float, std::size(kAncillaryDataPrune2)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune2}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  constexpr float kOutputScalesData[] = {0};
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kOutputScalesData);
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteFloat32,
      *kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune2};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune3)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune3)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune3}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  constexpr float kOutputScalesData[] = {0};
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kOutputScalesData);
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      *kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune3};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune0)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune0)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune0}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScalesPrune0);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune0);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      *kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune0};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedAltAxisInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectPrune4)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataPrune4)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune4}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScalesPrune4);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune4);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt8,
      *kOutputScales,
      *kOutputZeroPoints,
      (kOutputDims->size - 1),
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune4};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedAltAxisInt16) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int16_t output_data[std::size(kExpectPrune1)] = {};
  alignas(16) const AncillaryData<int16_t, std::size(kAncillaryDataPrune1)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune1}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScalesPrune1);
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPointsPrune1);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt16,
      *kOutputScales,
      *kOutputZeroPoints,
      (kOutputDims->size - 1),
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune1};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodePruneQuantizedInvalidZeroPointInt16) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int16_t output_data[std::size(kExpectPrune1)] = {};
  alignas(16) const AncillaryData<int16_t, std::size(kAncillaryDataPrune1)>
      kAncillaryData = {{kDcmPrune}, {kAncillaryDataPrune1}};

  const TfLiteIntArray* const kEncodedDims =
      tflite::testing::IntArrayFromInts(kEncodedShapePrune);
  static const TensorInDatum kEncodeTID = {
      kEncodedPrune,
      *kEncodedDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kEncodes = {
      &kEncodeTID,
  };

  constexpr int kAncillaryShape[] = {1, sizeof(kAncillaryData)};
  const TfLiteIntArray* const kAncillaryDims =
      tflite::testing::IntArrayFromInts(kAncillaryShape);
  static const TensorInDatum kAncillaryTID = {
      &kAncillaryData,
      *kAncillaryDims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> kAncillaries = {
      &kAncillaryTID};

  const TfLiteIntArray* const kOutputDims =
      tflite::testing::IntArrayFromInts(kOutputShapePrune);
  float kScales[] = {2, 1.0f, 1.0f};
  const TfLiteFloatArray* const kOutputScales =
      tflite::testing::FloatArrayFromFloats(kScales);
  const int kZeroPoints[] = {2, 0, -1};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kZeroPoints);
  static const TensorOutDatum kTOD = {
      output_data,
      *kOutputDims,
      kTfLiteInt16,
      *kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> kOutputs = {
      &kTOD};

  const std::initializer_list<const void*> kExpected = {kExpectPrune1};

  tflite::testing::TestDecode<kEncodes.size() + kAncillaries.size(),
                              kOutputs.size()>(
      kEncodes, kAncillaries, kOutputs, kExpected, tflite::Register_DECODE(),
      nullptr, nullptr, kTfLiteError);
}

TF_LITE_MICRO_TEST(DecodeHuffmanTable16BitsInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectHuffmanInt8)] = {};
  alignas(16) const AncillaryData<uint16_t, std::size(kAncillaryDataHuffman16)>
      kAncillaryData = {{kDcmHuffman16}, {kAncillaryDataHuffman16}};

  constexpr int kAncillaryShapeHuffman[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeHuffman);
  static const TensorInDatum tid_encode = {
      kEncodedHuffman,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeHuffman);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeHuffman);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectHuffmanInt8};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodeHuffmanTable16BitsInt16Fail) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int16_t output_data[std::size(kExpectHuffmanInt16)] = {};
  alignas(16) const AncillaryData<uint16_t, std::size(kAncillaryDataHuffman16)>
      kAncillaryData = {{kDcmHuffman16}, {kAncillaryDataHuffman16}};

  constexpr int kAncillaryShapeHuffman[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeHuffman);
  static const TensorInDatum tid_encode = {
      kEncodedHuffman,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeHuffman);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeHuffman);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt16,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectHuffmanInt16};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE(),
      nullptr, nullptr, kTfLiteError);
}

TF_LITE_MICRO_TEST(DecodeHuffmanTable32BitsInt8) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectHuffmanInt8)] = {};
  alignas(16) const AncillaryData<uint32_t, std::size(kAncillaryDataHuffman32)>
      kAncillaryData = {{kDcmHuffman32}, {kAncillaryDataHuffman32}};

  constexpr int kAncillaryShapeHuffman[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeHuffman);
  static const TensorInDatum tid_encode = {
      kEncodedHuffman,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeHuffman);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeHuffman);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectHuffmanInt8};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodeHuffmanTable32BitsInt16) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int16_t output_data[std::size(kExpectHuffmanInt16)] = {};
  alignas(16) const AncillaryData<uint32_t, std::size(kAncillaryDataHuffman32)>
      kAncillaryData = {{kDcmHuffman32}, {kAncillaryDataHuffman32}};

  constexpr int kAncillaryShapeHuffman[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeHuffman);
  static const TensorInDatum tid_encode = {
      kEncodedHuffman,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeHuffman);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeHuffman);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt16,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectHuffmanInt16};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE());
}

TF_LITE_MICRO_TEST(DecodeWithAltDecompressionMemory) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectLUT0)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataLUT0)>
      kAncillaryData = {{kDcmLUT0}, {kAncillaryDataLUT0}};

  constexpr int kAncillaryShapeLUT[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeLUT);
  static const TensorInDatum tid_encode = {
      kEncodedLUT,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeLUT);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeLUT);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      nullptr,  // using alternate decompression memory
      *output_dims, kTfLiteInt8, *output_scales, *output_zero_points, 0, {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectLUT0};

  std::initializer_list<tflite::MicroContext::AlternateMemoryRegion> amr = {
      {output_data, sizeof(output_data)}};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE(), &amr);
}

TF_LITE_MICRO_TEST(DecodeWithCustomRegistration) {
  // Align the tensor data the same as a Buffer in the TfLite schema
  alignas(16) int8_t output_data[std::size(kExpectCustom)] = {};
  alignas(16) const AncillaryData<int8_t, std::size(kAncillaryDataCustom)>
      kAncillaryData = {{kDcmCustom}, {kAncillaryDataCustom}};

  constexpr int kAncillaryShapeCustom[] = {1, sizeof(kAncillaryData)};

  const TfLiteIntArray* const encoded_dims =
      tflite::testing::IntArrayFromInts(kEncodedShapeCustom);
  static const TensorInDatum tid_encode = {
      kEncodedCustom,
      *encoded_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> encodes = {
      &tid_encode,
  };

  const TfLiteIntArray* const ancillary_dims =
      tflite::testing::IntArrayFromInts(kAncillaryShapeCustom);
  static const TensorInDatum tid_ancillary = {
      &kAncillaryData,
      *ancillary_dims,
  };
  static constexpr std::initializer_list<const TensorInDatum*> ancillaries = {
      &tid_ancillary};

  const TfLiteIntArray* const output_dims =
      tflite::testing::IntArrayFromInts(kOutputShapeCustom);
  constexpr float output_scales_data[] = {0};
  const TfLiteFloatArray* const output_scales =
      tflite::testing::FloatArrayFromFloats(output_scales_data);
  constexpr int output_zero_points_data[] = {0};
  const TfLiteIntArray* const output_zero_points =
      tflite::testing::IntArrayFromInts(output_zero_points_data);
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt8,
      *output_scales,
      *output_zero_points,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectCustom};

  const std::initializer_list<tflite::MicroContext::CustomDecodeRegistration>
      cdr = {
          {
              tflite::testing::kDecodeTypeCustom,
              0,  // reserved
              0,  // reserved
              0,  // reserved
              tflite::testing::DecodeStateCustom::CreateDecodeStateCustom,
          },
      };

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE(),
      nullptr, &cdr);
}

TF_LITE_MICRO_TESTS_END
