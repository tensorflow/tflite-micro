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

#include "tensorflow/lite/micro/kernels/decode_state_huffman.h"

#include <array>
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"
#include "tensorflow/lite/micro/kernels/decode_test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

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

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

using tflite::testing::AncillaryData;
using tflite::testing::TensorInDatum;
using tflite::testing::TensorOutDatum;

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
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum tod = {
      output_data, *output_dims, kTfLiteInt8, kOutputScales, *kOutputZeroPoints,
      0,           {},
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
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt16,
      kOutputScales,
      *kOutputZeroPoints,
      0,
      {},
  };
  static constexpr std::initializer_list<const TensorOutDatum*> outputs = {
      &tod};

  const std::initializer_list<const void*> expected = {kExpectHuffmanInt16};

  tflite::testing::TestDecode<encodes.size() + ancillaries.size(),
                              outputs.size()>(
      encodes, ancillaries, outputs, expected, tflite::Register_DECODE(),
      nullptr, kTfLiteError);
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
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum tod = {
      output_data, *output_dims, kTfLiteInt8, kOutputScales, *kOutputZeroPoints,
      0,           {},
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
  constexpr int kOutputZeroPointsData[] = {0};
  const TfLiteIntArray* const kOutputZeroPoints =
      tflite::testing::IntArrayFromInts(kOutputZeroPointsData);
  const TfLiteFloatArray kOutputScales = {kOutputZeroPoints->size};
  static const TensorOutDatum tod = {
      output_data,
      *output_dims,
      kTfLiteInt16,
      kOutputScales,
      *kOutputZeroPoints,
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

TF_LITE_MICRO_TESTS_END
