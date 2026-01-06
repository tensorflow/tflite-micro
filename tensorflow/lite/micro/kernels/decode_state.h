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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"

namespace tflite {

class DecodeState {
 public:
  DecodeState() = delete;

  DecodeState(const TfLiteContext* context, MicroProfilerInterface* profiler)
      : context_(context), micro_profiler_(profiler) {}

  virtual TfLiteStatus Setup(const TfLiteTensor& input,
                             const TfLiteTensor& ancillary,
                             const TfLiteTensor& output) = 0;
  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) = 0;

  static DecodeState* CreateDecodeStateLUT(const TfLiteContext* context,
                                           MicroProfilerInterface* profiler);
  static DecodeState* CreateDecodeStatePrune(const TfLiteContext* context,
                                             MicroProfilerInterface* profiler);
  static DecodeState* CreateDecodeStateHuffman(
      const TfLiteContext* context, MicroProfilerInterface* profiler);

  static uint8_t Type(const TfLiteTensor& ancillary) {
    return GetTensorData<uint8_t>(&ancillary)[kDcmDecodeTypeOffset];
  }

  static uint8_t Type(const TfLiteEvalTensor& ancillary) {
    return micro::GetTensorData<uint8_t>(&ancillary)[kDcmDecodeTypeOffset];
  }

  static uint8_t Version(const TfLiteTensor& ancillary) {
    return GetTensorData<uint8_t>(&ancillary)[kDcmVersionOffset];
  }

  static uint8_t Version(const TfLiteEvalTensor& ancillary) {
    return micro::GetTensorData<uint8_t>(&ancillary)[kDcmVersionOffset];
  }

 protected:
  virtual ~DecodeState() = default;

  // Decode Common Metadata constants
 public:
  static constexpr uint8_t kDcmTypeLUT = 0;
  static constexpr uint8_t kDcmTypeHuffman = 1;
  static constexpr uint8_t kDcmTypePrune = 2;
  static constexpr uint8_t kDcmTypeCustom = 127;

  static constexpr size_t kDcmSizeInBytes = 16;

 private:
  static constexpr size_t kDcmDecodeTypeOffset = 0;
  static constexpr size_t kDcmVersionOffset = 1;

  // DecodeState vars
 protected:
  const TfLiteContext* context_;
  MicroProfilerInterface* micro_profiler_;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_H_
