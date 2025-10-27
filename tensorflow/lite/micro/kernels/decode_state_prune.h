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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_PRUNE_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_PRUNE_H_

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/decode_state.h"

namespace tflite {

class DecodeStatePrune : public DecodeState {
 public:
  DecodeStatePrune() = delete;

  DecodeStatePrune(const TfLiteContext* context,
                   MicroProfilerInterface* profiler)
      : DecodeState(context, profiler) {}

  virtual TfLiteStatus Setup(const TfLiteTensor& input,
                             const TfLiteTensor& ancillary,
                             const TfLiteTensor& output) override;
  virtual TfLiteStatus Decode(const TfLiteEvalTensor& input,
                              const TfLiteEvalTensor& ancillary,
                              const TfLiteEvalTensor& output) override;

 private:
  // Prune Decode Common Metadata constants
  static constexpr size_t kDcmVersionOffset = 4;

 protected:
  virtual ~DecodeStatePrune() = default;

  template <typename T>
  void DecompressToBuffer(void* buffer);

  void DecompressToBufferPerChannelInt8(void* buffer);

 protected:
  const uint8_t* compressed_indices_ = nullptr;
  size_t count_indices_ = 0;
  size_t num_channels_ = 1;
  size_t elements_per_channel_ = 0;    // computed from use_alternate_axis_
  const void* value_table_ = nullptr;  // original non-pruned values
  int8_t* zero_points_ = nullptr;      // quantized per-channel zero points
  int8_t single_zero_point_ = 0;       // single channel zero point
  bool use_alternate_axis_ = false;    // shape channel axis:
                                       // false = first, true = last

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECODE_STATE_PRUNE_H_
