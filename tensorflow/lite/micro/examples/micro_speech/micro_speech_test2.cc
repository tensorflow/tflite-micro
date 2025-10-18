/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_30ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/noise_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_30ms_audio_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kArenaSize = 28584;  // xtensa p6
alignas(16) uint8_t g_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

TfLiteStatus RegisterOps(AudioPreprocessorOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  return kTfLiteOk;
}

TfLiteStatus LoadMicroSpeechModelAndPerformInference(
    const Features& features, const char* expected_label) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);

  MicroSpeechOpResolver op_resolver;
  RegisterOps(op_resolver);

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);

  TfLiteTensor* output = interpreter.output(0);

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  for (int i = 0; i < 4; i++) {
    std::copy_n(&features[0][0], kFeatureElementCount,
                tflite::GetTensorData<int8_t>(input));
    interpreter.Invoke();
  }

  // Dequantize output values
  volatile float category_predictions[kCategoryCount];
  for (int i = 0; i < kCategoryCount; i++) {
    category_predictions[i] =
        (tflite::GetTensorData<int8_t>(output)[i] - output_zero_point) *
        output_scale;

    if (category_predictions[i] > -1000.0f) {
        // Dummy read to satisfy compiler
    }
  }

  return kTfLiteOk;
}

TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);

  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));
  interpreter->Invoke();
  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
              feature_output);

  return kTfLiteOk;
}

TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);

  AudioPreprocessorOpResolver op_resolver;
  RegisterOps(op_resolver);

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  interpreter.AllocateTensors();

  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
  GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index], &interpreter);
    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }

  return kTfLiteOk;
}

TfLiteStatus TestAudioSample(const char* label, const int16_t* audio_data,
                             const size_t audio_data_size) {
      GenerateFeatures(audio_data, audio_data_size, &g_features);
      LoadMicroSpeechModelAndPerformInference(g_features, label);
  return kTfLiteOk;
}

}  // namespace

int main () {
  TestAudioSample("no", g_no_1000ms_audio_data, g_no_1000ms_audio_data_size);

  TestAudioSample("yes", g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size);

  TestAudioSample("silence", g_silence_1000ms_audio_data,
                  g_silence_1000ms_audio_data_size);

  TestAudioSample("silence", g_noise_1000ms_audio_data,
                  g_noise_1000ms_audio_data_size);
}