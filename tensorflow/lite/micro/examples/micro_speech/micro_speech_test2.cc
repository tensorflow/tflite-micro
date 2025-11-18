#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iterator>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/no_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/silence_1000ms_audio_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/testdata/yes_1000ms_audio_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

// Arena size is a guesstimate, followed by use of
// MicroInterpreter::arena_used_bytes() on both the AudioPreprocessor and
// MicroSpeech models and using the larger of the two results.
constexpr size_t kArenaSize = 28584;
alignas(16) uint8_t g_arena[kArenaSize];

using Features = int8_t[kFeatureCount][kFeatureSize];
Features g_features;

constexpr int kAudioSampleDurationCount =
    kFeatureDurationMs * kAudioSampleFrequency / 1000;
constexpr int kAudioSampleStrideCount =
    kFeatureStrideMs * kAudioSampleFrequency / 1000;

using MicroSpeechOpResolver = tflite::MicroMutableOpResolver<4>;
using AudioPreprocessorOpResolver = tflite::MicroMutableOpResolver<18>;

// Registers the ops used by the MicroSpeech model.
TfLiteStatus RegisterOps(MicroSpeechOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}

// Registers the ops used by the AudioPreprocessor model.
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

}  // namespace

int main(int argc, char** argv) {
  // Parse command-line argument
  if (argc != 2) {
    printf("ERROR: Incorrect usage.\n");
    printf("Usage: %s <num_invocations>\n", argv[0]);
    return 1;
  }

  int num_invocations = atoi(argv[1]);
  if (num_invocations <= 0) {
    printf("ERROR: Number of invocations must be greater than 0.\n");
    return 1;
  }

  // One-time setup for both models
  printf("Performing one-time setup for both models...\n");

  // Set up the AudioPreprocessor interpreter
  const tflite::Model* preprocessor_model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);
  AudioPreprocessorOpResolver preprocessor_op_resolver;
  if (RegisterOps(preprocessor_op_resolver) != kTfLiteOk) {
    printf("ERROR: Failed to register preprocessor ops.\n");
    return 1;
  }

  // Set up the MicroSpeech interpreter
  const tflite::Model* speech_model =
      tflite::GetModel(g_micro_speech_quantized_model_data);
  MicroSpeechOpResolver speech_op_resolver;
  if (RegisterOps(speech_op_resolver) != kTfLiteOk) {
    printf("ERROR: Failed to register speech ops.\n");
    return 1;
  }

  // Create BOTH interpreters first, sharing the same arena.
  tflite::MicroInterpreter preprocessor_interpreter(
      preprocessor_model, preprocessor_op_resolver, g_arena, kArenaSize);
  tflite::MicroInterpreter speech_interpreter(
      speech_model, speech_op_resolver, g_arena, kArenaSize);

  // Allocate tensors for the first model.
  if (preprocessor_interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: Preprocessor AllocateTensors() failed.\n");
    return 1;
  }
  // Now, the second interpreter will automatically allocate its memory *after*
  // the first one in the shared arena.
  if (speech_interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: Speech AllocateTensors() failed.\n");
    return 1;
  }

  // Get pointers to the input and output tensors of both models
  TfLiteTensor* preprocessor_input = preprocessor_interpreter.input(0); // <-- TYPO FIXED HERE
  TfLiteTensor* preprocessor_output = preprocessor_interpreter.output(0);
  TfLiteTensor* speech_input = speech_interpreter.input(0);

  printf("Setup complete.\n");

  printf("Running %d end-to-end invocations...\n", num_invocations);

  for (int i = 0; i < num_invocations; ++i) {
    // Generate Features
    const int16_t* audio_data = g_yes_1000ms_audio_data;
    size_t remaining_samples = g_yes_1000ms_audio_data_size;
    size_t feature_index = 0;

    while (remaining_samples >= kAudioSampleDurationCount &&
           feature_index < kFeatureCount)
    {
      std::copy_n(audio_data, kAudioSampleDurationCount,
                  tflite::GetTensorData<int16_t>(preprocessor_input));

      if (preprocessor_interpreter.Invoke() != kTfLiteOk) {
        printf("ERROR: Preprocessor Invoke() failed.\n");
        return 1;
      }

      std::copy_n(tflite::GetTensorData<int8_t>(preprocessor_output), kFeatureSize,
                  g_features[feature_index]);

      feature_index++;
      audio_data += kAudioSampleStrideCount;
      remaining_samples -= kAudioSampleStrideCount;
    }

    // Classify Features
    std::copy_n(&g_features[0][0], kFeatureElementCount,
                tflite::GetTensorData<int8_t>(speech_input));

    if (speech_interpreter.Invoke() != kTfLiteOk) {
      printf("ERROR: Speech Invoke() failed.\n");
      return 1;
    }
  }

  printf("Finished all invocations successfully.\n");

  return 0;
}