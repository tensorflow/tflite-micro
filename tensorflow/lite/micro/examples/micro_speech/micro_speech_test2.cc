#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>  // Required for atoi
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

// Helper function to generate a single feature slice.
TfLiteStatus GenerateSingleFeature(const int16_t* audio_data,
                                   const int audio_data_size,
                                   int8_t* feature_output,
                                   tflite::MicroInterpreter* interpreter) {
  TfLiteTensor* input = interpreter->input(0);
  std::copy_n(audio_data, audio_data_size,
              tflite::GetTensorData<int16_t>(input));
  if (interpreter->Invoke() != kTfLiteOk) {
    return kTfLiteError;
  }
  TfLiteTensor* output = interpreter->output(0);
  std::copy_n(tflite::GetTensorData<int8_t>(output), kFeatureSize,
              feature_output);
  return kTfLiteOk;
}

// Generates the full feature data from a single audio clip.
TfLiteStatus GenerateFeatures(const int16_t* audio_data,
                              const size_t audio_data_size,
                              Features* features_output) {
  const tflite::Model* model =
      tflite::GetModel(g_audio_preprocessor_int8_model_data);
  AudioPreprocessorOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  size_t remaining_samples = audio_data_size;
  size_t feature_index = 0;
  while (remaining_samples >= kAudioSampleDurationCount &&
         feature_index < kFeatureCount) {
    TF_LITE_ENSURE_STATUS(
        GenerateSingleFeature(audio_data, kAudioSampleDurationCount,
                              (*features_output)[feature_index], &interpreter));
    feature_index++;
    audio_data += kAudioSampleStrideCount;
    remaining_samples -= kAudioSampleStrideCount;
  }
  return kTfLiteOk;
}

}  // namespace

int main(int argc, char** argv) {
  // ====================================================================
  // 1. PARSE COMMAND-LINE ARGUMENTS
  // ====================================================================
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

  // ====================================================================
  // 2. PERFORM ONE-TIME SETUP
  // This is the "startup cost" that the delta measurement will cancel out.
  // ====================================================================
  printf("Performing one-time setup...\n");

  // Generate a single, representative feature set from an audio file.
  // The "yes" audio file is a good choice for a typical input.
  if (GenerateFeatures(g_yes_1000ms_audio_data, g_yes_1000ms_audio_data_size,
                       &g_features) != kTfLiteOk) {
    printf("ERROR: Feature generation failed.\n");
    return 1;
  }

  // Set up the MicroSpeech interpreter.
  const tflite::Model* model =
      tflite::GetModel(g_micro_speech_quantized_model_data);
  MicroSpeechOpResolver op_resolver;
  if (RegisterOps(op_resolver) != kTfLiteOk) {
    printf("ERROR: Failed to register ops.\n");
    return 1;
  }

  tflite::MicroInterpreter interpreter(model, op_resolver, g_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed.\n");
    return 1;
  }

  // Get the input tensor and copy the feature data into it.
  TfLiteTensor* input = interpreter.input(0);
  std::copy_n(&g_features[0][0], kFeatureElementCount,
              tflite::GetTensorData<int8_t>(input));

  printf("Setup complete.\n");

  // ====================================================================
  printf("Running %d invocations...\n", num_invocations);

  for (int i = 0; i < num_invocations; ++i) {
    if (interpreter.Invoke() != kTfLiteOk) {
      return 1;
    }
  }

  printf("Finished all invocations successfully.\n");

  return 0;
}