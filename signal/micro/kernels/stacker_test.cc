/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/micro/kernels/stacker_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

constexpr int kInputsSize = 1;
constexpr int kOutputsSize = 2;
constexpr int kTensorsSize = kInputsSize + kOutputsSize;

class StackerKernelRunner {
 public:
  StackerKernelRunner(int* input_dims_data, const int16_t* input_data,
                      int* output_dims_data, int16_t* output_data,
                      int* output_ready_dims_data, bool* ouput_ready_data)
      : tensors_{testing::CreateTensor(
                     input_data,
                     tflite::testing::IntArrayFromInts(input_dims_data)),
                 testing::CreateTensor(
                     output_data,
                     tflite::testing::IntArrayFromInts(output_dims_data)),
                 testing::CreateTensor(
                     ouput_ready_data,
                     testing::IntArrayFromInts(output_ready_dims_data))},
        inputs_array_{testing::IntArrayFromInts(inputs_array_data_)},
        outputs_array_{testing::IntArrayFromInts(outputs_array_data_)},
        kernel_runner_{*registration_, tensors_,       kTensorsSize,
                       inputs_array_,  outputs_array_, nullptr} {}

  micro::KernelRunner* kernel_runner() { return &kernel_runner_; }

 private:
  int inputs_array_data_[2] = {1, 0};
  int outputs_array_data_[3] = {2, 1, 2};
  TfLiteTensor tensors_[kTensorsSize] = {};
  TfLiteIntArray* inputs_array_ = nullptr;
  TfLiteIntArray* outputs_array_ = nullptr;
  TFLMRegistration* registration_ = tflm_signal::Register_STACKER();
  micro::KernelRunner kernel_runner_;
};

void TestStackerInvoke(int* output_dims_data, int16_t* output_data,
                       bool* ouput_ready_data, const int16_t* golden,
                       micro::KernelRunner* kernel_runner) {
  TfLiteIntArray* output_dims = testing::IntArrayFromInts(output_dims_data);

  const int output_len = ElementCount(*output_dims);

  TF_LITE_MICRO_EXPECT_EQ(kernel_runner->Invoke(), kTfLiteOk);
  TF_LITE_MICRO_EXPECT_EQ(*ouput_ready_data, 1);

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }
}

void TestStacker(int* input_dims_data, const int16_t* input_data,
                 int* output_dims_data, int16_t* output_data,
                 int* output_ready_dims_data, bool* ouput_ready_data,
                 const int16_t* golden, const unsigned char* flexbuffers_data,
                 const unsigned int flexbuffers_data_size) {
  StackerKernelRunner stacker_runner(input_dims_data, input_data,
                                     output_dims_data, output_data,
                                     output_ready_dims_data, ouput_ready_data);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(stacker_runner.kernel_runner()->InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestStackerInvoke(output_dims_data, output_data, ouput_ready_data, golden,
                    stacker_runner.kernel_runner());
}

// TestStackerReset() runs a test with the given inputs twice with a reset with
// the main purpose of testing the Stacker's Reset functionality. If you just
// want to make sure Stacker's Op output matches a set of golden values for an
// input use  TestStacker() instead.
void TestStackerReset(int* input_dims_data, const int16_t* input_data,
                      int* output_dims_data, int16_t* output_data,
                      int* output_ready_dims_data, bool* ouput_ready_data,
                      const int16_t* golden,
                      const unsigned char* flexbuffers_data,
                      const unsigned int flexbuffers_data_size) {
  StackerKernelRunner stacker_runner(input_dims_data, input_data,
                                     output_dims_data, output_data,
                                     output_ready_dims_data, ouput_ready_data);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(stacker_runner.kernel_runner()->InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestStackerInvoke(output_dims_data, output_data, ouput_ready_data, golden,
                    stacker_runner.kernel_runner());
  stacker_runner.kernel_runner()->Reset();
  TestStackerInvoke(output_dims_data, output_data, ouput_ready_data, golden,
                    stacker_runner.kernel_runner());
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(StackerTest3ChannelStep1) {
  int input_shape[] = {1, 3};
  int output_shape[] = {1, 6};
  int output_ready_shape[] = {0};
  const int16_t input[] = {0x1234, 0x5678, 0x4321};
  const int16_t golden[] = {0x1234, 0x5678, 0x4321, 0x1234, 0x5678, 0x4321};

  int16_t output[6];
  bool output_ready = false;

  tflite::TestStacker(input_shape, input, output_shape, output,
                      output_ready_shape, &output_ready, golden,
                      g_gen_data_stacker_3_channels_step_1,
                      g_gen_data_size_stacker_3_channels_step_1);
}

TF_LITE_MICRO_TEST(StackerTest10ChannelStep2_1stTest) {
  int input_shape[] = {1, 10};
  int output_shape[] = {1, 20};
  int output_ready_shape[] = {0};

  int16_t output[20];
  bool output_ready = false;

  const int16_t input[10] = {252, 477,  1071, 166,  1022,
                             312, 1171, 1586, 1491, 145};

  const int16_t golden[] = {252,  477,  1071, 166,  1022, 312,  1171,
                            1586, 1491, 145,  252,  477,  1071, 166,
                            1022, 312,  1171, 1586, 1491, 145};
  tflite::TestStacker(input_shape, input, output_shape, output,
                      output_ready_shape, &output_ready, golden,
                      g_gen_data_stacker_10_channels_step_2,
                      g_gen_data_size_stacker_10_channels_step_2);
}

TF_LITE_MICRO_TEST(StackerTest10ChannelStep2_2ndTest) {
  int input_shape[] = {1, 10};
  int output_shape[] = {1, 20};
  int output_ready_shape[] = {0};

  int16_t output[20];
  bool output_ready = false;

  const int16_t input[10] = {1060, 200, 69,  1519, 883,
                             1317, 182, 724, 143,  334};

  const int16_t golden[] = {1060, 200, 69, 1519, 883, 1317, 182, 724, 143, 334,
                            1060, 200, 69, 1519, 883, 1317, 182, 724, 143, 334};

  tflite::TestStacker(input_shape, input, output_shape, output,
                      output_ready_shape, &output_ready, golden,
                      g_gen_data_stacker_10_channels_step_2,
                      g_gen_data_size_stacker_10_channels_step_2);
}

TF_LITE_MICRO_TEST(StackerTestReset3ChannelStep1) {
  int input_shape[] = {1, 3};
  int output_shape[] = {1, 6};
  int output_ready_shape[] = {0};
  const int16_t input[] = {0x1234, 0x5678, 0x4321};
  const int16_t golden[] = {0x1234, 0x5678, 0x4321, 0x1234, 0x5678, 0x4321};

  int16_t output[6];
  bool output_ready = false;

  tflite::TestStackerReset(input_shape, input, output_shape, output,
                           output_ready_shape, &output_ready, golden,
                           g_gen_data_stacker_3_channels_step_1,
                           g_gen_data_size_stacker_3_channels_step_1);
}

TF_LITE_MICRO_TEST(StackerTestReset10ChannelStep2_1stTest) {
  int input_shape[] = {1, 10};
  int output_shape[] = {1, 20};
  int output_ready_shape[] = {0};

  int16_t output[20];
  bool output_ready = false;

  const int16_t input[10] = {252, 477,  1071, 166,  1022,
                             312, 1171, 1586, 1491, 145};

  const int16_t golden[] = {252,  477,  1071, 166,  1022, 312,  1171,
                            1586, 1491, 145,  252,  477,  1071, 166,
                            1022, 312,  1171, 1586, 1491, 145};
  tflite::TestStackerReset(input_shape, input, output_shape, output,
                           output_ready_shape, &output_ready, golden,
                           g_gen_data_stacker_10_channels_step_2,
                           g_gen_data_size_stacker_10_channels_step_2);
}

TF_LITE_MICRO_TEST(StackerTestReset10ChannelStep2_2ndTest) {
  int input_shape[] = {1, 10};
  int output_shape[] = {1, 20};
  int output_ready_shape[] = {0};

  int16_t output[20];
  bool output_ready = false;

  const int16_t input[10] = {1060, 200, 69,  1519, 883,
                             1317, 182, 724, 143,  334};

  const int16_t golden[] = {1060, 200, 69, 1519, 883, 1317, 182, 724, 143, 334,
                            1060, 200, 69, 1519, 883, 1317, 182, 724, 143, 334};

  tflite::TestStackerReset(input_shape, input, output_shape, output,
                           output_ready_shape, &output_ready, golden,
                           g_gen_data_stacker_10_channels_step_2,
                           g_gen_data_size_stacker_10_channels_step_2);
}

TF_LITE_MICRO_TESTS_END
