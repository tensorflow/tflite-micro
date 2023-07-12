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
#include <cstring>

#include "signal/micro/kernels/delay_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

constexpr int kInputsSize = 1;
constexpr int kOutputsSize = 1;
constexpr int kTensorsSize = kInputsSize + kOutputsSize;

class DelayKernelRunner {
 public:
  DelayKernelRunner(int* input_dims_data, int16_t* input_data,
                    int* output_dims_data, int16_t* output_data)
      : tensors_{testing::CreateTensor(
                     input_data, testing::IntArrayFromInts(input_dims_data)),
                 testing::CreateTensor(
                     output_data, testing::IntArrayFromInts(output_dims_data))},
        inputs_array_{testing::IntArrayFromInts(inputs_array_data_)},
        outputs_array_{testing::IntArrayFromInts(outputs_array_data_)},
        kernel_runner_{*registration_, tensors_,       kTensorsSize,
                       inputs_array_,  outputs_array_, nullptr} {}

  micro::KernelRunner& kernel_runner() { return kernel_runner_; }

 private:
  int inputs_array_data_[kInputsSize + 1] = {kInputsSize, 0};
  int outputs_array_data_[kOutputsSize + 1] = {kOutputsSize, 1};
  TfLiteTensor tensors_[kTensorsSize] = {};
  TfLiteIntArray* inputs_array_ = nullptr;
  TfLiteIntArray* outputs_array_ = nullptr;

  TFLMRegistration* registration_ = tflm_signal::Register_DELAY();
  micro::KernelRunner kernel_runner_;
};

void TestDelayInvoke(const int16_t* input_data, int16_t* output_data,
                     const int16_t* golden, int input_size, int input_num,
                     micro::KernelRunner* runner, int16_t* input_buffer) {
  for (int i = 0; i < input_num; i++) {
    memcpy(input_buffer, &input_data[i * input_size],
           sizeof(input_data[0]) * input_size);
    TF_LITE_MICRO_EXPECT_EQ(runner->Invoke(), kTfLiteOk);
    for (int j = 0; j < input_size; ++j) {
      TF_LITE_MICRO_EXPECT_EQ(golden[i * input_size + j], output_data[j]);
    }
  }
}

void TestDelay(int* input_dims_data, const int16_t* input_data,
               int* output_dims_data, int16_t* output_data,
               const int16_t* golden, int input_size, int input_num,
               const unsigned char* flexbuffers_data,
               const unsigned int flexbuffers_data_size,
               int16_t* input_buffer) {
  DelayKernelRunner delay_runner(input_dims_data, input_buffer,
                                 output_dims_data, output_data);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(delay_runner.kernel_runner().InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestDelayInvoke(input_data, output_data, golden, input_size, input_num,
                  &delay_runner.kernel_runner(), input_buffer);
}
// TestDelayReset() runs a test with the given inputs twice with a reset with
// the main purpose of testing the Delay's Reset functionality. If you just
// want to make sure Delay's Op output matches a set of golden values for an
// input use  TestDelay() instead.
void TestDelayReset(int* input_dims_data, const int16_t* input_data,
                    int* output_dims_data, int16_t* output_data,
                    const int16_t* golden, int input_size, int input_num,
                    const unsigned char* flexbuffers_data,
                    const unsigned int flexbuffers_data_size,
                    int16_t* input_buffer) {
  DelayKernelRunner delay_runner(input_dims_data, input_buffer,
                                 output_dims_data, output_data);

  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(delay_runner.kernel_runner().InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestDelayInvoke(input_data, output_data, golden, input_size, input_num,
                  &delay_runner.kernel_runner(), input_buffer);
  delay_runner.kernel_runner().Reset();
  TestDelayInvoke(input_data, output_data, golden, input_size, input_num,
                  &delay_runner.kernel_runner(), input_buffer);
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DelayTestSingleDimDelayLessThanFrameSize) {
  const int kInputSize = 8;
  const int kInputNum = 2;
  int input_shape[] = {1, kInputSize};
  int output_shape[] = {1, kInputSize};
  // The buffer that gets passed to the model.
  int16_t input_buffer[kInputSize];
  // The input data. Gets copied to input_buffer kInputNum times.
  const int16_t input[kInputNum * kInputSize] = {
      0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
      0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
  };
  int16_t output[kInputNum * kInputSize] = {0};
  const int16_t golden[kInputNum * kInputSize] = {0x0, 0x0, 0x0, 0x1, 0x2, 0x3,
                                                  0x4, 0x5, 0x6, 0x7, 0x8, 0x0,
                                                  0x0, 0x0, 0x0, 0x0};
  tflite::TestDelay(input_shape, input, output_shape, output, golden,
                    kInputSize, kInputNum, g_gen_data_3_delay,
                    g_gen_data_size_3_delay, input_buffer);
}

TF_LITE_MICRO_TEST(DelayTestSingleDimDelayGreaterThanFrameSize) {
  const int kInputSize = 3;
  const int kInputNum = 3;
  int input_shape[] = {1, kInputSize};
  int output_shape[] = {1, kInputSize};
  // The buffer that gets passed to the model.
  int16_t input_buffer[kInputSize];
  // The input data. Gets copied to input_buffer kInputNum times.
  const int16_t input[kInputNum * kInputSize] = {
      0x1, 0x2, 0x3, 0x4, 0x0, 0x0, 0x0, 0x0, 0x0,
  };
  int16_t output[kInputNum * kInputSize] = {0};
  const int16_t golden[kInputNum * kInputSize] = {
      0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x2, 0x3, 0x4,
  };
  tflite::TestDelay(input_shape, input, output_shape, output, golden,
                    kInputSize, kInputNum, g_gen_data_5_delay,
                    g_gen_data_size_5_delay, input_buffer);
}

TF_LITE_MICRO_TEST(DelayTestMultiDimDelayLessThanFrameSize) {
  const int kInputSize = 16;
  const int kInputNum = 2;
  int input_shape[] = {2, 4, 4};
  int output_shape[] = {2, 4, 4};
  // The buffer that gets passed to the model.
  int16_t input_buffer[kInputSize];
  // The op will be invoked 2 times (Input X, X=0,1)
  // For each invocation, the input's shape is (4, 4) but flattened for clarity
  // On each invocation, the input data is copied to input_buffer first.
  const int16_t input[kInputNum * kInputSize] = {
      0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB,
      0xC, 0xD, 0xE, 0xF, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
      0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
  };
  int16_t output[kInputNum * kInputSize] = {0};
  // For each invocation, we expect the following output (Output X, X=0,1)
  // Each time, the output's shape is (4, 4) but flattened for clarity
  const int16_t golden[kInputNum * kInputSize] = {
      // Output 0
      0x0,
      0x0,
      0x0,
      0x1,
      0x0,
      0x0,
      0x0,
      0x5,
      0x0,
      0x0,
      0x0,
      0x9,
      0x0,
      0x0,
      0x0,
      0xD,
      // Output 1
      0x2,
      0x3,
      0x4,
      0x0,
      0x6,
      0x7,
      0x8,
      0x0,
      0xA,
      0xB,
      0xC,
      0x0,
      0xE,
      0xF,
      0x0,
      0x0,
  };
  tflite::TestDelay(input_shape, input, output_shape, output, golden,
                    kInputSize, kInputNum, g_gen_data_3_delay,
                    g_gen_data_size_3_delay, input_buffer);
}

TF_LITE_MICRO_TEST(DelayTestMultiDimDelayGreaterThanFrameSize) {
  const int kInputSize = 16;
  const int kInputNum = 3;
  int input_shape[] = {2, 4, 4};
  int output_shape[] = {2, 4, 4};
  // The buffer that gets passed to the model.
  int16_t input_buffer[kInputSize];
  // The op will be invoked 3 times (Input X, X=0,1,2)
  // For each invocation, the input's shape is (4, 4) but flattened for clarity
  // On each invocation, the input data is copied to input_buffer first.
  const int16_t input[kInputNum * kInputSize] = {
      0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC,
      0xD, 0xE, 0xF, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
      0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
      0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
  };
  int16_t output[kInputNum * kInputSize] = {0};
  // For each invocation, we expect the following output (Output X, X=0,1,2)
  // Each time, the output's shape is (4, 4) but flattened for clarity
  const int16_t golden[kInputNum * kInputSize] = {
      // Output 0
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      // Output 1
      0x0,
      0x1,
      0x2,
      0x3,
      0x0,
      0x5,
      0x6,
      0x7,
      0x0,
      0x9,
      0xA,
      0xB,
      0x0,
      0xD,
      0xE,
      0xF,
      // Output 2
      0x4,
      0x0,
      0x0,
      0x0,
      0x8,
      0x0,
      0x0,
      0x0,
      0xC,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
      0x0,
  };
  tflite::TestDelay(input_shape, input, output_shape, output, golden,
                    kInputSize, kInputNum, g_gen_data_5_delay,
                    g_gen_data_size_5_delay, input_buffer);
}

TF_LITE_MICRO_TEST(DelayTestResetSingleDimDelayLessThanFrameSize) {
  const int kInputSize = 8;
  const int kInputNum = 2;
  int input_shape[] = {1, kInputSize};
  int output_shape[] = {1, kInputSize};
  // The buffer that gets passed to the model.
  int16_t input_buffer[kInputSize];
  // The input data. Gets copied to input_buffer kInputNum times.
  const int16_t input[kInputNum * kInputSize] = {
      0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
      0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
  };
  int16_t output[kInputNum * kInputSize] = {0};
  const int16_t golden[kInputNum * kInputSize] = {0x0, 0x0, 0x0, 0x1, 0x2, 0x3,
                                                  0x4, 0x5, 0x6, 0x7, 0x8, 0x0,
                                                  0x0, 0x0, 0x0, 0x0};
  tflite::TestDelayReset(input_shape, input, output_shape, output, golden,
                         kInputSize, kInputNum, g_gen_data_3_delay,
                         g_gen_data_size_3_delay, input_buffer);
}

TF_LITE_MICRO_TEST(DelayTestResetSingleResetDimDelayGreaterThanFrameSize) {
  const int kInputSize = 3;
  const int kInputNum = 3;
  int input_shape[] = {1, kInputSize};
  int output_shape[] = {1, kInputSize};
  // The buffer that gets passed to the model.
  int16_t input_buffer[kInputSize];
  // The input data. Gets copied to input_buffer kInputNum times.
  const int16_t input[kInputNum * kInputSize] = {
      0x1, 0x2, 0x3, 0x4, 0x0, 0x0, 0x0, 0x0, 0x0,
  };
  int16_t output[kInputNum * kInputSize] = {0};
  const int16_t golden[kInputNum * kInputSize] = {
      0x0, 0x0, 0x0, 0x0, 0x0, 0x1, 0x2, 0x3, 0x4,
  };
  tflite::TestDelayReset(input_shape, input, output_shape, output, golden,
                         kInputSize, kInputNum, g_gen_data_5_delay,
                         g_gen_data_size_5_delay, input_buffer);
}

TF_LITE_MICRO_TESTS_END
