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

#include "signal/micro/kernels/framer_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

constexpr int kFrameSizeIndex = 0;  // 'frame_size'
constexpr int kFrameStepIndex = 1;  // 'frame_step'
constexpr int kPrefillIndex = 2;    // 'prefill'
constexpr int kInputsSize = 1;
constexpr int kOutputsSize = 2;
constexpr int kTensorsSize = kInputsSize + kOutputsSize;

class FramerKernelRunner {
 public:
  FramerKernelRunner(int* input_dims_data, int16_t* input_data,
                     int* output_dims_data, int16_t* output_data,
                     int* output_ready_dims_data, bool* output_ready)
      : inputs_array_{testing::IntArrayFromInts(inputs_array_data_)},
        outputs_array_{testing::IntArrayFromInts(outputs_array_data_)} {
    tensors_[0] = testing::CreateTensor(
        input_data, testing::IntArrayFromInts(input_dims_data));

    tensors_[1] = testing::CreateTensor(
        output_data, testing::IntArrayFromInts(output_dims_data));

    tensors_[2] = testing::CreateTensor(
        output_ready, testing::IntArrayFromInts(output_ready_dims_data));

    // go/tflm-static-cleanups for reasoning new is being used like this
    kernel_runner_ = new (kernel_runner_buffer) micro::KernelRunner(
        *registration_, tensors_, kTensorsSize, inputs_array_, outputs_array_,
        /*builtin_data=*/nullptr);
  }

  micro::KernelRunner& kernel_runner() { return *kernel_runner_; }

 private:
  uint8_t kernel_runner_buffer[sizeof(micro::KernelRunner)];
  int inputs_array_data_[kInputsSize + 1] = {kInputsSize, 0};
  int outputs_array_data_[kOutputsSize + 1] = {kOutputsSize, 1, 2};
  TfLiteTensor tensors_[kTensorsSize] = {};
  TfLiteIntArray* inputs_array_ = nullptr;
  TfLiteIntArray* outputs_array_ = nullptr;
  TFLMRegistration* registration_ = tflm_signal::Register_FRAMER();
  micro::KernelRunner* kernel_runner_ = nullptr;
};

alignas(alignof(FramerKernelRunner)) uint8_t
    framer_kernel_runner_buffer[sizeof(FramerKernelRunner)];

void TestFramerInvoke(int* input_dims_data, int16_t* input_data,
                      int* output_dims_data, const int16_t* golden,
                      int golden_len, int* output_ready_dims_data,
                      const unsigned char* flexbuffers_data,
                      const unsigned int flexbuffers_data_size,
                      int16_t* output_data, bool* output_ready,
                      micro::KernelRunner* runner) {
  FlexbufferWrapper fbw(flexbuffers_data, flexbuffers_data_size);
  int frame_size = fbw.ElementAsInt32(kFrameSizeIndex);
  int frame_step = fbw.ElementAsInt32(kFrameStepIndex);
  bool prefill = fbw.ElementAsBool(kPrefillIndex);
  int latency_samples = frame_size - frame_step;
  int input_size = input_dims_data[input_dims_data[0]];
  int outer_dims = 1;
  for (int i = 1; i < input_dims_data[0]; i++) {
    outer_dims *= input_dims_data[i];
  }
  int n_frames = output_dims_data[output_dims_data[0] - 1];
  TF_LITE_MICRO_EXPECT_EQ(frame_size, output_dims_data[output_dims_data[0]]);
  for (int i = 0; i < golden_len - latency_samples; i += input_size) {
    for (int outer_dim = 0; outer_dim < outer_dims; outer_dim++) {
      memcpy(&input_data[outer_dim * input_size], &golden[latency_samples + i],
             input_size * sizeof(int16_t));
    }
    TF_LITE_MICRO_EXPECT_EQ(runner->Invoke(), kTfLiteOk);
    TF_LITE_MICRO_EXPECT_EQ(*output_ready, (i >= latency_samples) || prefill);
    if (*output_ready == true) {
      for (int outer_dim = 0; outer_dim < outer_dims; outer_dim++) {
        for (int frame = 0; frame < n_frames; frame++) {
          int output_idx =
              outer_dim * frame_size * n_frames + frame * frame_size;
          int golden_idx = i + frame * frame_step;
          TF_LITE_MICRO_EXPECT_EQ(
              0, memcmp(&golden[golden_idx], &output_data[output_idx],
                        frame_size * sizeof(int16_t)));
        }
      }
    }
  }
}

void TestFramer(int* input_dims_data, int16_t* input_data,
                int* output_dims_data, const int16_t* golden, int golden_len,
                int* output_ready_dims_data,
                const unsigned char* flexbuffers_data,
                const unsigned int flexbuffers_data_size,
                int16_t* output_data) {
  bool output_ready = false;
  FramerKernelRunner* framer_runner = new (framer_kernel_runner_buffer)
      FramerKernelRunner(input_dims_data, input_data, output_dims_data,
                         output_data, output_ready_dims_data, &output_ready);
  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(framer_runner->kernel_runner().InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestFramerInvoke(input_dims_data, input_data, output_dims_data, golden,
                   golden_len, output_ready_dims_data, flexbuffers_data,
                   flexbuffers_data_size, output_data, &output_ready,
                   &framer_runner->kernel_runner());
}

void TestFramerReset(int* input_dims_data, int16_t* input_data,
                     int* output_dims_data, const int16_t* golden,
                     int golden_len, int* output_ready_dims_data,
                     const unsigned char* flexbuffers_data,
                     const unsigned int flexbuffers_data_size,
                     int16_t* output_data) {
  bool output_ready = false;
  FramerKernelRunner* framer_runner = new (framer_kernel_runner_buffer)
      FramerKernelRunner(input_dims_data, input_data, output_dims_data,
                         output_data, output_ready_dims_data, &output_ready);
  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(framer_runner->kernel_runner().InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestFramerInvoke(input_dims_data, input_data, output_dims_data, golden,
                   golden_len, output_ready_dims_data, flexbuffers_data,
                   flexbuffers_data_size, output_data, &output_ready,
                   &framer_runner->kernel_runner());
  framer_runner->kernel_runner().Reset();
  TestFramerInvoke(input_dims_data, input_data, output_dims_data, golden,
                   golden_len, output_ready_dims_data, flexbuffers_data,
                   flexbuffers_data_size, output_data, &output_ready,
                   &framer_runner->kernel_runner());
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FramerTest_3_1_0) {
  const int kInputSize = 1;
  const int kOutputSize = 3;
  int input_dims_data[] = {1, kInputSize};
  int output_dims_data[] = {2, 1, kOutputSize};
  int output_ready_dims_data[] = {0};
  const int16_t golden[] = {0x0, 0x0, 0x1234, 0x5678, 0x4321, 0x7777};
  int16_t input_data;
  int16_t output_data[kOutputSize];

  tflite::TestFramer(input_dims_data, &input_data, output_dims_data, golden,
                     sizeof(golden) / sizeof(int16_t), output_ready_dims_data,
                     g_gen_data_3_1_0_framer, g_gen_data_size_3_1_0_framer,
                     output_data);
}

TF_LITE_MICRO_TEST(FramerTest_5_2_1) {
  const int kInputSize = 2;
  const int kOutputSize = 5;
  int input_dims_data[] = {1, kInputSize};
  int output_dims_data[] = {2, 1, kOutputSize};
  int output_ready_dims_data[] = {0};
  const int16_t golden[] = {0x0, 0x0, 0x0, 0x1010, 0x0202, 0x7070, 0x0606};

  int16_t input_data[kInputSize];
  int16_t output_data[kOutputSize];

  tflite::TestFramer(input_dims_data, input_data, output_dims_data, golden,
                     sizeof(golden) / sizeof(int16_t), output_ready_dims_data,
                     g_gen_data_5_2_1_framer, g_gen_data_size_5_2_1_framer,
                     output_data);
}

TF_LITE_MICRO_TEST(FramerTest_5_2_1_NFrames2) {
  const int kInputSize = 4;
  const int kOutputSize = 5;
  const int kNFrames = 2;
  int input_dims_data[] = {1, kInputSize};
  int output_dims_data[] = {2, kNFrames, kOutputSize};
  int output_ready_dims_data[] = {0};
  const int16_t golden[] = {0x0, 0x0, 0x0, 0x1010, 0x0202, 0x7070, 0x0606};

  int16_t input_data[kInputSize];
  int16_t output_data[kNFrames * kOutputSize];

  tflite::TestFramer(input_dims_data, input_data, output_dims_data, golden,
                     sizeof(golden) / sizeof(int16_t), output_ready_dims_data,
                     g_gen_data_5_2_1_framer, g_gen_data_size_5_2_1_framer,
                     output_data);
}

TF_LITE_MICRO_TEST(FramerTest_5_2_1_NFrames2OuterDims4) {
  const int kInputSize = 4;
  const int kOutputSize = 5;
  int input_dims_data[] = {3, 2, 2, kInputSize};
  int output_dims_data[] = {4, 2, 2, 2, kOutputSize};
  int output_ready_dims_data[] = {0};
  const int16_t golden[] = {0x0, 0x0, 0x0, 0x1010, 0x0202, 0x7070, 0x0606};

  int16_t input_data[2 * 2 * kInputSize];
  int16_t output_data[2 * 2 * 2 * kOutputSize];

  tflite::TestFramer(input_dims_data, input_data, output_dims_data, golden,
                     sizeof(golden) / sizeof(int16_t), output_ready_dims_data,
                     g_gen_data_5_2_1_framer, g_gen_data_size_5_2_1_framer,
                     output_data);
}

TF_LITE_MICRO_TEST(TestReset) {
  const int kInputSize = 1;
  const int kOutputSize = 3;
  int input_dims_data[] = {1, kInputSize};
  int output_dims_data[] = {2, 1, kOutputSize};
  int output_ready_dims_data[] = {0};
  const int16_t golden[] = {0x0, 0x0, 0x1234, 0x5678, 0x4321, 0x7777};
  int16_t input_data;
  int16_t output_data[kOutputSize];
  tflite::TestFramerReset(input_dims_data, &input_data, output_dims_data,
                          golden, sizeof(golden) / sizeof(int16_t),
                          output_ready_dims_data, g_gen_data_3_1_0_framer,
                          g_gen_data_size_3_1_0_framer, output_data);
}

TF_LITE_MICRO_TESTS_END
