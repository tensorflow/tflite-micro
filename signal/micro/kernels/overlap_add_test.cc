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

#include "signal/micro/kernels/overlap_add_flexbuffers_generated_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {

constexpr int kFrameStepIndex = 1;
constexpr int kInputsSize = 1;
constexpr int kOutputsSize = 1;
constexpr int kTensorsSize = kInputsSize + kOutputsSize;

template <typename T>
class OverlapAddKernelRunner {
 public:
  OverlapAddKernelRunner(int* input_dims_data, T* input_data,
                         int* output_dims_data, T* output_data)
      : inputs_array_{testing::IntArrayFromInts(inputs_array_data_)},
        outputs_array_{testing::IntArrayFromInts(outputs_array_data_)} {
    tensors_[0] = testing::CreateTensor(
        input_data, testing::IntArrayFromInts(input_dims_data));

    tensors_[1] = tflite::testing::CreateTensor(
        output_data, testing::IntArrayFromInts(output_dims_data));

    registration_ = tflm_signal::Register_OVERLAP_ADD();

    // go/tflm-static-cleanups for reasoning new is being used like this
    kernel_runner_ = new (kernel_runner_buffer) tflite::micro::KernelRunner(
        *registration_, tensors_, kTensorsSize, inputs_array_, outputs_array_,
        /*builtin_data=*/nullptr);
  }

  micro::KernelRunner& GetKernelRunner() { return *kernel_runner_; }

 private:
  uint8_t kernel_runner_buffer[sizeof(micro::KernelRunner)];
  int inputs_array_data_[kInputsSize + 1] = {1, 0};
  int outputs_array_data_[kOutputsSize + 1] = {1, 1};
  TfLiteTensor tensors_[kTensorsSize] = {};
  TfLiteIntArray* inputs_array_ = nullptr;
  TfLiteIntArray* outputs_array_ = nullptr;
  TFLMRegistration* registration_ = nullptr;
  micro::KernelRunner* kernel_runner_ = nullptr;
};

// We can use any of the templated types here - int16_t was picked arbitrarily
alignas(alignof(OverlapAddKernelRunner<int16_t>)) uint8_t
    overlap_add_kernel_runner_buffer[sizeof(OverlapAddKernelRunner<int16_t>)];

template <typename T>
void TestOverlapAddInvoke(int* input_dims_data, T* input_data,
                          int* output_dims_data, const T* golden_input,
                          const T* golden_output, int iters,
                          const unsigned char* flexbuffers_data,
                          const unsigned int flexbuffers_data_size,
                          T* output_data, tflite::micro::KernelRunner* runner) {
  tflite::FlexbufferWrapper fbw(flexbuffers_data, flexbuffers_data_size);
  int frame_step = fbw.ElementAsInt32(kFrameStepIndex);
  int frame_size = input_dims_data[input_dims_data[0]];
  int n_frames = input_dims_data[input_dims_data[0] - 1];
  int outer_dims = 1;
  for (int i = 1; i < input_dims_data[0] - 1; i++) {
    outer_dims *= input_dims_data[i];
  }
  for (int i = 0; i < iters; i++) {
    for (int outer_dim = 0; outer_dim < outer_dims; outer_dim++) {
      int input_idx = outer_dim * n_frames * frame_size;
      int golden_input_idx = i * n_frames * frame_size;
      memcpy(&input_data[input_idx], &golden_input[golden_input_idx],
             n_frames * frame_size * sizeof(T));
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner->Invoke());
    for (int outer_dim = 0; outer_dim < outer_dims; outer_dim++) {
      int output_idx = outer_dim * n_frames * frame_step;
      int golden_output_idx = i * n_frames * frame_step;
      TF_LITE_MICRO_EXPECT_EQ(
          0, memcmp(&output_data[output_idx], &golden_output[golden_output_idx],
                    n_frames * frame_step * sizeof(T)));
    }
  }
}

template <typename T>
void TestOverlapAdd(int* input_dims_data, T* input_data, int* output_dims_data,
                    const T* golden_input, const T* golden_output, int iters,
                    const unsigned char* flexbuffers_data,
                    const unsigned int flexbuffers_data_size, T* output_data) {
  OverlapAddKernelRunner<T>* overlap_add_runner =
      new (overlap_add_kernel_runner_buffer) OverlapAddKernelRunner<T>(
          input_dims_data, input_data, output_dims_data, output_data);
  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(overlap_add_runner->GetKernelRunner().InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestOverlapAddInvoke<T>(input_dims_data, input_data, output_dims_data,
                          golden_input, golden_output, iters, flexbuffers_data,
                          flexbuffers_data_size, output_data,
                          &overlap_add_runner->GetKernelRunner());
}

template <typename T>
void TestOverlapAddReset(int* input_dims_data, T* input_data,
                         int* output_dims_data, const T* golden_input,
                         const T* golden_output, int iters,
                         const unsigned char* flexbuffers_data,
                         const unsigned int flexbuffers_data_size,
                         T* output_data) {
  OverlapAddKernelRunner<T>* overlap_add_runner =
      new (overlap_add_kernel_runner_buffer) OverlapAddKernelRunner<T>(
          input_dims_data, input_data, output_dims_data, output_data);
  // TfLite uses a char* for the raw bytes whereas flexbuffers use an unsigned
  // char*. This small discrepancy results in compiler warnings unless we
  // reinterpret_cast right before passing in the flexbuffer bytes to the
  // KernelRunner.
  TF_LITE_MICRO_EXPECT_EQ(overlap_add_runner->GetKernelRunner().InitAndPrepare(
                              reinterpret_cast<const char*>(flexbuffers_data),
                              flexbuffers_data_size),
                          kTfLiteOk);
  TestOverlapAddInvoke(input_dims_data, input_data, output_dims_data,
                       golden_input, golden_output, iters, flexbuffers_data,
                       flexbuffers_data_size, output_data,
                       &overlap_add_runner->GetKernelRunner());
  overlap_add_runner->GetKernelRunner().Reset();
  TestOverlapAddInvoke(input_dims_data, input_data, output_dims_data,
                       golden_input, golden_output, iters, flexbuffers_data,
                       flexbuffers_data_size, output_data,
                       &overlap_add_runner->GetKernelRunner());
}

}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OverlapAddTestInt16) {
  const int kInputSize = 3;
  const int kOutputSize = 1;
  int input_dims_data[] = {2, 1, kInputSize};
  int output_dims_data[] = {1, kOutputSize};
  int16_t input_data[kInputSize];
  int16_t output_data = 0;
  const int16_t golden_input[] = {125, -12, -895, 1000, 65,  -212,
                                  63,  71,  52,   1,    -17, 32};
  const int16_t golden_output[] = {125, 988, -767, -140};

  tflite::TestOverlapAdd(input_dims_data, input_data, output_dims_data,
                         golden_input, golden_output,
                         sizeof(golden_output) / sizeof(int16_t),
                         g_gen_data_overlap_add_int16,
                         g_gen_data_size_overlap_add_int16, &output_data);
}

TF_LITE_MICRO_TEST(OverlapAddTestFloat) {
  const int kInputSize = 3;
  const int kOutputSize = 1;
  int input_dims_data[] = {2, 1, kInputSize};
  int output_dims_data[] = {1, kOutputSize};
  float input_data[kInputSize];
  float output_data = 0;
  const float golden_input[] = {12.5, -1.2, -89.5, 100.0, 6.5,  -21.2,
                                6.3,  7.1,  5.2,   0.1,   -1.7, 3.2};
  const float golden_output[] = {12.5, 98.8, -76.7, -14.0};

  tflite::TestOverlapAdd(input_dims_data, input_data, output_dims_data,
                         golden_input, golden_output,
                         sizeof(golden_output) / sizeof(float),
                         g_gen_data_overlap_add_float,
                         g_gen_data_size_overlap_add_float, &output_data);
}

TF_LITE_MICRO_TEST(OverlapAddTestNframes4Int16) {
  const int kInputSize = 3;
  const int kOutputSize = 1;
  const int kNFrames = 4;
  int input_dims_data[] = {2, kNFrames, kInputSize};
  int output_dims_data[] = {1, kNFrames * kOutputSize};
  int16_t input_data[kNFrames * kInputSize];
  int16_t output_data[kNFrames * kOutputSize];
  const int16_t golden_input[] = {125, -12, -895, 1000, 65,  -212,
                                  63,  71,  52,   1,    -17, 32};
  const int16_t golden_output[] = {125, 988, -767, -140};

  const int kIters =
      sizeof(golden_input) / kInputSize / kNFrames / sizeof(int16_t);
  tflite::TestOverlapAdd(input_dims_data, input_data, output_dims_data,
                         golden_input, golden_output, kIters,
                         g_gen_data_overlap_add_int16,
                         g_gen_data_size_overlap_add_int16, output_data);
}

TF_LITE_MICRO_TEST(OverlapAddTestNframes4OuterDims4Int16) {
  const int kInputSize = 3;
  const int kOutputSize = 1;
  const int kNFrames = 4;
  int input_dims_data[] = {4, 2, 2, kNFrames, kInputSize};
  int output_dims_data[] = {3, 2, 2, kNFrames * kOutputSize};
  int16_t input_data[2 * 2 * kNFrames * kInputSize];
  int16_t output_data[2 * 2 * kNFrames * kOutputSize];
  const int16_t golden_input[] = {125, -12, -895, 1000, 65,  -212,
                                  63,  71,  52,   1,    -17, 32};
  const int16_t golden_output[] = {125, 988, -767, -140};

  const int kIters =
      sizeof(golden_input) / kInputSize / kNFrames / sizeof(int16_t);
  tflite::TestOverlapAdd(input_dims_data, input_data, output_dims_data,
                         golden_input, golden_output, kIters,
                         g_gen_data_overlap_add_int16,
                         g_gen_data_size_overlap_add_int16, output_data);
}

TF_LITE_MICRO_TEST(testReset) {
  const int kInputSize = 3;
  const int kOutputSize = 1;
  const int kNFrames = 4;
  int input_dims_data[] = {4, 2, 2, kNFrames, kInputSize};
  int output_dims_data[] = {3, 2, 2, kNFrames * kOutputSize};
  int16_t input_data[2 * 2 * kNFrames * kInputSize];
  int16_t output_data[2 * 2 * kNFrames * kOutputSize];
  const int16_t golden_input[] = {125, -12, -895, 1000, 65,  -212,
                                  63,  71,  52,   1,    -17, 32};
  const int16_t golden_output[] = {125, 988, -767, -140};

  const int kIters =
      sizeof(golden_input) / kInputSize / kNFrames / sizeof(int16_t);
  tflite::TestOverlapAddReset(input_dims_data, input_data, output_dims_data,
                              golden_input, golden_output, kIters,
                              g_gen_data_overlap_add_int16,
                              g_gen_data_size_overlap_add_int16, output_data);
}

TF_LITE_MICRO_TESTS_END
