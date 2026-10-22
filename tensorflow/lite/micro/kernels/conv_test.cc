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

#include "tensorflow/lite/micro/kernels/conv_test.h"

#include <cmath>
#include <type_traits>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/testdata/conv_test_data.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {
// Common inputs and outputs.
constexpr int kInputElements = 16;
static int kInputShape[] = {4, 2, 2, 4, 1};
static const float kInputData[kInputElements] = {1, 1, 1, 1, 2, 2, 2, 2,
                                                 1, 2, 3, 4, 1, 2, 3, 4};

constexpr int kFilterElements = 12;
static int kFilterShape[] = {4, 3, 2, 2, 1};
static const float kFilterData[kFilterElements] = {1,  2, 3,  4,  -1, 1,
                                                   -1, 1, -1, -1, 1,  1};

constexpr int kBiasElements = 3;
static int kBiasShape[] = {1, 3};
static const float kBiasData[kBiasElements] = {1, 2, 3};

constexpr int kOutputElements = 12;
static int kOutputShape[] = {4, 2, 1, 2, 3};
static const float kGoldenData[kOutputElements] = {18, 2, 5, 18, 2, 5,
                                                   17, 4, 3, 37, 4, 3};

#ifdef USE_TFLM_COMPRESSION

// compressed filter data for kBinQuant scheme, matches kFilterData
// Align the tensor data the same as a Buffer in the schema
alignas(tflite::MicroArenaBufferAlignment()) constexpr uint8_t
    kBinQuantFilterData[] = {
        0x05, 0x38, 0x20, 0x90, 0x00,
};
constexpr float kBinQuantFilterValueTable[] = {
    1, 2, 3, 4, -1,
};
constexpr size_t kBinQuantFilterValueTableElements =
    std::extent<decltype(kBinQuantFilterValueTable)>::value;
constexpr int kBinQuantFilterBitWidth = 3;
// compressed bias data for kBinQuant scheme, matches kBiasData
// Align the tensor data the same as a Buffer in the schema
alignas(tflite::MicroArenaBufferAlignment()) constexpr uint8_t
    kBinQuantBiasData[] = {0x18};
constexpr int kBinQuantBiasBitWidth = 2;

// Common inputs and outputs for quantized compressed tensor tests.
// Values from TfLite conv_test.cc SimplePerChannelTest.
static int kInputShapeQ1[] = {4, 1, 2, 3, 2};
static const float kInputDataQ1[] = {
    // [1 * 2 * 3 * 2] as [batch, y, x, input_channel]
    3,  2,   // batch = 0, y = 0, x = 0
    1,  -1,  // batch = 0, y = 0, x = 1
    -2, -3,  // batch = 0, y = 0, x = 2
    4,  3,   // batch = 0, y = 1, x = 0
    2,  -2,  // batch = 0, y = 1, x = 1
    -3, -4,  // batch = 0, y = 1, x = 2
};
constexpr size_t kInputElementsQ1 = std::extent<decltype(kInputDataQ1)>::value;

constexpr int kNumChannelsQ1 = 2;
static int kFilterShapeQ1[] = {4, 2, 2, 2, 2};
// Original filter data:
// static constexpr float kFilterDataQ1[] = {
//     // [2 * 2 * 2 * 2] as [output_channel, y, x, input_channel]
//     1, 2,  // out channel = 0, y = 0, x = 0
//     3, 4,  // out channel = 0, y = 0, x = 1
//     3, 4,  // out channel = 0, y = 1, x = 0
//     5, 6,  // out channel = 0, y = 1, x = 1
//     7, 8,  // out channel = 1, y = 0, x = 0
//     5, 6,  // out channel = 1, y = 0, x = 1
//     3, 4,  // out channel = 1, y = 1, x = 0
//     1, 2,  // out channel = 1, y = 1, x = 1
// };

static int kBiasShapeQ1[] = {1, 2};
static const float kBiasDataQ1[] = {3, -2};
constexpr size_t kBiasElementsQ1 = std::extent<decltype(kBiasDataQ1)>::value;

static int kOutputShapeQ1[] = {4, 1, 1, 2, 2};
static const float kGoldenDataQ1[] = {31, 64, -57, -46};
constexpr int kOutputElementsQ1 = std::extent<decltype(kGoldenDataQ1)>::value;
static const float kGoldenDataQ1_16[] = {31, 63.99804688, -57, -46};

// compressed filter data for kBinQuant scheme, matches kFilterDataQ1
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantFilterDataQ1[] = {
    0x05, 0x34, 0xE5, 0xDE, 0x54, 0xC1,
};
constexpr float kBinQuantFilterValueTableQ1[] = {
    1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
};
constexpr size_t kBinQuantFilterValueTableElementsQ1 =
    std::extent<decltype(kBinQuantFilterValueTableQ1)>::value;
constexpr int kBinQuantFilterBitWidthQ1 = 3;
// compressed bias data for kBinQuant scheme, matches kBiasDataQ1
// Align the tensor data the same as a Buffer in the schema
alignas(16) constexpr uint8_t kBinQuantBiasDataQ1[] = {0x00};
constexpr int kBinQuantBiasBitWidthQ1 = 1;

static TfLiteConvParams common_conv_params_q1 = {
    kTfLitePaddingValid,  // padding
    1,                    // stride_width
    1,                    // stride_height
    kTfLiteActNone,       // activation
    1,                    // dilation_width_factor
    1,                    // dilation_height_factor
    kTfLiteNoType         // quantized_bias_type
};

#endif  // USE_TFLM_COMPRESSION

static TfLiteConvParams common_conv_params = {
    kTfLitePaddingValid,  // padding
    2,                    // stride_width
    2,                    // stride_height
    kTfLiteActNone,       // activation
    1,                    // dilation_width_factor
    1,                    // dilation_height_factor
    kTfLiteNoType         // quantized_bias_type
};

// Fills `data` with multiples of 0.25 in [-1, 1]. Products and sums of these
// values are exact in float32, so the result does not depend on the order in
// which a kernel accumulates them.
void FillConvTestPattern(float* data, int size, int seed) {
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<float>((i * 7 + seed) % 9 - 4) * 0.25f;
  }
}

// Fills `data` with values k / 7 in [-11/7, 11/7]. These are not exactly
// representable, so the result depends on the accumulation order.
void FillConvTestPatternInexact(float* data, int size, int seed) {
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<float>((i * 13 + seed) % 23 - 11) / 7.0f;
  }
}

// Runs a float convolution whose input, filter and bias are generated with
// FillConvTestPattern (or FillConvTestPatternInexact when `inexact_data` is
// true) using seeds 1, 2 and 3, and compares against `golden`. When
// `bias_shape` is nullptr the node is run without a bias tensor.
void TestConvFloatPattern(int* input_shape, float* input_data,
                          int* filter_shape, float* filter_data,
                          int* bias_shape, float* bias_data, int* output_shape,
                          const float* golden, float* output_data,
                          TfLiteConvParams* conv_params,
                          bool inexact_data = false) {
  void (*fill)(float*, int, int) =
      inexact_data ? FillConvTestPatternInexact : FillConvTestPattern;
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_shape);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  const int output_count = ElementCount(*output_dims);
  fill(input_data, ElementCount(*input_dims), 1);
  fill(filter_data, ElementCount(*filter_dims), 2);

  constexpr int kMaxTensors = 4;
  TfLiteTensor tensors[kMaxTensors];
  int tensors_size = 0;
  tensors[tensors_size++] = CreateTensor(input_data, input_dims);
  tensors[tensors_size++] = CreateTensor(filter_data, filter_dims);
  if (bias_shape != nullptr) {
    TfLiteIntArray* bias_dims = IntArrayFromInts(bias_shape);
    fill(bias_data, ElementCount(*bias_dims), 3);
    tensors[tensors_size++] = CreateTensor(bias_data, bias_dims);
  }
  tensors[tensors_size++] = CreateTensor(output_data, output_dims);

  int inputs_with_bias[] = {3, 0, 1, 2};
  int inputs_without_bias[] = {2, 0, 1};
  int* inputs_array_data =
      bias_shape != nullptr ? inputs_with_bias : inputs_without_bias;
  int outputs_array_data[] = {1, tensors_size - 1};
  micro::KernelRunner runner(Register_CONV_2D(), tensors, tensors_size,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data), conv_params);
  ASSERT_EQ(runner.InitAndPrepare(reinterpret_cast<const char*>(conv_params)),
            kTfLiteOk);
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
  // Exact data must match exactly (up to the 1e-5 used elsewhere in this
  // file). Inexact data may differ by a few ULP depending on the accumulation
  // order, so a relative term is added.
  const float rel_tolerance = inexact_data ? 1e-5f : 0.0f;
  for (int i = 0; i < output_count; ++i) {
    EXPECT_NEAR(golden[i], output_data[i],
                1e-5f + rel_tolerance * std::fabs(golden[i]));
  }
}

class BiasMetadataFailureMicroContext : public MicroContext {
 public:
  explicit BiasMetadataFailureMicroContext(TfLiteTensor* tensors)
      : tensors_(tensors) {}

  void* AllocatePersistentBuffer(size_t bytes) override {
    if (persistent_buffer_used_ + bytes > sizeof(persistent_buffer_)) {
      return nullptr;
    }
    void* result = persistent_buffer_ + persistent_buffer_used_;
    persistent_buffer_used_ += bytes;
    return result;
  }
  TfLiteStatus RequestScratchBufferInArena(size_t, int* buffer_idx) override {
    *buffer_idx = 0;
    return kTfLiteOk;
  }
  void* GetScratchBuffer(int) override { return nullptr; }
  TfLiteTensor* AllocateTempTfLiteTensor(int tensor_idx) override {
    if (tensor_idx == kConvBiasTensor && ++bias_allocation_count_ == 2) {
      return nullptr;
    }
    ++allocation_count_[tensor_idx];
    return &tensors_[tensor_idx];
  }
  void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) override {
    for (int i = 0; i < 4; ++i) {
      if (tensor == &tensors_[i]) {
        ++deallocation_count_[i];
        return;
      }
    }
  }
  uint8_t* AllocateTempBuffer(size_t, size_t) override { return nullptr; }
  void DeallocateTempBuffer(uint8_t*) override {}
  TfLiteEvalTensor* GetEvalTensor(int) override { return nullptr; }
  TfLiteStatus set_external_context(void*) override { return kTfLiteError; }
  void* external_context() override { return nullptr; }
  MicroGraph& graph() override { return *graph_; }

  int bias_allocation_count() const { return bias_allocation_count_; }
  int allocation_count(int tensor_idx) const {
    return allocation_count_[tensor_idx];
  }
  int deallocation_count(int tensor_idx) const {
    return deallocation_count_[tensor_idx];
  }

 private:
  alignas(MicroArenaBufferAlignment()) uint8_t persistent_buffer_[128] = {};
  size_t persistent_buffer_used_ = 0;
  TfLiteTensor* tensors_;
  int bias_allocation_count_ = 0;
  int allocation_count_[4] = {};
  int deallocation_count_[4] = {};
  MicroGraph* graph_ = nullptr;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(ConvTest, RealBiasMetadataFailureIsPropagated) {
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  int input_shape[] = {4, 1, 1, 1, 1};
  int filter_shape[] = {4, 1, 1, 1, 1};
  int bias_shape[] = {1, 1};
  int output_shape[] = {4, 1, 1, 1, 1};
  float input_data[] = {1};
  float filter_data[] = {1};
  float bias_data[] = {1};
  float output_data[] = {0};
  TfLiteTensor tensors[] = {
      CreateTensor(input_data, IntArrayFromInts(input_shape)),
      CreateTensor(filter_data, IntArrayFromInts(filter_shape)),
      CreateTensor(bias_data, IntArrayFromInts(bias_shape)),
      CreateTensor(output_data, IntArrayFromInts(output_shape)),
  };
  int inputs_data[] = {3, 0, 1, 2};
  int outputs_data[] = {1, 3};
  TfLiteNode node = {};
  node.inputs = IntArrayFromInts(inputs_data);
  node.outputs = IntArrayFromInts(outputs_data);
  TfLiteConvParams params = {kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1,
                             kTfLiteNoType};
  node.builtin_data = &params;

  tflite::testing::BiasMetadataFailureMicroContext micro_context(tensors);
  TfLiteContext context = {};
  micro_context.InitTfLiteContext(&context);
  const TFLMRegistration registration = tflite::Register_CONV_2D();
  node.user_data = registration.init(&context, nullptr, 0);
  ASSERT_NE(node.user_data, nullptr);

  const TfLiteStatus status = registration.prepare(&context, &node);
  if (micro_context.bias_allocation_count() == 2) {
    EXPECT_EQ(status, kTfLiteError);
    EXPECT_EQ(micro_context.allocation_count(0),
              micro_context.deallocation_count(0));
    EXPECT_EQ(micro_context.allocation_count(1),
              micro_context.deallocation_count(1));
    EXPECT_EQ(micro_context.allocation_count(3),
              micro_context.deallocation_count(3));
    EXPECT_EQ(micro_context.deallocation_count(2), 1);
  } else {
    EXPECT_EQ(micro_context.bias_allocation_count(), 1);
    EXPECT_EQ(status, kTfLiteOk);
  }
}

TEST(ConvTest, OptionalBiasShouldPrepareAndInvoke) {
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  int input_shape[] = {4, 1, 2, 2, 1};
  int filter_shape[] = {4, 1, 1, 1, 1};
  int output_shape[] = {4, 1, 2, 2, 1};
  float input_data[] = {1, 2, 3, 4};
  float filter_data[] = {2};
  float output_data[] = {0, 0, 0, 0};
  TfLiteTensor tensors[] = {
      CreateTensor(input_data, IntArrayFromInts(input_shape)),
      CreateTensor(filter_data, IntArrayFromInts(filter_shape)),
      {},
      CreateTensor(output_data, IntArrayFromInts(output_shape)),
  };
  int inputs_data[] = {3, 0, 1, kTfLiteOptionalTensor};
  int outputs_data[] = {1, 3};
  TfLiteConvParams params = {kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1,
                             kTfLiteNoType};
  tflite::micro::KernelRunner runner(tflite::Register_CONV_2D(), tensors, 4,
                                     IntArrayFromInts(inputs_data),
                                     IntArrayFromInts(outputs_data), &params);

  ASSERT_EQ(runner.InitAndPrepare(reinterpret_cast<const char*>(&params)),
            kTfLiteOk);
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
  const float expected[] = {2, 4, 6, 8};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(output_data[i], expected[i]);
  }
}

TEST(ConvTest, SimpleTestQuantized4bitPerChannel) {
  const int output_dims_count = 12;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, tflite::testing::kBiasData,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      tflite::testing::kGoldenData, golden_quantized, output_scale,
      output_zero_point, &tflite::testing::common_conv_params,
      tflite::Register_CONV_2D(), output_data, kTfLiteInt4);
}

TEST(ConvTest, SimpleTestQuantizedPerChannel) {
  const int output_dims_count = 12;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, tflite::testing::kBiasData,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      tflite::testing::kGoldenData, golden_quantized, output_scale,
      output_zero_point, &tflite::testing::common_conv_params,
      tflite::Register_CONV_2D(), output_data);
}

#ifdef USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestQuantizedPerChannelCompressed) {
  const float input_scale = 0.5f;
  const float output_scale = 0.5f;
  const int input_zero_point = -1;
  const int output_zero_point = -1;
  constexpr float filter_scales[] = {tflite::testing::kNumChannelsQ1, 1.0f,
                                     2.0f};
  constexpr int filter_zero_points[] = {tflite::testing::kNumChannelsQ1, 0, 0};
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int8_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kBinQuantFilterValueTableElementsQ1];
  int32_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int8_t golden_quantized[tflite::testing::kOutputElementsQ1];
  int8_t output_quantized[tflite::testing::kOutputElementsQ1];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int32_t> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = filter_quantized;
  filter_comp_info.value_table_stride =
      tflite::testing::kBinQuantFilterValueTableElementsQ1 /
      tflite::testing::kNumChannelsQ1;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidthQ1;
  filter_comp_info.compressed = tflite::testing::kBinQuantFilterDataQ1;
  filter_comp_info.data = tflite::testing::kBinQuantFilterValueTableQ1;
  filter_comp_info.dims_data = tflite::testing::kFilterShapeQ1;
  filter_comp_info.scales = filter_scales;
  filter_comp_info.zero_points = filter_zero_points;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride =
      tflite::testing::kBiasElementsQ1 / tflite::testing::kNumChannelsQ1;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidthQ1;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasDataQ1;
  bias_comp_info.data = tflite::testing::kBiasDataQ1;
  bias_comp_info.dims_data = tflite::testing::kBiasShapeQ1;
  bias_comp_info.scales = bias_scales;
  bias_comp_info.zero_points = bias_zero_points;

  tflite::testing::TestConvQuantizedPerChannelCompressed(
      tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1,
      golden_quantized, output_quantized, output_scale, output_zero_point,
      &tflite::testing::common_conv_params_q1, tflite::Register_CONV_2D(),
      &filter_comp_info, &bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestFloat) {
  float output_data[tflite::testing::kOutputElements];

  tflite::testing::TestConvFloat(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      tflite::testing::kBiasShape, tflite::testing::kBiasData,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      &tflite::testing::common_conv_params, tflite::Register_CONV_2D(),
      output_data);
}

#ifdef USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestFloatCompressed) {
  tflite::testing::TestCompressionInfo<const float> filter_comp_info = {};
  tflite::testing::TestCompressionInfo<const float> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = tflite::testing::kBinQuantFilterValueTable;
  filter_comp_info.value_table_stride =
      tflite::testing::kBinQuantFilterValueTableElements;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidth;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = tflite::testing::kBiasData;
  bias_comp_info.value_table_stride = tflite::testing::kBiasElements;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidth;

  float output_data[tflite::testing::kOutputElements];

  tflite::testing::TestConvFloat(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      tflite::testing::kFilterShape,
      reinterpret_cast<const float*>(tflite::testing::kBinQuantFilterData),
      tflite::testing::kBiasShape,
      reinterpret_cast<const float*>(tflite::testing::kBinQuantBiasData),
      tflite::testing::kOutputShape, tflite::testing::kGoldenData,
      &tflite::testing::common_conv_params, tflite::Register_CONV_2D(),
      output_data, &filter_comp_info, &bias_comp_info);
}

#endif

TEST(ConvTest, InputAndFilterSameWidthHeight) {
  const int output_dims_count = 2;
  float output_data[output_dims_count];

  int kFilterShape[] = {4, 1, 2, 4, 1};
  const float filter_values[] = {1, 2, 3, 4, -1, -1, 1, 1};
  int kBiasShape[] = {1, 1};
  const float bias_values[] = {0};
  int kOutputShape[] = {4, 2, 1, 1, 1};
  const float expected_output[] = {10, 34};

  tflite::testing::TestConvFloat(
      tflite::testing::kInputShape, tflite::testing::kInputData, kFilterShape,
      filter_values, kBiasShape, bias_values, kOutputShape, expected_output,
      &tflite::testing::common_conv_params, tflite::Register_CONV_2D(),
      output_data);
}

TEST(ConvTest, InputOutputDifferentTypeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  const int output_dims_count = tflite::ElementCount(*output_dims);
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t output_data[tflite::testing::kOutputElements];
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateTensor(tflite::testing::kFilterData, filter_dims),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/0.0f,
                            /*zero_point=*/0),
  };
  EXPECT_EQ(kTfLiteError, tflite::testing::InvokeConv(
                              tensors, tensors_size, output_dims_count,
                              &tflite::testing::common_conv_params,
                              tflite::Register_CONV_2D(), output_data));
}

TEST(ConvTest, HybridModeIsError) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims = IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims = IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims = IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims = IntArrayFromInts(tflite::testing::kOutputShape);
  const int output_dims_count = tflite::ElementCount(*output_dims);
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  int8_t filter_data[tflite::testing::kFilterElements] = {};
  float output_data[tflite::testing::kOutputElements];
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(tflite::testing::kInputData, input_dims),
      CreateQuantizedTensor(filter_data, filter_dims,
                            /*scale=*/0.0f,
                            /*zero_point=*/0),
      CreateTensor(tflite::testing::kBiasData, bias_dims),
      CreateTensor(output_data, output_dims),
  };
  EXPECT_EQ(kTfLiteError, tflite::testing::InvokeConv(
                              tensors, tensors_size, output_dims_count,
                              &tflite::testing::common_conv_params,
                              tflite::Register_CONV_2D(), output_data));
}

TEST(ConvTest, SimpleTestQuantized16x8PerChannel64bBias) {
  const int output_dims_count = 12;
  int16_t output_data[output_dims_count];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  std::int64_t bias_quantized[tflite::testing::kBiasElements];
  int16_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, tflite::testing::kBiasData,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      tflite::testing::kGoldenData, golden_quantized, output_scale,
      output_zero_point, &tflite::testing::common_conv_params,
      tflite::Register_CONV_2D(), output_data);
}

#ifdef USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestQuantized16x8PerChannel64bBiasCompressed) {
  const float input_scale = 128.0f / 65536;
  const float output_scale = 128.0f / 65536;
  const int input_zero_point = 0;
  const int output_zero_point = 0;
  constexpr float filter_scales[] = {tflite::testing::kNumChannelsQ1, 1.0f,
                                     2.0f};
  constexpr int filter_zero_points[] = {tflite::testing::kNumChannelsQ1, 0, 0};
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kBinQuantFilterValueTableElementsQ1];
  int64_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int16_t golden_quantized[tflite::testing::kOutputElementsQ1];
  int16_t output_quantized[tflite::testing::kOutputElementsQ1];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int64_t> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = filter_quantized;
  filter_comp_info.value_table_stride =
      tflite::testing::kBinQuantFilterValueTableElementsQ1 /
      tflite::testing::kNumChannelsQ1;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidthQ1;
  filter_comp_info.compressed = tflite::testing::kBinQuantFilterDataQ1;
  filter_comp_info.data = tflite::testing::kBinQuantFilterValueTableQ1;
  filter_comp_info.dims_data = tflite::testing::kFilterShapeQ1;
  filter_comp_info.scales = filter_scales;
  filter_comp_info.zero_points = filter_zero_points;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride =
      tflite::testing::kBiasElementsQ1 / tflite::testing::kNumChannelsQ1;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidthQ1;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasDataQ1;
  bias_comp_info.data = tflite::testing::kBiasDataQ1;
  bias_comp_info.dims_data = tflite::testing::kBiasShapeQ1;
  bias_comp_info.scales = bias_scales;
  bias_comp_info.zero_points = bias_zero_points;

  tflite::testing::TestConvQuantizedPerChannelCompressed<int16_t, std::int64_t>(
      tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1_16,
      golden_quantized, output_quantized, output_scale, output_zero_point,
      &tflite::testing::common_conv_params_q1, tflite::Register_CONV_2D(),
      &filter_comp_info, &bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestQuantized16x8PerChannel32bBias) {
  const int output_dims_count = 12;
  int16_t output_data[output_dims_count];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int16_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, tflite::testing::kBiasData,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      tflite::testing::kGoldenData, golden_quantized, output_scale,
      output_zero_point, &tflite::testing::common_conv_params,
      tflite::Register_CONV_2D(), output_data);
}

#ifdef USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestQuantized16x8PerChannel32bBiasCompressed) {
  const float input_scale = 128.0f / 65536;
  const float output_scale = 128.0f / 65536;
  const int input_zero_point = 0;
  const int output_zero_point = 0;
  constexpr float filter_scales[] = {tflite::testing::kNumChannelsQ1, 1.0f,
                                     2.0f};
  constexpr int filter_zero_points[] = {tflite::testing::kNumChannelsQ1, 0, 0};
  // bias scales and zero points will be computed
  float bias_scales[std::extent<decltype(filter_scales)>::value] = {};
  int bias_zero_points[std::extent<decltype(filter_scales)>::value] = {};

  int16_t input_quantized[tflite::testing::kInputElementsQ1];
  int8_t filter_quantized[tflite::testing::kBinQuantFilterValueTableElementsQ1];
  int32_t bias_quantized[tflite::testing::kBiasElementsQ1];
  int16_t golden_quantized[tflite::testing::kOutputElementsQ1];
  int16_t output_quantized[tflite::testing::kOutputElementsQ1];

  tflite::testing::TestCompressionQuantizedInfo<int8_t> filter_comp_info = {};
  tflite::testing::TestCompressionQuantizedInfo<int32_t> bias_comp_info = {};

  filter_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  filter_comp_info.value_table = filter_quantized;
  filter_comp_info.value_table_stride =
      tflite::testing::kBinQuantFilterValueTableElementsQ1 /
      tflite::testing::kNumChannelsQ1;
  filter_comp_info.bit_width = tflite::testing::kBinQuantFilterBitWidthQ1;
  filter_comp_info.compressed = tflite::testing::kBinQuantFilterDataQ1;
  filter_comp_info.data = tflite::testing::kBinQuantFilterValueTableQ1;
  filter_comp_info.dims_data = tflite::testing::kFilterShapeQ1;
  filter_comp_info.scales = filter_scales;
  filter_comp_info.zero_points = filter_zero_points;

  bias_comp_info.scheme = tflite::CompressionScheme::kBinQuant;
  bias_comp_info.value_table = bias_quantized;
  bias_comp_info.value_table_stride =
      tflite::testing::kBiasElementsQ1 / tflite::testing::kNumChannelsQ1;
  bias_comp_info.bit_width = tflite::testing::kBinQuantBiasBitWidthQ1;
  bias_comp_info.compressed = tflite::testing::kBinQuantBiasDataQ1;
  bias_comp_info.data = tflite::testing::kBiasDataQ1;
  bias_comp_info.dims_data = tflite::testing::kBiasShapeQ1;
  bias_comp_info.scales = bias_scales;
  bias_comp_info.zero_points = bias_zero_points;

  tflite::testing::TestConvQuantizedPerChannelCompressed(
      tflite::testing::kInputShapeQ1, tflite::testing::kInputDataQ1,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kOutputShapeQ1, tflite::testing::kGoldenDataQ1_16,
      golden_quantized, output_quantized, output_scale, output_zero_point,
      &tflite::testing::common_conv_params_q1, tflite::Register_CONV_2D(),
      &filter_comp_info, &bias_comp_info);
}

#endif  // USE_TFLM_COMPRESSION

TEST(ConvTest, SimpleTestDilatedQuantizedPerChannel) {
  const int output_dims_count = 24;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  const int input_elements = 48;
  int input_shape[] = {4, 2, 4, 6, 1};
  const float input_data[] = {
      // b = 0
      1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
      // b = 1
      1, 2, 3, 4, 5, 6, 2, 6, 2, 4, 4, 2, 3, 2, 6, 5, 1, 4, 1, 2, 1, 4, 6, 3};
  const int output_elements = 24;
  int output_shape[] = {4, 2, 2, 2, 3};
  const float golden_data[] = {25, 2, 7, 25, 2, 7, 10, 2, -3, 10, 2, -3,
                               39, 7, 6, 50, 3, 4, 14, 4, -5, 15, 0, -7};

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[output_elements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TfLiteConvParams conv_params{tflite::testing::common_conv_params};
  conv_params.dilation_width_factor = 3;
  conv_params.dilation_height_factor = 2;

  tflite::testing::TestConvQuantizedPerChannel(
      input_shape, input_data, input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, tflite::testing::kBiasData,
      bias_quantized, scales, zero_points, output_shape, golden_data,
      golden_quantized, output_scale, output_zero_point, &conv_params,
      tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, SimpleTestQuantizedPerChannelRelu6) {
  const int output_dims_count = 12;
  int8_t output_data[output_dims_count];

  const float bias_values[] = {1, 2, -3};
  const float golden_data[] = {6, 2, 0, 6, 2, 0, 6, 4, 0, 6, 4, 0};

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, bias_values,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      golden_data, golden_quantized, output_scale, output_zero_point,
      &tflite::testing::common_conv_params, tflite::Register_CONV_2D(),
      output_data);
}

TEST(ConvTest, SimpleTestQuantized16x8PerChannelRelu664bBias) {
  const int output_dims_count = 12;
  int16_t output_data[output_dims_count];

  const float bias_values[] = {1, 2, -3};
  const float golden_data[] = {6, 2, 0, 6, 2, 0, 6, 4, 0, 6, 4, 0};

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  std::int64_t bias_quantized[tflite::testing::kBiasElements];
  int16_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TfLiteConvParams conv_params{tflite::testing::common_conv_params};
  conv_params.activation = kTfLiteActRelu6;
  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, bias_values,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      golden_data, golden_quantized, output_scale, output_zero_point,
      &conv_params, tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, SimpleTestQuantized16x8PerChannelRelu632bBias) {
  const int output_dims_count = 12;
  int16_t output_data[output_dims_count];

  const float bias_values[] = {1, 2, -3};
  const float golden_data[] = {6, 2, 0, 6, 2, 0, 6, 4, 0, 6, 4, 0};

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int16_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TfLiteConvParams conv_params{tflite::testing::common_conv_params};
  conv_params.activation = kTfLiteActRelu6;
  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInputShape, tflite::testing::kInputData,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilterShape, tflite::testing::kFilterData,
      filter_quantized, tflite::testing::kBiasShape, bias_values,
      bias_quantized, scales, zero_points, tflite::testing::kOutputShape,
      golden_data, golden_quantized, output_scale, output_zero_point,
      &conv_params, tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, Kernel1x1QuantizedPerChannel) {
  // conv params:
  // padding, stride_<width,height>, activation, dilation_<width, height>
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  int input_shape[] = {4, 1, 2, 2, 4};  // [len,N,H,W,C]
  constexpr int input_elements =
      1 * 2 * 2 *
      4;  // input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4];
  constexpr float input_data[input_elements] = {1, 1, 1, 1, 2, 2, 2, 2,
                                                1, 2, 3, 4, 1, 2, 3, 4};

  int filter_shape[] = {4, 3, 1, 1, 4};
  constexpr int filter_elements =
      3 * 1 * 1 * 4;  //      filter_shape[1] * filter_shape[2] *
                      //      filter_shape[3] * filter_shape[4];
  const float filter_data[filter_elements] = {1,  2, 3,  4,  -1, 1,
                                              -1, 1, -1, -1, 1,  1};

  constexpr int bias_elements = 3;  // filter_shape[1];
  int bias_shape[] = {1, bias_elements};
  constexpr float bias_data[bias_elements] = {1, 2, 3};

  int output_shape[] = {4, 1, 2, 2, bias_elements};
  constexpr int output_elements = 4 * 3;
  int8_t output_data[output_elements];

  const float golden_data[output_elements] = {11, 2, 3, 21, 2, 3,
                                              31, 4, 7, 31, 4, 7};

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      input_shape, input_data, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_data, filter_quantized, bias_shape, bias_data,
      bias_quantized, scales, zero_points, output_shape, golden_data,
      golden_quantized, output_scale, output_zero_point, &conv_params,
      tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, Kernel1x1QuantizedPerChannelRelu6) {
  // conv params:
  // padding, stride_<width,height>, activation, dilation_<width, height>
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActRelu6, 1, 1, kTfLiteNoType};

  int input_shape[] = {4, 1, 2, 2, 4};  // [len,N,H,W,C]
  constexpr int input_elements =
      1 * 2 * 2 *
      4;  // input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4];
  constexpr float input_data[input_elements] = {1, 1, 1, 1, 2, 2, 2, 2,
                                                1, 2, 3, 4, 1, 2, 3, 4};

  int filter_shape[] = {4, 3, 1, 1, 4};
  constexpr int filter_elements =
      3 * 1 * 1 * 4;  //      filter_shape[1] * filter_shape[2] *
                      //      filter_shape[3] * filter_shape[4];
  const float filter_data[filter_elements] = {1,  2, 3,  4,  -1, 1,
                                              -1, 1, -1, -1, 1,  1};

  constexpr int bias_elements = 3;  // filter_shape[1];
  int bias_shape[] = {1, bias_elements};
  constexpr float bias_data[bias_elements] = {1, 2, -3};

  int output_shape[] = {4, 1, 2, 2, bias_elements};
  constexpr int output_elements = 4 * 3;
  int8_t output_data[output_elements];

  const float golden_data[output_elements] = {6, 2, 0, 6, 2, 0,
                                              6, 4, 1, 6, 4, 1};

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      input_shape, input_data, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_data, filter_quantized, bias_shape, bias_data,
      bias_quantized, scales, zero_points, output_shape, golden_data,
      golden_quantized, output_scale, output_zero_point, &conv_params,
      tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, Kernel1x1Quantized16x8PerChannelRelu6) {
  // conv params:
  // padding, stride_<width,height>, activation, dilation_<width, height>
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActRelu6, 1, 1, kTfLiteNoType};

  int input_shape[] = {4, 1, 2, 2, 4};  // [len,N,H,W,C]
  const int input_elements = 1 * 2 * 2 * 4;
  const float input_data[input_elements] = {1, 1, 1, 1, 2, 2, 2, 2,
                                            1, 2, 3, 4, 1, 2, 3, 4};

  int filter_shape[] = {4, 3, 1, 1, 4};
  const int filter_elements = 3 * 1 * 1 * 4;
  const float filter_data[filter_elements] = {1,  2, 3,  4,  -1, 1,
                                              -1, 1, -1, -1, 1,  1};

  const int bias_elements = 3;
  int bias_shape[] = {1, bias_elements};
  const float bias_data[bias_elements] = {1, 2, -3};

  int output_shape[] = {4, 1, 2, 2, bias_elements};
  const int output_elements = 4 * 3;
  int16_t output_data[output_elements];

  const float golden_data[output_elements] = {6, 2, 0, 6, 2, 0,
                                              6, 4, 1, 6, 4, 1};

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int16_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  std::int64_t bias_quantized[bias_elements];
  int16_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      input_shape, input_data, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_data, filter_quantized, bias_shape, bias_data,
      bias_quantized, scales, zero_points, output_shape, golden_data,
      golden_quantized, output_scale, output_zero_point, &conv_params,
      tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, BroadcastPerLayerQuantizationToPerChannelShouldMatchGolden) {
  const int output_dims_count = 12;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0f;
  const float filter_scale = 1.0f;
  const float output_scale = 1.0f;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];

  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kOutputShape);

  // Create per-layer quantized int8_t input tensor.
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      tflite::testing::kInputData, input_quantized, input_dims, input_scale, 0);
  int input_zero_points[2] = {1, 0};
  float input_scales[2] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-layer quantized int8_t filter tensor.
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      tflite::testing::kFilterData, filter_quantized, filter_dims, filter_scale,
      0);
  int filter_zero_points[2] = {1, 0};
  float filter_scales[2] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-layer quantized int32_t bias tensor.
  tflite::SymmetricQuantize(tflite::testing::kBiasData, bias_quantized,
                            tflite::testing::kBiasElements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  int bias_zero_points[2] = {1, 0};
  float bias_scales[2] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-layer quantized int8_t output tensor.
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, 0 /* quantized dimension */);
  int output_zero_points[2] = {1, 0};
  float output_scales[2] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  tflite::Quantize(tflite::testing::kGoldenData, golden_quantized,
                   output_dims_count, output_scale, 0);

  tflite::testing::ValidateConvGoldens(tensors, tensors_size, golden_quantized,
                                       output_dims_count,
                                       &tflite::testing::common_conv_params,
                                       tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, Int8Filter1x3x3x1ShouldMatchGoldenEvenInputPaddingSame) {
  using tflite::ElementCount;
  using tflite::kConvFilter1x3x3x1;
  using tflite::kConvGoldenOutput4x4InputPaddingSame2x2;
  using tflite::kConvInput1x4x4x1;
  using tflite::kConvZeroBias;
  using tflite::testing::CreateTensor;
  using tflite::testing::FloatArrayFromFloats;
  using tflite::testing::IntArrayFromInts;
  using tflite::testing::ValidateConvGoldens;

  constexpr int kInDepth = 1;
  constexpr int kOutDepth = 1;

  // Input quantization parameters: same scale and zero point for all input
  // elements.
  constexpr float kInputScale = 0.00392120517f;
  constexpr int kInputZeroPoint = -128;
  float input_scales[] = {1, kInputScale};
  int input_zero_points[] = {1, kInputZeroPoint};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points),
                                          0};
  // Create input tensor of size 1x4x4x1.
  int input_shape[] = {4, 1, 4, 4, kInDepth};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteTensor input_tensor = CreateTensor(kConvInput1x4x4x1, input_dims);
  input_tensor.params = {kInputScale, kInputZeroPoint};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Filter quantization parameters.
  int filter_zero_points[kOutDepth + 1] = {kOutDepth, 0};
  float filter_scales[kOutDepth + 1] = {kOutDepth, 0.00448552053f};
  TfLiteAffineQuantization filter_quant;
  filter_quant.scale = FloatArrayFromFloats(filter_scales);
  filter_quant.zero_point = IntArrayFromInts(filter_zero_points);
  filter_quant.quantized_dimension = 0;

  // Create filter tensor of size 1x3x3x1.
  int filter_shape[] = {4, kOutDepth, 3, 3, kInDepth};
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_shape);
  TfLiteTensor filter_tensor = CreateTensor(kConvFilter1x3x3x1, filter_dims);
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Bias quantization parameters: same zero point, but different scale per
  // output channel.
  int bias_zero_points[kOutDepth + 1] = {kOutDepth, 0};
  float bias_scales[kOutDepth + 1] = {kOutDepth, 0.00001758864f};
  TfLiteAffineQuantization bias_quant;
  bias_quant.scale = FloatArrayFromFloats(bias_scales);
  bias_quant.zero_point = IntArrayFromInts(bias_zero_points);
  bias_quant.quantized_dimension = 0;

  // Create size 1 zero bias tensor.
  int bias_shape[] = {1, kOutDepth};
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_shape);
  TfLiteTensor bias_tensor = CreateTensor(kConvZeroBias, bias_dims);
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Output quantization parameters: same zero point and scale for all elements.
  const float output_scale = 0.00627814838f;
  const int output_zero_point = -7;
  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {FloatArrayFromFloats(output_scales),
                                           IntArrayFromInts(output_zero_points),
                                           0};

  // Create output tensor of 1x2x2x1.
  int8_t output_data[4 * 2 * 2 * kOutDepth];
  int output_shape[] = {4, 1, 2, 2, kOutDepth};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  const int output_dims_count = ElementCount(*output_dims);
  TfLiteTensor output_tensor = CreateTensor(output_data, output_dims);
  output_tensor.params = {output_scale, output_zero_point};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  TfLiteConvParams conv_params{tflite::testing::common_conv_params};
  conv_params.padding = kTfLitePaddingSame;

  ValidateConvGoldens(
      tensors, tensors_size, kConvGoldenOutput4x4InputPaddingSame2x2,
      output_dims_count, &conv_params, tflite::Register_CONV_2D(), output_data,
      1.0 /* tolerance */);
}

TEST(ConvTest, Int8Filter1x3x3x1ShouldMatchGoldenOddInputPaddingSame) {
  using tflite::ElementCount;
  using tflite::kConvFilter1x3x3x1;
  using tflite::kConvGoldenOutput5x5InputPaddingSame3x3;
  using tflite::kConvInput1x5x5x1;
  using tflite::kConvZeroBias;
  using tflite::testing::CreateTensor;
  using tflite::testing::FloatArrayFromFloats;
  using tflite::testing::IntArrayFromInts;
  using tflite::testing::ValidateConvGoldens;

  constexpr int kInDepth = 1;
  constexpr int kOutDepth = 1;

  // Input quantization parameters: same scale and zero point for all input
  // elements.
  constexpr float kInputScale = 0.00392120517f;
  constexpr int kInputZeroPoint = -128;
  float input_scales[] = {1, kInputScale};
  int input_zero_points[] = {1, kInputZeroPoint};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points),
                                          0};
  // Create input tensor of size 1x5x5x1.
  int input_shape[] = {4, 1, 5, 5, kInDepth};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteTensor input_tensor = CreateTensor(kConvInput1x5x5x1, input_dims);
  input_tensor.params = {kInputScale, kInputZeroPoint};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Filter quantization parameters.
  int filter_zero_points[kOutDepth + 1] = {kOutDepth, 0};
  float filter_scales[kOutDepth + 1] = {kOutDepth, 0.00448552053f};
  TfLiteAffineQuantization filter_quant;
  filter_quant.scale = FloatArrayFromFloats(filter_scales);
  filter_quant.zero_point = IntArrayFromInts(filter_zero_points);
  filter_quant.quantized_dimension = 0;

  // Create filter tensor of size 1x3x3x1.
  int filter_shape[] = {4, kOutDepth, 3, 3, kInDepth};
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_shape);
  TfLiteTensor filter_tensor = CreateTensor(kConvFilter1x3x3x1, filter_dims);
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Bias quantization parameters: same zero point, but different scale per
  // output channel.
  int bias_zero_points[kOutDepth + 1] = {kOutDepth, 0};
  float bias_scales[kOutDepth + 1] = {kOutDepth, 0.00001758864f};
  TfLiteAffineQuantization bias_quant;
  bias_quant.scale = FloatArrayFromFloats(bias_scales);
  bias_quant.zero_point = IntArrayFromInts(bias_zero_points);
  bias_quant.quantized_dimension = 0;

  // Create size 1 zero bias tensor.
  int bias_shape[] = {1, kOutDepth};
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_shape);
  TfLiteTensor bias_tensor = CreateTensor(kConvZeroBias, bias_dims);
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Output quantization parameters: same zero point and scale for all elements.
  const float output_scale = 0.00627814838f;
  const int output_zero_point = -7;
  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {FloatArrayFromFloats(output_scales),
                                           IntArrayFromInts(output_zero_points),
                                           0};

  // Create output tensor.
  int8_t output_data[4 * 3 * 3 * kOutDepth];
  int output_shape[] = {4, 1, 3, 3, kOutDepth};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  const int output_dims_count = ElementCount(*output_dims);
  TfLiteTensor output_tensor = CreateTensor(output_data, output_dims);
  output_tensor.params = {output_scale, output_zero_point};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  TfLiteConvParams conv_params{tflite::testing::common_conv_params};
  conv_params.padding = kTfLitePaddingSame;

  ValidateConvGoldens(
      tensors, tensors_size, kConvGoldenOutput5x5InputPaddingSame3x3,
      output_dims_count, &conv_params, tflite::Register_CONV_2D(), output_data,
      1.0 /* tolerance */);
}

TEST(ConvTest, FilterDimsNotMatchingAffineQuantization) {
  const int output_dims_count = 12;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5f;
  const float output_scale = 1.0f;

  int8_t input_quantized[tflite::testing::kInputElements];
  int8_t filter_quantized[tflite::testing::kFilterElements];
  int32_t bias_quantized[tflite::testing::kBiasElements];
  int8_t golden_quantized[tflite::testing::kOutputElements];
  int zero_points[tflite::testing::kBiasElements + 1];
  float scales[tflite::testing::kBiasElements + 1];

  TfLiteIntArray* input_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* bias_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kBiasShape);
  TfLiteIntArray* output_dims =
      tflite::testing::IntArrayFromInts(tflite::testing::kOutputShape);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      tflite::testing::kInputData, input_quantized, input_dims, input_scale, 0);
  TfLiteTensor filter_tensor =
      tflite::testing::CreateSymmetricPerChannelQuantizedTensor(
          tflite::testing::kFilterData, filter_quantized, filter_dims,
          filter_scales, filter_zero_points, &filter_quant,
          0 /* quantized dimension */);
  TfLiteTensor bias_tensor =
      tflite::testing::CreatePerChannelQuantizedBiasTensor(
          tflite::testing::kBiasData, bias_quantized, bias_dims, input_scale,
          &filter_scales[1], scales, zero_points, &bias_quant, 0);
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, 0 /* quantized dimension */);

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, 128};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  tflite::Quantize(tflite::testing::kGoldenData, golden_quantized,
                   output_dims_count, output_scale, 0);

  // Set filter quant to mismatched dimension.
  TfLiteAffineQuantization* quant = reinterpret_cast<TfLiteAffineQuantization*>(
      filter_tensor.quantization.params);

  // Choose arbitrary incorrect scale and zero point sizes which are neither 1
  // (for broadcast case) nor the quantized dimension size.
  quant->scale->size = 2;
  tflite::testing::ValidateConvFailsDuringPrepare(
      tensors, tensors_size, &tflite::testing::common_conv_params,
      tflite::Register_CONV_2D(), output_data);

  // Set scale back to correct dimension, and make zero point array too short.
  quant->scale->size = tflite::testing::kFilterShape[0];
  quant->zero_point->size = 2;
  tflite::testing::ValidateConvFailsDuringPrepare(
      tensors, tensors_size, &tflite::testing::common_conv_params,
      tflite::Register_CONV_2D(), output_data);
}

TEST(ConvTest, Int8Input32x1Filter32x32ShouldMatchGolden) {
  constexpr int kSampleSize = 32;
  constexpr int kNumFilters = 32;
  int input_shape[] = {4, 1, 1, 1, kSampleSize};
  int filter_shape[] = {4, kNumFilters, 1, 1, kSampleSize};
  int bias_shape[] = {1, kSampleSize};
  int output_shape[] = {4, 1, 1, 1, kSampleSize};
  float filter_values[kNumFilters * kSampleSize];
  float input_values[kSampleSize];
  float bias_values[kSampleSize];

  // Generated these outputs using the floating point reference conv kernel.
  // TODO(b/149942509): Do this comparison automatically on random inputs.
  float expected_output[kSampleSize] = {
      5168.000000,  3377.000000,  306.000000,   -4045.000000, -4556.000000,
      -1227.000000, 822.000000,   1591.000000,  5176.000000,  3385.000000,
      314.000000,   -4037.000000, -4548.000000, -1219.000000, 830.000000,
      1599.000000,  5184.000000,  3393.000000,  322.000000,   -4029.000000,
      -4540.000000, -1211.000000, 838.000000,   1607.000000,  5192.000000,
      3401.000000,  330.000000,   -4021.000000, -4532.000000, -1203.000000,
      846.000000,   1615.000000};

  for (int i = 0; i < kSampleSize; i++) {
    bias_values[i] = i;
    // Generate inputs from -16 to 15.
    input_values[i] = i - 16;
  }

  // Generate samples of varying values between -128 and 127.
  for (int i = 0; i < kNumFilters * kSampleSize; i++) {
    filter_values[i] = (i * 25) % 256 - 128;
  }

  TfLiteConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_height_factor = 1;
  conv_params.dilation_width_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;
  conv_params.padding = kTfLitePaddingValid;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);
  const int output_dims_count = tflite::ElementCount(*output_dims);

  // Quantization Parameters.  All scales except output are 1.0, and all zero
  // points are 0. This direct-maps the values to floating point and makes it
  // easy to reason about them.
  int input_zero_point = 0;
  float input_scale = 1.0f;
  int filter_zero_point = 0;
  float filter_scale = 1.0f;
  int output_zero_point = 0;
  // Output scale of 50 is needed to accommodate a float range of [-6400, 6350]
  float output_scale = 50.0f;

  // Create per-tensor quantized int8_t input tensor.
  int8_t input_quantized[kSampleSize];
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, input_zero_point);
  // Set zero point and scale arrays with a single element for each.
  int input_zero_points[] = {1, input_zero_point};
  float input_scales[] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-tensor quantized int8_t filter tensor.
  int8_t filter_quantized[kNumFilters * kSampleSize];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale,
      filter_zero_point);
  // Set zero point and scale arrays with a single element for each.
  int filter_zero_points[] = {1, filter_zero_point};
  float filter_scales[] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-tensor quantized int32_t bias tensor.
  int32_t bias_quantized[kSampleSize];
  tflite::SymmetricQuantize(bias_values, bias_quantized, kSampleSize,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  // There is a single zero point of 0, and a single scale of
  // input_scale * filter_scale.
  int bias_zero_points[] = {1, 0};
  float bias_scales[] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-tensor quantized int8_t output tensor.
  int8_t output_quantized[kSampleSize];
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point);
  // Set zero point and scale arrays with a single element for each.
  int output_zero_points[] = {1, output_zero_point};
  float output_scales[] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  int8_t golden_quantized[kSampleSize];
  tflite::Quantize(expected_output, golden_quantized, output_dims_count,
                   output_scale, output_zero_point);

  // Rounding errors due to quantization should not exceed 1.
  constexpr int kQuantizationTolerance = 1;

  tflite::testing::ValidateConvGoldens(
      tensors, kTensorsSize, golden_quantized, output_dims_count, &conv_params,
      tflite::Register_CONV_2D(), output_quantized, kQuantizationTolerance);
}

// This test is created based on
// https://github.com/tensorflow/tflite-micro/issues/329
// Input, output and filter are all 8 bits.
// Filter tensor is of dimension 8x3x3x3 with different scales per output
// channel. Some arbitrary parameters come from the above issue.
TEST(ConvTest, Int8Filter8x3x3x3PerChannelScaleRelu6ShouldMatchGolden) {
  using tflite::ElementCount;
  using tflite::kConvBiasQuantized8;
  using tflite::kConvFilter8x3x3x3;
  using tflite::kConvGoldenOutput1x16x16x8;
  using tflite::kConvInput1x32x32x3;
  using tflite::testing::CreateTensor;
  using tflite::testing::FloatArrayFromFloats;
  using tflite::testing::IntArrayFromInts;
  using tflite::testing::ValidateConvGoldens;

  constexpr int kInDepth = 3;
  constexpr int kOutDepth = 8;

  // Input quantization parameters: same scale and zero point for all input
  // elements.
  constexpr float kInputScale = 0.00784313772f;
  constexpr int kInputZeroPoint = -1;
  float input_scales[] = {1, kInputScale};
  int input_zero_points[] = {1, kInputZeroPoint};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points),
                                          0};
  // Create input tensor of size 1x32x32x3.
  int input_shape[] = {4, 1, 32, 32, kInDepth};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteTensor input_tensor = CreateTensor(kConvInput1x32x32x3, input_dims);
  input_tensor.params = {kInputScale, kInputZeroPoint};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Filter quantization parameters: same zero point, but different scale per
  // output channel.
  int filter_zero_points[kOutDepth + 1] = {kOutDepth, 0, 0, 0, 0, 0, 0, 0, 0};
  float filter_scales[kOutDepth + 1] = {
      kOutDepth,      2.18926089e-05, 0.00453596329,
      0.000504297379, 0.00184638216,  0.00596635276,
      0.000199135626, 0.0047677448,   0.00193942268};
  TfLiteAffineQuantization filter_quant;
  filter_quant.scale = FloatArrayFromFloats(filter_scales);
  filter_quant.zero_point = IntArrayFromInts(filter_zero_points);
  filter_quant.quantized_dimension = 0;

  // Create filter tensor of size 8x3x3x3.
  int filter_shape[] = {4, kOutDepth, 3, 3, kInDepth};
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_shape);
  TfLiteTensor filter_tensor = CreateTensor(kConvFilter8x3x3x3, filter_dims);
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Bias quantization parameters: same zero point, but different scale per
  // output channel.
  int bias_zero_points[kOutDepth + 1] = {kOutDepth, 0, 0, 0, 0, 0, 0, 0, 0};
  float bias_scales[kOutDepth + 1] = {
      kOutDepth,      1.71706745e-07, 3.5576184e-05,
      3.95527377e-06, 1.44814294e-05, 4.67949249e-05,
      1.56184819e-06, 3.73940784e-05, 1.52111588e-05};
  TfLiteAffineQuantization bias_quant;
  bias_quant.scale = FloatArrayFromFloats(bias_scales);
  bias_quant.zero_point = IntArrayFromInts(bias_zero_points);
  bias_quant.quantized_dimension = 0;

  // Create per output channel bias of size 8
  int bias_shape[] = {1, kOutDepth};
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_shape);
  TfLiteTensor bias_tensor = CreateTensor(kConvBiasQuantized8, bias_dims);
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Output quantization parameters: same zero point and scale for all elements.
  const float output_scale = 0.0235294122f;
  const int output_zero_point = -128;
  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {FloatArrayFromFloats(output_scales),
                                           IntArrayFromInts(output_zero_points),
                                           0};

  // Create output tensor of 16x16x8
  int8_t output_data[1 * 16 * 16 * kOutDepth];
  int output_shape[] = {4, 1, 16, 16, kOutDepth};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  const int output_dims_count = ElementCount(*output_dims);
  TfLiteTensor output_tensor = CreateTensor(output_data, output_dims);
  output_tensor.params = {output_scale, output_zero_point};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  TfLiteConvParams conv_params{tflite::testing::common_conv_params};
  conv_params.activation = kTfLiteActRelu6;

  ValidateConvGoldens(tensors, tensors_size, kConvGoldenOutput1x16x16x8,
                      output_dims_count, &conv_params,
                      tflite::Register_CONV_2D(), output_data,
                      1.0 /* tolerance */);
}

// 1x1 kernel, stride 1 (CMSIS-NN direct matmul path).
TEST(ConvTest, Float1x1Stride1ShouldMatchGolden) {
  int input_shape[] = {4, 1, 2, 4, 4};
  int filter_shape[] = {4, 8, 1, 1, 4};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 2, 4, 8};
  static float input_data[1 * 2 * 4 * 4];
  static float filter_data[8 * 1 * 1 * 4];
  static float bias_data[8];
  static float output_data[1 * 2 * 4 * 8];
  static const float kGolden[1 * 2 * 4 * 8] = {
      -0.5,    -0.8125, 0,      -0.3125, -0.625, -0.9375, -1.25,  2.375,
      1.625,   1,       0.9375, 0.3125,  -0.875, -1.5,    -2.125, 0.625,
      1.5,     1.125,   0.75,   0.375,   -1.125, -1.5,    -1.875, 0.5625,
      -0.3125, -1,      2.8125, 2.125,   -0.25,  -0.9375, -1.625, -0.0625,
      -0.4375, -0.875,  2.625,  2.1875,  -0.5,   -0.9375, -1.375, -0.125,
      -1.125,  -1.875,  0.75,   0,       1.5,    0.75,    0,      0.375,
      -1.25,   -1.75,   0.5625, 0.0625,  1.25,   0.75,    0.25,   0.3125,
      -1.375,  -1.625,  0.375,  0.125,   1,      0.75,    0.5,    0.25};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 1x1 kernel, stride 2 (CMSIS-NN 1x1 patch-GEMM path).
TEST(ConvTest, Float1x1Stride2ShouldMatchGolden) {
  int input_shape[] = {4, 1, 6, 8, 16};
  int filter_shape[] = {4, 8, 1, 1, 16};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 3, 4, 8};
  static float input_data[1 * 6 * 8 * 16];
  static float filter_data[8 * 1 * 1 * 16];
  static float bias_data[8];
  static float output_data[1 * 3 * 4 * 8];
  static const float kGolden[1 * 3 * 4 * 8] = {
      2.625,  -4.0625, 8.375, -2.8125, 2.875,  -3.25,  -1.5,   0.25,
      -0.625, -2.875,  3.875, -2.875,  7.25,   -4,     1.625,  -1.75,
      -2.75,  -0.5625, 0.5,   -1.8125, 2.625,  -3.625, 5.875,  -2.625,
      -3.75,  2.875,   -1.75, 0.375,   -0.875, -2.125, 1.125,  -2.375,
      7,      -4.125,  3.875, -1.625,  -0.375, -1.375, -3.5,   3.375,
      2.625,  -4.0625, 8.375, -2.8125, 2.875,  -3.25,  -1.5,   0.25,
      -0.625, -2.875,  3.875, -2.875,  7.25,   -4,     1.625,  -1.75,
      -2.75,  -0.5625, 0.5,   -1.8125, 2.625,  -3.625, 5.875,  -2.625,
      2.375,  -3.0625, 0.5,   0.6875,  -2.5,   1.625,  -4.375, 7.625,
      7,      -4.125,  3.875, -1.625,  -0.375, -1.375, -3.5,   3.375,
      2.625,  -4.0625, 8.375, -2.8125, 2.875,  -3.25,  -1.5,   0.25,
      -0.625, -2.875,  3.875, -2.875,  7.25,   -4,     1.625,  -1.75};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 2, 2, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 1xN kernel, batch 2 (CMSIS-NN generic 1xN path).
TEST(ConvTest, Float1xNValidShouldMatchGolden) {
  int input_shape[] = {4, 2, 1, 10, 4};
  int filter_shape[] = {4, 8, 1, 4, 4};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 2, 1, 7, 8};
  static float input_data[2 * 1 * 10 * 4];
  static float filter_data[8 * 1 * 4 * 4];
  static float bias_data[8];
  static float output_data[2 * 1 * 7 * 8];
  static const float kGolden[2 * 1 * 7 * 8] = {
      2.625,  -4.0625, 8.375,   -2.8125, 2.875,  -3.25,  -1.5,    0.25,
      7,      -4.125,  3.875,   -1.625,  -0.375, -1.375, -3.5,    3.375,
      2.375,  -3.0625, 0.5,     0.6875,  -2.5,   1.625,  -4.375,  7.625,
      -1.125, -0.875,  -1.75,   4.125,   -3.5,   5.75,   -4.125,  2.875,
      -2.375, 0.75,    -2.3125, 6.4375,  -3.375, 3.125,  -3.3125, 0.9375,
      -3.625, 5.1875,  -2.3125, 2,       -2.125, -0.625, -0.8125, -1.5625,
      -3.75,  2.875,   -1.75,   0.375,   -0.875, -2.125, 1.125,   -2.375,
      7,      -4.125,  3.875,   -1.625,  -0.375, -1.375, -3.5,    3.375,
      2.375,  -3.0625, 0.5,     0.6875,  -2.5,   1.625,  -4.375,  7.625,
      -1.125, -0.875,  -1.75,   4.125,   -3.5,   5.75,   -4.125,  2.875,
      -2.375, 0.75,    -2.3125, 6.4375,  -3.375, 3.125,  -3.3125, 0.9375,
      -3.625, 5.1875,  -2.3125, 2,       -2.125, -0.625, -0.8125, -1.5625,
      -3.75,  2.875,   -1.75,   0.375,   -0.875, -2.125, 1.125,   -2.375,
      -2.75,  -0.5625, 0.5,     -1.8125, 2.625,  -3.625, 5.875,   -2.625};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 1xN kernel with SAME padding and stride_w 2 (CMSIS-NN generic 1xN path).
TEST(ConvTest, Float1xNSameStride2ShouldMatchGolden) {
  int input_shape[] = {4, 1, 1, 10, 4};
  int filter_shape[] = {4, 8, 1, 4, 4};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 1, 5, 8};
  static float input_data[1 * 1 * 10 * 4];
  static float filter_data[8 * 1 * 4 * 4];
  static float bias_data[8];
  static float output_data[1 * 1 * 5 * 8];
  static const float kGolden[1 * 1 * 5 * 8] = {
      -0.5,    -2.5,    2.25,    -2,      5.5625, -3.1875, 2.125,   -1.5625,
      7,       -4.125,  3.875,   -1.625,  -0.375, -1.375,  -3.5,    3.375,
      -1.125,  -0.875,  -1.75,   4.125,   -3.5,   5.75,    -4.125,  2.875,
      -3.625,  5.1875,  -2.3125, 2,       -2.125, -0.625,  -0.8125, -1.5625,
      -2.5625, -0.4375, 0.5625,  -0.6875, 0.875,  -2.625,  4,       -1.75};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 2, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 1x3 kernel (CMSIS-NN specialized k3 path).
TEST(ConvTest, Float1x3ShouldMatchGolden) {
  int input_shape[] = {4, 1, 1, 8, 4};
  int filter_shape[] = {4, 4, 1, 3, 4};
  int bias_shape[] = {1, 4};
  int output_shape[] = {4, 1, 1, 6, 4};
  static float input_data[1 * 1 * 8 * 4];
  static float filter_data[4 * 1 * 3 * 4];
  static float bias_data[4];
  static float output_data[1 * 1 * 6 * 4];
  static const float kGolden[1 * 1 * 6 * 4] = {
      1,     -3.4375, 0.5625,  1.75,  5.3125,  -2.5,  -1.3125, 6.0625,
      2.875, -1.5625, -1.5,    3.625, -0.6875, 1.625, -1.125,  0.0625,
      -2,    3.6875,  -0.1875, -1.25, -2.75,   1.25,  1.3125,  -2};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 3x3 kernel, SAME padding, Relu (CMSIS-NN patch-GEMM path).
TEST(ConvTest, Float3x3SameReluShouldMatchGolden) {
  int input_shape[] = {4, 1, 5, 5, 4};
  int filter_shape[] = {4, 8, 3, 3, 4};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 5, 5, 8};
  static float input_data[1 * 5 * 5 * 4];
  static float filter_data[8 * 3 * 3 * 4];
  static float bias_data[8];
  static float output_data[1 * 5 * 5 * 8];
  static const float kGolden[1 * 5 * 5 * 8] = {
      0,      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0.375,  0,
      1.625,  1.125,  0.625,  0.125,  0,      1.375,  3.9375, 3.4375, 5.1875,
      4.6875, 4.1875, 3.6875, 3.1875, 4.9375, 2,      1.5,    3.25,   2.75,
      2.25,   1.75,   1.25,   3,      2.1875, 1.6875, 3.4375, 2.9375, 2.4375,
      1.9375, 1.4375, 3.1875, 0.6875, 0.1875, 1.9375, 1.4375, 0.9375, 0.4375,
      0,      1.6875, 2.9375, 2.4375, 4.1875, 3.6875, 3.1875, 2.6875, 2.1875,
      3.9375, 0,      0,      0.8125, 0.3125, 0,      0,      0,      0.5625,
      0,      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      1.25,   0.75,   2.5,
      2,      1.5,    1,      0.5,    2.25,   0.6875, 0.1875, 1.9375, 1.4375,
      0.9375, 0.4375, 0,      1.6875, 2.375,  1.875,  3.625,  3.125,  2.625,
      2.125,  1.625,  3.375,  2.5625, 2.0625, 3.8125, 3.3125, 2.8125, 2.3125,
      1.8125, 3.5625, 2.9375, 2.4375, 4.1875, 3.6875, 3.1875, 2.6875, 2.1875,
      3.9375, 0,      0,      0.8125, 0.3125, 0,      0,      0,      0.5625,
      0,      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      0,      0,
      0,      0,      0,      0,      0,      0,      0,      1.0625, 0.5625,
      0.0625, 0,      0,      0.8125, 3.9375, 3.4375, 5.1875, 4.6875, 4.1875,
      3.6875, 3.1875, 4.9375, 2,      1.5,    3.25,   2.75,   2.25,   1.75,
      1.25,   3};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 1, 1, kTfLiteActRelu, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 3x3 kernel, stride 2, Relu6, batch 2 (CMSIS-NN patch-GEMM path).
TEST(ConvTest, Float3x3ValidStride2Relu6ShouldMatchGolden) {
  int input_shape[] = {4, 2, 7, 7, 8};
  int filter_shape[] = {4, 8, 3, 3, 8};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 2, 3, 3, 8};
  static float input_data[2 * 7 * 7 * 8];
  static float filter_data[8 * 3 * 3 * 8];
  static float bias_data[8];
  static float output_data[2 * 3 * 3 * 8];
  static const float kGolden[2 * 3 * 3 * 8] = {
      0.5, 0, 1.75, 1.25, 0.75, 0.25, 0,    1.5,  0,    0,    0.0625, 0,
      0,   0, 0,    0,    0,    0,    0,    0,    0,    0,    0,      0,
      6,   6, 6,    6,    6,    6,    6,    6,    0,    0,    0,      0,
      0,   0, 0,    0,    0.5,  0,    1.75, 1.25, 0.75, 0.25, 0,      1.5,
      6,   6, 6,    6,    6,    6,    6,    6,    0,    0,    0,      0,
      0,   0, 0,    0,    6,    6,    6,    6,    6,    6,    6,      6,
      0,   0, 0,    0,    0,    0,    0,    0,    6,    6,    6,      6,
      6,   6, 6,    6,    0,    0,    0,    0,    0,    0,    0,      0,
      0.5, 0, 1.75, 1.25, 0.75, 0.25, 0,    1.5,  0,    0,    0.0625, 0,
      0,   0, 0,    0,    0,    0,    0,    0,    0,    0,    0,      0,
      6,   6, 6,    6,    6,    6,    6,    6,    0,    0,    0,      0,
      0,   0, 0,    0,    0.5,  0,    1.75, 1.25, 0.75, 0.25, 0,      1.5};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 2, 2, kTfLiteActRelu6, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 3x3 kernel with dilation 2 (CMSIS-NN patch-GEMM path).
TEST(ConvTest, Float3x3Dilation2ShouldMatchGolden) {
  int input_shape[] = {4, 1, 7, 7, 2};
  int filter_shape[] = {4, 8, 3, 3, 2};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 3, 3, 8};
  static float input_data[1 * 7 * 7 * 2];
  static float filter_data[8 * 3 * 3 * 2];
  static float bias_data[8];
  static float output_data[1 * 3 * 3 * 8];
  static const float kGolden[1 * 3 * 3 * 8] = {
      1.8125,  1.3125,  3.0625,  2.5625,  2.0625,  1.5625,  1.0625,  2.8125,
      -1.5625, -2.0625, -0.3125, -0.8125, -1.3125, -1.8125, -2.3125, -0.5625,
      -1,      -1.5,    0.25,    -0.25,   -0.75,   -1.25,   -1.75,   0,
      2.375,   1.875,   3.625,   3.125,   2.625,   2.125,   1.625,   3.375,
      -2.125,  -2.625,  -0.875,  -1.375,  -1.875,  -2.375,  -2.875,  -1.125,
      1.8125,  1.3125,  3.0625,  2.5625,  2.0625,  1.5625,  1.0625,  2.8125,
      -2.6875, -3.1875, -1.4375, -1.9375, -2.4375, -2.9375, -3.4375, -1.6875,
      0.125,   -0.375,  1.375,   0.875,   0.375,   -0.125,  -0.625,  1.125,
      2.375,   1.875,   3.625,   3.125,   2.625,   2.125,   1.625,   3.375};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 2, 2, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 3x3 kernel, dilation 2, SAME padding, 4 output channels (CMSIS-NN direct
// loop path).
TEST(ConvTest, Float3x3Dilation2DirectShouldMatchGolden) {
  int input_shape[] = {4, 1, 6, 6, 2};
  int filter_shape[] = {4, 4, 3, 3, 2};
  int bias_shape[] = {1, 4};
  int output_shape[] = {4, 1, 6, 6, 4};
  static float input_data[1 * 6 * 6 * 2];
  static float filter_data[4 * 3 * 3 * 2];
  static float bias_data[4];
  static float output_data[1 * 6 * 6 * 4];
  static const float kGolden[1 * 6 * 6 * 4] = {
      -0.8125, -1.3125, 0.4375,  -0.0625, -1.375,  -1.875,  -0.125,  -0.625,
      -0.9375, -1.4375, 0.3125,  -0.1875, -0.75,   -1.25,   0.5,     0,
      0.75,    0.25,    2,       1.5,     0.0625,  -0.4375, 1.3125,  0.8125,
      0.875,   0.375,   2.125,   1.625,   0.875,   0.375,   2.125,   1.625,
      -2.0625, -2.5625, -0.8125, -1.3125, -1.3125, -1.8125, -0.0625, -0.5625,
      -0.5625, -1.0625, 0.6875,  0.1875,  -0.125,  -0.625,  1.125,   0.625,
      -0.8125, -1.3125, 0.4375,  -0.0625, 0.875,   0.375,   2.125,   1.625,
      1.25,    0.75,    2.5,     2,       -0.4375, -0.9375, 0.8125,  0.3125,
      1.25,    0.75,    2.5,     2,       -1.1875, -1.6875, 0.0625,  -0.4375,
      -0.8125, -1.3125, 0.4375,  -0.0625, -2.5,    -3,      -1.25,   -1.75,
      -0.4375, -0.9375, 0.8125,  0.3125,  -0.4375, -0.9375, 0.8125,  0.3125,
      1.25,    0.75,    2.5,     2,       -1.1875, -1.6875, 0.0625,  -0.4375,
      -0.4375, -0.9375, 0.8125,  0.3125,  0.875,   0.375,   2.125,   1.625,
      -0.9375, -1.4375, 0.3125,  -0.1875, -0.75,   -1.25,   0.5,     0,
      -1.6875, -2.1875, -0.4375, -0.9375, 0.625,   0.125,   1.875,   1.375,
      -0.4375, -0.9375, 0.8125,  0.3125,  -0.25,   -0.75,   1,       0.5,
      1.3125,  0.8125,  2.5625,  2.0625,  -1.3125, -1.8125, -0.0625, -0.5625,
      1.5,     1,       2.75,    2.25,    -0.125,  -0.625,  1.125,   0.625};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 1, 1, kTfLiteActNone, 2, 2, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 3x3 kernel without a bias tensor (CMSIS-NN patch-GEMM path).
TEST(ConvTest, Float3x3NoBiasShouldMatchGolden) {
  int input_shape[] = {4, 1, 5, 5, 4};
  int filter_shape[] = {4, 8, 3, 3, 4};
  int output_shape[] = {4, 1, 5, 5, 8};
  static float input_data[1 * 5 * 5 * 4];
  static float filter_data[8 * 3 * 3 * 4];
  static float output_data[1 * 5 * 5 * 8];
  static const float kGolden[1 * 5 * 5 * 8] = {
      -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375,
      -2.375,  -2.375,  -2.375,  -2.375,  -2.375,  -2.375,  -2.375,  -2.375,
      0.625,   0.625,   0.625,   0.625,   0.625,   0.625,   0.625,   0.625,
      4.1875,  4.1875,  4.1875,  4.1875,  4.1875,  4.1875,  4.1875,  4.1875,
      2.25,    2.25,    2.25,    2.25,    2.25,    2.25,    2.25,    2.25,
      2.4375,  2.4375,  2.4375,  2.4375,  2.4375,  2.4375,  2.4375,  2.4375,
      0.9375,  0.9375,  0.9375,  0.9375,  0.9375,  0.9375,  0.9375,  0.9375,
      3.1875,  3.1875,  3.1875,  3.1875,  3.1875,  3.1875,  3.1875,  3.1875,
      -0.1875, -0.1875, -0.1875, -0.1875, -0.1875, -0.1875, -0.1875, -0.1875,
      -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375,
      -4.125,  -4.125,  -4.125,  -4.125,  -4.125,  -4.125,  -4.125,  -4.125,
      -1.875,  -1.875,  -1.875,  -1.875,  -1.875,  -1.875,  -1.875,  -1.875,
      1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,     1.5,
      0.9375,  0.9375,  0.9375,  0.9375,  0.9375,  0.9375,  0.9375,  0.9375,
      2.625,   2.625,   2.625,   2.625,   2.625,   2.625,   2.625,   2.625,
      2.8125,  2.8125,  2.8125,  2.8125,  2.8125,  2.8125,  2.8125,  2.8125,
      3.1875,  3.1875,  3.1875,  3.1875,  3.1875,  3.1875,  3.1875,  3.1875,
      -0.1875, -0.1875, -0.1875, -0.1875, -0.1875, -0.1875, -0.1875, -0.1875,
      -4.125,  -4.125,  -4.125,  -4.125,  -4.125,  -4.125,  -4.125,  -4.125,
      -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375, -2.4375,
      -3.1875, -3.1875, -3.1875, -3.1875, -3.1875, -3.1875, -3.1875, -3.1875,
      -3.5,    -3.5,    -3.5,    -3.5,    -3.5,    -3.5,    -3.5,    -3.5,
      0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,  0.0625,
      4.1875,  4.1875,  4.1875,  4.1875,  4.1875,  4.1875,  4.1875,  4.1875,
      2.25,    2.25,    2.25,    2.25,    2.25,    2.25,    2.25,    2.25};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(input_shape, input_data, filter_shape,
                                        filter_data, /*bias_shape=*/nullptr,
                                        /*bias_data=*/nullptr, output_shape,
                                        kGolden, output_data, &conv_params);
}

// Grouped convolution (2 groups); not supported by the CMSIS-NN float kernel,
// so it must fall back to the reference kernel.
TEST(ConvTest, FloatGroupedConvFallsBackToReference) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int filter_shape[] = {4, 4, 2, 2, 2};
  int bias_shape[] = {1, 4};
  int output_shape[] = {4, 1, 2, 2, 4};
  static float input_data[1 * 3 * 3 * 4];
  static float filter_data[4 * 2 * 2 * 2];
  static float bias_data[4];
  static float output_data[1 * 2 * 2 * 4];
  static const float kGolden[1 * 2 * 2 * 4] = {
      -1,   -1.8125, -0.5,   2,       1.25, -0.8125, 0.9375, 2.75,
      1.25, -1.0625, -0.125, -0.8125, -1,   0.5,     1.875,  0.5};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 3x3 kernel, SAME padding, inexact data. 42 output positions, large enough to
// span multiple CMSIS-NN patch-GEMM tiles with a remainder; 9 output channels
// and 5 input channels (not multiples of 4).
TEST(ConvTest, FloatPatchGemmMultiTileInexactShouldMatchGolden) {
  int input_shape[] = {4, 1, 6, 7, 5};
  int filter_shape[] = {4, 9, 3, 3, 5};
  int bias_shape[] = {1, 9};
  int output_shape[] = {4, 1, 6, 7, 9};
  static float input_data[1 * 6 * 7 * 5];
  static float filter_data[9 * 3 * 3 * 5];
  static float bias_data[9];
  static float output_data[1 * 6 * 7 * 9];
  static const float kGolden[1 * 6 * 7 * 9] = {
      3.2857146,  -8.938776,    11.693878,   -9.918367,    12.122449,
      -8.081633,  4.571429,     -2.9591837,  -5.7959185,   -9.877551,
      6.959184,   -2.0204082,   -3.959184,   11.469388,    -9.244898,
      9.000001,   -12.183674,   12.163265,   -11.714286,   8.795918,
      -9.183674,  12.265306,    -2.8979592,  -0.22448963,  9.020409,
      -10.367347, 4.979592,     -4.632653,   8.755102,     -7.4285717,
      5.0204086,  -8.346939,    6.918368,    -3.6326532,   0.3673469,
      6.714286,   9.020409,     -6.306123,   0.8979591,    4.346939,
      -7.22449,   9.836735,     -9.714286,   6.877551,     -6.571429,
      10.938776,  -11.510204,   8.285714,    -8.061225,    3.7551022,
      1.0204083,  -5.9387755,   12.44898,    -10,          -3.6734695,
      -2.244898,  9.510204,     -6.4285717,  4.3877554,    -5.44898,
      6.7755103,  -4.9387755,   1.1836736,   -3.244898,    7.244898,
      -6.2040815, 4.755102,     -1.1836737,  6.4897957,    -2.7346938,
      -4.9183674, 1.8163269,    -1.0816326,  2.1632655,    -5.3877554,
      5.3673472,  -8.755102,    6.2244897,   -3.6734695,   1.4489796,
      -0.9387757, -5.693877,    -1.1632651,  1.4897957,    2.7346938,
      -1.1836735, 4.285714,     -6.204082,   0.2040818,    -2.7755103,
      4.714286,   -6.3673472,   0.38775536,  -3.1836734,   -3,
      8.448979,   2.0612245,    3.6530614,   -6.020408,    1.5102042,
      2.510204,   -2.1224487,   -4.4081635,  5.979592,     -2.8775513,
      0.93877584, 3.8163264,    4.346939,    -4.5102043,   2.4693878,
      1.9387755,  5.632653,     -1.4693877,  -4.346939,    3.571429,
      -9.163265,  6.2653065,    -4.7959185,  2.9387758,    -5.755102,
      5.265306,   -0.6122451,   5.7142854,   -3.4489796,   -3.6938775,
      -2.0612245, -7.6938777,   7.857143,    -0.53061247,  7.0408163,
      -4.632653,  -0.34693852,  -4.5102043,  -3.9795918,   5.4693875,
      -2.387755,  -5.163265,    0.51020396,  5.244898,     4.3469386,
      2.5102043,  -4.9591837,   4.9387755,   -5.8163266,   0.51020426,
      -5.6734695, 5.510204,     -3.9591837,  0.65306145,   6.2040815,
      4.244898,   -0.061224584, -5.3061223,  4.3469386,    4.142857,
      -2.1632652, -5.6530614,   3.0612247,   -7.0000005,   5.9387755,
      -3.1836736, 6.469388,     -5.4285717,  0.34693882,   -1.8571428,
      0.63265306, 2.6530612,    -1.4285715,  4.8163266,    -9.122449,
      4.632653,   0.75510174,   6.4081635,   -8.591837,    4.571429,
      -6.2040815, 0.38775513,   4.6326528,   1.367347,     2.3265307,
      -4.591837,  3.510204,     3.632653,    1.8775513,    -3.6326532,
      5.4081635,  -3.3877554,   -0.91836715, 1.0816326,    1.4693877,
      -2.7959182, 1.3877552,    2.2857141,   3.1836734,    0.7959184,
      -1.5918367, -1.6326529,   -3.5510204,  0.06122469,   -2.6326532,
      0.7755104,  -5.2040815,   2.4285715,   0.6734693,    1.7346938,
      0.9183672,  -1.3061223,   0.612245,    -0.7959186,   2.0204082,
      -2.6734695, 4.367347,     -6.8979597,  4.3673472,    -5.9591837,
      2.0204084,  -5.4081635,   3.8571432,   -6.122449,    -0.14285716,
      4.897959,   4.3061223,    2.7755105,   -7.6734695,   9.102041,
      2.653061,   2.4081635,    -8.163265,   9.89796,      -7.2448983,
      4.2448983,  2.122449,     3.7551022,   -1.6530614,   -3.367347,
      2.367347,   6.6938777,    -1.6530613,  -3.8979592,   2.7755103,
      -6.0408163, 1.5714285,    0.26530588,  3.1224492,    -3.8979592,
      0.34693873, 2.7142856,    3.6734693,   -0.061224494, -2.387755,
      -1.8979591, -4.2244897,   3.5918367,   -0.3061227,   4.244898,
      -2.9387755, -1.6734692,   -1.3469388,  -3.367347,    5.408163,
      1.510204,   -3.591837,    -1.5102042,  4.3265305,    3.122449,
      2.3877554,  -5.3877554,   8.897959,    -8.734694,    2.2653065,
      -4.9183674, 9.244898,     -7.102041,   3.7755105,    2.4489796,
      6.755102,   -1.1428572,   -6.693877,   3.714286,     -1.0816326,
      2.1632655,  -5.3877554,   5.3673472,   -8.755102,    6.2244897,
      -3.6734695, 1.4489796,    -0.9387757,  -5.693877,    -1.1632651,
      1.4897957,  2.7346938,    -1.1836735,  4.285714,     -6.204082,
      0.2040818,  -2.7755103,   4.714286,    -6.3673472,   0.38775536,
      -3.1836734, -3,           8.448979,    2.0612245,    3.6530614,
      -6.020408,  3.3265305,    2.7755103,   0.34693897,   -3.9591837,
      3.469388,   -2.2448983,   1.8979595,   0.40816322,   4.5510206,
      -6.5510206, 5.204082,     -7.9183674,  7.591837,     -4.122449,
      2,          2.4897962,    -7.346939,   9.102041,     1.897959,
      5.183674,   -9.367347,    10.816327,   -11.714286,   11.285714,
      -8.428572,  3.3061228,    0.4897959,   0.7142858,    -5,
      3.8367348,  4.2244897,    -5.714286,   8.285715,     -9.632653,
      3.8979595,  -7.9183674,   1.8775514,   -8.142858,    7.1836734,
      -7.5306125, 10.612245,    -1.2857141,  -1.9183674,   7.3061223,
      -8.816327,  -0.7142856,   -2.8367348,  9.591837,     -8.020409,
      7.22449,    -7.571429,    10.489796,   -4.7755103,   -2.2040815,
      -10.346939, 10.918367,    -7.2448983,  -1.4693877,   4.7755103,
      -8.22449,   11.632653,    -12.632653,  13.326531,    -5.6530614,
      5.2040815,  -6.9387755,   7.204082,    -4.9387755,   0.7551022,
      0.81632656, -6.163265,    9.387755};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params, /*inexact_data=*/true);
}

// 2x2 kernel with SAME padding: TF pads 0 rows/cols at the top/left and 1 at
// the bottom/right (CMSIS-NN patch-GEMM path).
TEST(ConvTest, Float2x2SameAsymmetricPaddingShouldMatchGolden) {
  int input_shape[] = {4, 1, 5, 5, 4};
  int filter_shape[] = {4, 8, 2, 2, 4};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 5, 5, 8};
  static float input_data[1 * 5 * 5 * 4];
  static float filter_data[8 * 2 * 2 * 4];
  static float bias_data[8];
  static float output_data[1 * 5 * 5 * 8];
  static const float kGolden[1 * 5 * 5 * 8] = {
      0.5625,  -1.8125, 3.125,   1.3125,  -0.5,    1.0625,  -3,      1.375,
      2.6875,  -1.875,  0.875,   1.9375,  -2.0625, 1.8125,  -3.875,  2.8125,
      -0.25,   0.875,   -0.8125, 0.875,   -2.5,    1.4375,  -3.0625, 3.6875,
      -2.0625, -0.3125, -1.375,  0.9375,  -1.8125, 2.1875,  -1.125,  0.625,
      -1.4375, -1.5,    0.125,   1.75,    -1.125,  0.5,     0.4375,  -0.1875,
      -2.3125, 0.6875,  0.875,   -0.625,  2.9375,  -3.0625, 1.0625,  -2.125,
      -0.75,   0.0625,  3.125,   -0.5625, 0.8125,  -2.875,  -0.375,  -1.25,
      1.9375,  -1.6875, 2,       -1.0625, 0.9375,  -2.6875, 1,       0.1875,
      0.6875,  -2.3125, 2,       -0.4375, 2.1875,  -1.375,  -1.5625, 2.75,
      -0.75,   -0.375,  1.6875,  0.9375,  0.75,    0,       -1.875,  0.1875,
      2.6875,  -1.875,  0.875,   1.9375,  -2.0625, 1.8125,  -3.875,  2.8125,
      -0.25,   0.875,   -0.8125, 0.875,   -2.5,    1.4375,  -3.0625, 3.6875,
      -2.0625, -0.3125, -1.375,  0.9375,  -1.8125, 2.1875,  -1.125,  0.625,
      -2.75,   -0.375,  -0.8125, 2.125,   0,       -1,      1.9375,  -1.3125,
      -1.1875, -0.375,  2.125,   -1,      1.5,     -1.625,  -0.25,   -0.5625,
      -0.75,   0.0625,  3.125,   -0.5625, 0.8125,  -2.875,  -0.375,  -1.25,
      1.9375,  -1.6875, 2,       -1.0625, 0.9375,  -2.6875, 1,       0.1875,
      0.6875,  -2.3125, 2,       -0.4375, 2.1875,  -1.375,  -1.5625, 2.75,
      0.5625,  -1.8125, 3.125,   1.3125,  -0.5,    1.0625,  -3,      1.375,
      1.1875,  -0.375,  -0.25,   1.5625,  -1.125,  0.6875,  -2.5625, 2.0625,
      1.25,    -2.3125, 0.875,   0.125,   -1.1875, 0.875,   -2.6875, 4.4375,
      -0.4375, -1.4375, -0.1875, 1.625,   -1.625,  3,       -2.5,    2.125,
      -1.5625, 0,       -0.6875, 3.6875,  -1.5,    0.625,   -1.75,   0.375,
      -2.125,  2,       -0.625,  1.25,    -0.8125, -1.1875, -0.4375, -0.8125,
      -1.25,   0.5,     0.5625,  0.0625,  -0.4375, -0.9375, 0.25,    -0.25};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 4x4 kernel, SAME, stride 2 on 7x7: TF pads 1 at the top/left and 2 at the
// bottom/right. 7 output channels, 5 input channels (CMSIS-NN direct loop
// path).
TEST(ConvTest, Float4x4SameStride2AsymmetricPaddingShouldMatchGolden) {
  int input_shape[] = {4, 1, 7, 7, 5};
  int filter_shape[] = {4, 7, 4, 4, 5};
  int bias_shape[] = {1, 7};
  int output_shape[] = {4, 1, 4, 4, 7};
  static float input_data[1 * 7 * 7 * 5];
  static float filter_data[7 * 4 * 4 * 5];
  static float bias_data[7];
  static float output_data[1 * 4 * 4 * 7];
  static const float kGolden[1 * 4 * 4 * 7] = {
      0.125,   -3.1875, 3.625,   0.875,   -1.3125, 0.4375,  -1.75,   -1.3125,
      2.6875,  -0.625,  -1.125,  2.875,   -2.125,  -0.375,  1.3125,  -3.5,
      -0.4375, 4.3125,  -1.625,  -1.375,  1.6875,  -1.8125, -0.625,  2.8125,
      -1.0625, -1.5625, 1.875,   -0.875,  -3,      8.3125,  0.5,     -3.9375,
      -1.0625, 2.375,   -0.375,  11.75,   -2.25,   -3.875,  -1,      1.875,
      1.375,   -2.5,    -1.75,   -6.5,    1.125,   2,       4,       -2.4375,
      -7.1875, -3.6875, -0.0625, 2.4375,  2.6875,  -1,      -4.125,  -1.0625,
      -0.375,  -4.625,  0.125,   7.6875,  -2.1875, -4.1875, 0.5625,  -5.6875,
      -3.875,  10.3125, 0.875,   -5.75,   -0.5625, 1.25,    -1.75,   10.4375,
      -0.4375, -5.6875, -1.9375, 3.5,     0.5,     5.75,    -2.125,  -1.5625,
      -0.4375, 1.8125,  -0.4375, -1.5625, 1.5625,  -0.0625, -2.8125, -1.625,
      2.375,   4.125,   -4.25,   -2,      -6.8125, 0.75,    3.25,    6.3125,
      -5.8125, -6.125,  -6.8125, -0.25,   3.5,     6.125,   -4.75,   -5.5,
      5.5625,  -0.125,  -0.4375, 3.1875,  -1.0625, -1.9375, 2.8125,  -0.3125};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 2, 2, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 1x2 kernel with SAME padding: 0 at the left and 1 at the right. 5 output
// channels, 3 input channels (CMSIS-NN generic 1xN path).
TEST(ConvTest, Float1x2SameAsymmetricPaddingShouldMatchGolden) {
  int input_shape[] = {4, 1, 1, 9, 3};
  int filter_shape[] = {4, 5, 1, 2, 3};
  int bias_shape[] = {1, 5};
  int output_shape[] = {4, 1, 1, 9, 5};
  static float input_data[1 * 1 * 9 * 3];
  static float filter_data[5 * 1 * 2 * 3];
  static float bias_data[5];
  static float output_data[1 * 1 * 9 * 5];
  static const float kGolden[1 * 1 * 9 * 5] = {
      0.375,   -1.25,   -0.0625, 1.125,   -0.5,    -0.1875, -1.8125, 2.75,
      0.5625,  -1.0625, -1.875,  -0.125,  1.0625,  -1.125,  0.625,   0.375,
      -1.25,   -0.0625, 1.125,   -0.5,    -0.1875, -1.8125, 2.75,    0.5625,
      -1.0625, -1.875,  -0.125,  1.0625,  -1.125,  0.625,   0.375,   -1.25,
      -0.0625, 1.125,   -0.5,    -0.1875, -1.8125, 2.75,    0.5625,  -1.0625,
      -1.0625, 0.125,   1.3125,  -0.3125, 0.875};
  TfLiteConvParams conv_params = {
      kTfLitePaddingSame, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// 1x1 kernel with 3 output channels and 5 input channels (CMSIS-NN 1x1 path).
TEST(ConvTest, Float1x1OutChannels3ShouldMatchGolden) {
  int input_shape[] = {4, 1, 3, 5, 5};
  int filter_shape[] = {4, 3, 1, 1, 5};
  int bias_shape[] = {1, 3};
  int output_shape[] = {4, 1, 3, 5, 3};
  static float input_data[1 * 3 * 5 * 5];
  static float filter_data[3 * 1 * 1 * 5];
  static float bias_data[3];
  static float output_data[1 * 3 * 5 * 3];
  static const float kGolden[1 * 3 * 5 * 3] = {
      -0.375,  1.3125, 3,       -0.1875, 1.25,    3.25,    -1.125,  -0.5,
      1.25,    -1.5,   -1.6875, -0.1875, -1.3125, -1.75,   0.0625,  -0.5625,
      -1.8125, -0.25,  -0.375,  -1.875,  0,       1.5,     -0.8125, 0.8125,
      1.6875,  -0.875, 1.0625,  -0.375,  1.3125,  3,       -0.1875, 1.25,
      3.25,    -1.125, -0.5,    1.25,    -1.5,    -1.6875, -0.1875, -1.3125,
      -1.75,   0.0625, -0.5625, -1.8125, -0.25};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// stride_h 2, stride_w 1, dilation_h 1, dilation_w 2 (CMSIS-NN patch-GEMM
// path).
TEST(ConvTest, FloatStrideH2W1DilationH1W2ShouldMatchGolden) {
  int input_shape[] = {4, 1, 7, 8, 4};
  int filter_shape[] = {4, 8, 3, 3, 4};
  int bias_shape[] = {1, 8};
  int output_shape[] = {4, 1, 3, 4, 8};
  static float input_data[1 * 7 * 8 * 4];
  static float filter_data[8 * 3 * 3 * 4];
  static float bias_data[8];
  static float output_data[1 * 3 * 4 * 8];
  static const float kGolden[1 * 3 * 4 * 8] = {
      3.875,   3.375,   5.125,  4.625,   4.125,   3.625,   3.125,   4.875,
      -2.875,  -3.375,  -1.625, -2.125,  -2.625,  -3.125,  -3.625,  -1.875,
      0.5,     0,       1.75,   1.25,    0.75,    0.25,    -0.25,   1.5,
      -4,      -4.5,    -2.75,  -3.25,   -3.75,   -4.25,   -4.75,   -3,
      -1.1875, -1.6875, 0.0625, -0.4375, -0.9375, -1.4375, -1.9375, -0.1875,
      3.875,   3.375,   5.125,  4.625,   4.125,   3.625,   3.125,   4.875,
      3.875,   3.375,   5.125,  4.625,   4.125,   3.625,   3.125,   4.875,
      -2.875,  -3.375,  -1.625, -2.125,  -2.625,  -3.125,  -3.625,  -1.875,
      -1.75,   -2.25,   -0.5,   -1,      -1.5,    -2,      -2.5,    -0.75,
      -0.625,  -1.125,  0.625,  0.125,   -0.375,  -0.875,  -1.375,  0.375,
      -1.1875, -1.6875, 0.0625, -0.4375, -0.9375, -1.4375, -1.9375, -0.1875,
      3.875,   3.375,   5.125,  4.625,   4.125,   3.625,   3.125,   4.875};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 1, 2, kTfLiteActNone, 2, 1, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}

// stride_h 1, stride_w 2, dilation_h 2, dilation_w 1 (CMSIS-NN direct loop
// path).
TEST(ConvTest, FloatStrideH1W2DilationH2W1ShouldMatchGolden) {
  int input_shape[] = {4, 1, 7, 8, 2};
  int filter_shape[] = {4, 4, 3, 3, 2};
  int bias_shape[] = {1, 4};
  int output_shape[] = {4, 1, 3, 3, 4};
  static float input_data[1 * 7 * 8 * 2];
  static float filter_data[4 * 3 * 3 * 2];
  static float bias_data[4];
  static float output_data[1 * 3 * 3 * 4];
  static const float kGolden[1 * 3 * 3 * 4] = {
      0.5,    0,     1.75,  1.25,  0.5,   0,    1.75,   1.25,   -0.625,
      -1.125, 0.625, 0.125, -1.75, -2.25, -0.5, -1,     -0.625, -1.125,
      0.625,  0.125, 0.5,   0,     1.75,  1.25, 2.1875, 1.6875, 3.4375,
      2.9375, 0.5,   0,     1.75,  1.25,  0.5,  0,      1.75,   1.25};
  TfLiteConvParams conv_params = {
      kTfLitePaddingValid, 2, 1, kTfLiteActNone, 1, 2, kTfLiteNoType};

  tflite::testing::TestConvFloatPattern(
      input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
      output_shape, kGolden, output_data, &conv_params);
}
TF_LITE_MICRO_TESTS_MAIN