#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace ops {
namespace micro {

const TFLMRegistration* Register_ONE_HOT();
}  // namespace micro
}  // namespace ops
}  // namespace tflite

namespace tflite {
namespace testing {
namespace {

// Helper function for OneHot operation test
template <typename T>
void TestOneHot(const int* indices_dims, const int32_t* indices_data,
                const int* depth_dims, const int32_t* depth_data,
                const int* on_dims, const T* on_data, const int* off_dims,
                const T* off_data, const int* output_dims,
                const T* expected_output_data, T* output_data, int axis = -1) {
  // 1. Tensor Setting
  TfLiteIntArray* in_dims = IntArrayFromInts(indices_dims);
  TfLiteIntArray* d_dims = IntArrayFromInts(depth_dims);
  TfLiteIntArray* on_val_dims = IntArrayFromInts(on_dims);
  TfLiteIntArray* off_val_dims = IntArrayFromInts(off_dims);
  TfLiteIntArray* out_dims = IntArrayFromInts(output_dims);

  const int output_dims_count = ElementCount(*out_dims);

  // 2. Create Input Tensor
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(indices_data, in_dims), CreateTensor(depth_data, d_dims),
      CreateTensor(on_data, on_val_dims),  CreateTensor(off_data, off_val_dims),
      CreateTensor(output_data, out_dims),  // Output Tensor
  };

  // 3. Parameter setting
  TfLiteOneHotParams builtin_data = {axis};

  // 4. KernelRunner execution
  int inputs_array_data[] = {4, 0, 1, 2, 3};  // indices, depth, on, off
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};  // output tensor index
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  // tflite::ops::micro::Register_ONE_HOT)
  const TFLMRegistration registration = *tflite::ops::micro::Register_ONE_HOT();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // 5. Result evaluation
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OneHot_BasicInt32) {
  // Indices: [0, 1, 2]
  const int indices_dims[] = {1, 3};
  const int32_t indices_data[] = {0, 1, 2};

  // Depth: 3
  const int depth_dims[] = {1, 1};
  const int32_t depth_data[] = {3};

  // On: 1, Off: 0
  const int on_dims[] = {1, 1};
  const int32_t on_data[] = {1};
  const int off_dims[] = {1, 1};
  const int32_t off_data[] = {0};

  // Output: [3, 3] -> Identity Matrix
  const int output_dims[] = {2, 3, 3};
  const int32_t expected_output[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  int32_t output_data[9];

  tflite::testing::TestOneHot(indices_dims, indices_data, depth_dims,
                              depth_data, on_dims, on_data, off_dims, off_data,
                              output_dims, expected_output, output_data);
}

TF_LITE_MICRO_TESTS_END