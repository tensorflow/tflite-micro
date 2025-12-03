#include "tensorflow/lite/micro/kernels/one_hot.h"

#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace {

using tflite::micro::KernelRunner;

// dims 배열 → TfLiteIntArray 로 캐스팅
TfLiteIntArray* IntArrayFromInts(const int* dims) {
  return const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(dims));
}

// int32 Tensor 생성 헬퍼
TfLiteTensor CreateInt32Tensor(int32_t* data, TfLiteIntArray* dims) {
  TfLiteTensor t;
  memset(&t, 0, sizeof(TfLiteTensor));
  t.type = kTfLiteInt32;
  t.dims = dims;
  t.data.i32 = data;
  t.allocation_type = kTfLiteMemNone;
  return t;
}

}  // namespace

// ★ 여기서 main 이 자동으로 정의됩니다.
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OneHot_BasicInt32) {
  // indices: [0,1,2], shape [3]
  int indices_shape_arr[] = {1, 3};
  TfLiteIntArray* indices_shape = IntArrayFromInts(indices_shape_arr);
  int32_t indices_data[3] = {0, 1, 2};

  // depth: scalar (3)
  int depth_shape_arr[] = {0};
  TfLiteIntArray* depth_shape = IntArrayFromInts(depth_shape_arr);
  int32_t depth_data[1] = {3};

  // on/off: scalar
  TfLiteIntArray* scalar_shape = depth_shape;
  int32_t on_value_data[1] = {1};
  int32_t off_value_data[1] = {0};

  // output: [3,3]
  int output_shape_arr[] = {2, 3, 3};
  TfLiteIntArray* output_shape = IntArrayFromInts(output_shape_arr);
  int32_t output_data[9] = {0};

  TfLiteTensor tensors[5];
  tensors[0] = CreateInt32Tensor(indices_data, indices_shape);
  tensors[1] = CreateInt32Tensor(depth_data, depth_shape);
  tensors[2] = CreateInt32Tensor(on_value_data, scalar_shape);
  tensors[3] = CreateInt32Tensor(off_value_data, scalar_shape);
  tensors[4] = CreateInt32Tensor(output_data, output_shape);

  int inputs_arr[] = {4, 0, 1, 2, 3};
  int outputs_arr[] = {1, 4};
  TfLiteIntArray* inputs = IntArrayFromInts(inputs_arr);
  TfLiteIntArray* outputs = IntArrayFromInts(outputs_arr);

  const TFLMRegistration* registration = tflite::ops::micro::Register_ONE_HOT();

  KernelRunner runner(*registration, tensors, 5, inputs, outputs,
                      /*builtin_data=*/nullptr,
                      /*error_reporter=*/nullptr);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  int32_t expected[9] = {
      1, 0, 0, 0, 1, 0, 0, 0, 1,
  };
  for (int i = 0; i < 9; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected[i], output_data[i]);
  }
}

TF_LITE_MICRO_TESTS_END  // ★ 여기까지

}  // namespace tflite
