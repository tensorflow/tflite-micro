#include "tensorflow/lite/micro/kernels/one_hot.h"

#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/c/builtin_op_data.h"  // ★ 이거 추가
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

// 헬퍼들은 익명 namespace (전역) 안에만 둡니다.
namespace {

using tflite::micro::KernelRunner;

// dims 배열 → TfLiteIntArray로 캐스팅
TfLiteIntArray* IntArrayFromInts(const int* dims) {
  return const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(dims));
}

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

// ★★★ 여기서부터는 절대 namespace 안에 넣지 마세요 ★★★
TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(OneHot_BasicInt32) {
  int indices_shape_arr[] = {1, 3};  // rank=1, dim=3
  TfLiteIntArray* indices_shape = IntArrayFromInts(indices_shape_arr);
  int32_t indices_data[3] = {0, 1, 2};

  int depth_shape_arr[] = {0};  // scalar
  TfLiteIntArray* depth_shape = IntArrayFromInts(depth_shape_arr);
  int32_t depth_data[1] = {3};

  TfLiteIntArray* scalar_shape = depth_shape;
  int32_t on_value_data[1] = {1};
  int32_t off_value_data[1] = {0};

  int output_shape_arr[] = {2, 3, 3};  // [3,3]
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

  TfLiteOneHotParams params;
  memset(&params, 0, sizeof(params));
  params.axis = -1;  // 마지막 축 기준 one-hot

  KernelRunner runner(*registration, tensors, 5, inputs, outputs,
                      /*builtin_data=*/&params,
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

TF_LITE_MICRO_TESTS_END
