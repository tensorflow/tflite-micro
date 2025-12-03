#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ONE_HOT_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ONE_HOT_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace ops {
namespace micro {

// ONE_HOT 커널 등록 함수 (all_ops_resolver 등에서 사용)
const TFLMRegistration* Register_ONE_HOT();

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ONE_HOT_H_
