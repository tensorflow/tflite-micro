#ifndef THIRD_PARTY_TFLITE_MICRO_TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_
#define THIRD_PARTY_TFLITE_MICRO_TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_

#include "tensorflow/lite/c/common.h"

// TFLMRegistration defines the API that TFLM kernels need to implement.
// This will be replacing the current TfLiteRegistration_V1 struct with
// something more compatible Embedded enviroment TFLM is used in. 
struct TFLMRegistration {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  void (*reset)(TfLiteContext* context, void* buffer);
  int32_t builtin_code;
  const char* custom_name;
};

#endif  // THIRD_PARTY_TFLITE_MICRO_TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_
