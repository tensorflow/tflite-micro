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

#ifndef TENSROFLOW_LITE_MICRO_KERNELS_ELEMENTWISE_H_
#define TENSROFLOW_LITE_MICRO_KERNELS_ELEMENTWISE_H_

#include "tensorflow/lite/c/common.h"

namespace tflite {

bool IsNumericSupportedType(const TfLiteType type);

bool IsLogicalSupportedType(const TfLiteType type);

TfLiteStatus GenericPrepareLogical(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus GenericPrepareNumeric(TfLiteContext* context, TfLiteNode* node);

template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                             T func(T), TfLiteType expected_type);

TfLiteStatus EvalNumeric(TfLiteContext* context, TfLiteNode* node,
                                float float_func(float));

TfLiteStatus EvalLogical(TfLiteContext* context, TfLiteNode* node,
                                bool bool_func(bool));

}  // namespace tflite

#endif // TENSROFLOW_LITE_MICRO_KERNELS_ELEMENTWISE_H_
