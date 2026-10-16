/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SIGNAL_TENSORFLOW_CORE_OPS_SHAPE_INFERENCE_UTILS_H_
#define SIGNAL_TENSORFLOW_CORE_OPS_SHAPE_INFERENCE_UTILS_H_

#include <cstdint>

#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {

// Reads an integer attribute from TF_ShapeInferenceContext.
// Sets `status` and returns false if the attribute is missing or invalid.
bool GetAttrInt(TF_ShapeInferenceContext* ctx, const char* attr_name,
                int* value, TF_Status* status);

// Checks that the `dim_index`-th dimension of `shape` equals `expected`
// whenever that dimension's value is statically known.
bool CheckDimValue(TF_ShapeInferenceContext* ctx, TF_ShapeHandle* shape,
                   int64_t dim_index, int64_t expected, TF_Status* status);

// Replaces the last dimension of `shape` (rank >= 1) with `new_dim` and
// stores the resulting shape in `result`.
bool ReplaceLastDim(TF_ShapeInferenceContext* ctx, TF_ShapeHandle* shape,
                    int64_t new_dim, TF_ShapeHandle* result, TF_Status* status);

}  // namespace signal
}  // namespace tensorflow

#endif  // SIGNAL_TENSORFLOW_CORE_OPS_SHAPE_INFERENCE_UTILS_H_
