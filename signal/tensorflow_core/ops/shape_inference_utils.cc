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

#include "signal/tensorflow_core/ops/shape_inference_utils.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef TFLM_USE_TF_SHAPE_INFERENCE_HEADER
#include "tensorflow/core/framework/shape_inference.h"
#else
namespace tensorflow {
namespace shape_inference {
// TF_ShapeInferenceContext in tensorflow/c/ops.cc is a reinterpret_cast of
// tensorflow::shape_inference::InferenceContext. Because tensorflow/c/ops.h
// only exposes TF_ShapeInferenceContext_GetAttrType, we read integer
// attributes via InferenceContext::DebugString(), which is exported by
// libtensorflow_framework.so and returns only a standard std::string.
class InferenceContext {
 public:
  std::string DebugString() const;
};
}  // namespace shape_inference
}  // namespace tensorflow
#endif

namespace tensorflow {
namespace signal {

bool GetAttrInt(TF_ShapeInferenceContext* ctx, const char* attr_name,
                int* value, TF_Status* status) {
  if (ctx == nullptr || attr_name == nullptr || value == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Null argument passed to GetAttrInt");
    return false;
  }

  const auto* cc_ctx =
      reinterpret_cast<const ::tensorflow::shape_inference::InferenceContext*>(
          ctx);
  const std::string debug_str = cc_ctx->DebugString();
  const size_t open_bracket = debug_str.find('[');
  if (open_bracket == std::string::npos) {
    char msg[128];
    std::snprintf(msg, sizeof(msg), "Attribute '%s' not found in shape context",
                  attr_name);
    TF_SetStatus(status, TF_INVALID_ARGUMENT, msg);
    return false;
  }

  const size_t name_len = std::strlen(attr_name);
  const char* p = debug_str.c_str() + open_bracket + 1;
  int bracket_depth = 0;
  bool in_quotes = false;
  bool at_attr_start = true;

  while (*p != '\0') {
    if (!in_quotes && bracket_depth == 0 && at_attr_start) {
      if (std::strncmp(p, attr_name, name_len) == 0 && p[name_len] == '=') {
        const char* val_start = p + name_len + 1;
        char* val_end = nullptr;
        const int64_t parsed = std::strtoll(val_start, &val_end, 10);
        if (val_end != val_start && (*val_end == ',' || *val_end == ']')) {
          *value = static_cast<int>(parsed);
          TF_SetStatus(status, TF_OK, "");
          return true;
        }
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                      "Attribute '%s' is not an integer in shape context",
                      attr_name);
        TF_SetStatus(status, TF_INVALID_ARGUMENT, msg);
        return false;
      }
      at_attr_start = false;
    }

    const char c = *p++;
    if (in_quotes) {
      if (c == '\\' && *p != '\0') {
        ++p;
      } else if (c == '"') {
        in_quotes = false;
      }
    } else if (c == '"') {
      in_quotes = true;
    } else if (c == '[') {
      ++bracket_depth;
    } else if (c == ']') {
      if (bracket_depth == 0) {
        break;
      }
      --bracket_depth;
    } else if (c == ',' && bracket_depth == 0) {
      while (*p == ' ') {
        ++p;
      }
      at_attr_start = true;
    }
  }

  char msg[128];
  std::snprintf(msg, sizeof(msg), "Attribute '%s' not found in shape context",
                attr_name);
  TF_SetStatus(status, TF_INVALID_ARGUMENT, msg);
  return false;
}

bool CheckDimValue(TF_ShapeInferenceContext* ctx, TF_ShapeHandle* shape,
                   int64_t dim_index, int64_t expected, TF_Status* status) {
  DimensionHandle dim;
  TF_ShapeInferenceContextDim(ctx, shape, dim_index, dim.get());
  if (TF_DimensionHandleValueKnown(dim.get())) {
    const int64_t actual = TF_DimensionHandleValue(dim.get());
    if (actual != expected) {
      char msg[128];
      std::snprintf(msg, sizeof(msg),
                    "Dimension must be %" PRId64 " but is %" PRId64, expected,
                    actual);
      TF_SetStatus(status, TF_INVALID_ARGUMENT, msg);
    }
  }
  return TF_GetCode(status) == TF_OK;
}

bool ReplaceLastDim(TF_ShapeInferenceContext* ctx, TF_ShapeHandle* shape,
                    int64_t new_dim, TF_ShapeHandle* result,
                    TF_Status* status) {
  ShapeHandle prefix;
  TF_ShapeInferenceContextSubshape(ctx, shape, 0, -1, prefix.get(), status);
  if (TF_GetCode(status) == TF_OK) {
    ShapeHandle suffix(TF_ShapeInferenceContextVectorFromSize(
        ctx, static_cast<size_t>(new_dim)));
    TF_ShapeInferenceContextConcatenateShapes(ctx, prefix.get(), suffix.get(),
                                              result, status);
  }
  return TF_GetCode(status) == TF_OK;
}

ShapeHandle ShapeInferenceContext::Input(int index) {
  ShapeHandle out;
  if (ok()) {
    TF_ShapeInferenceContextGetInput(ctx_, index, out.get(), status_);
  }
  return out;
}

ShapeHandle ShapeInferenceContext::WithRank(const ShapeHandle& shape,
                                            int64_t rank) {
  ShapeHandle out;
  if (ok()) {
    TF_ShapeInferenceContextWithRank(ctx_, shape.get(), rank, out.get(),
                                     status_);
  }
  return out;
}

ShapeHandle ShapeInferenceContext::WithRankAtLeast(const ShapeHandle& shape,
                                                   int64_t rank) {
  ShapeHandle out;
  if (ok()) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx_, shape.get(), rank, out.get(),
                                            status_);
  }
  return out;
}

DimensionHandle ShapeInferenceContext::Dim(const ShapeHandle& shape,
                                           int64_t index) {
  DimensionHandle dim;
  if (ok()) {
    TF_ShapeInferenceContextDim(ctx_, shape.get(), index, dim.get());
  }
  return dim;
}

bool ShapeInferenceContext::ValueKnown(const DimensionHandle& dim) const {
  return ok() && TF_DimensionHandleValueKnown(dim.get()) != 0;
}

int64_t ShapeInferenceContext::Value(const DimensionHandle& dim) const {
  return TF_DimensionHandleValue(dim.get());
}

bool ShapeInferenceContext::CheckDim(const ShapeHandle& shape,
                                     int64_t dim_index, int64_t expected) {
  if (!ok()) {
    return false;
  }
  return CheckDimValue(ctx_, shape.get(), dim_index, expected, status_);
}

ShapeHandle ShapeInferenceContext::Subshape(const ShapeHandle& shape,
                                            int64_t start, int64_t end) {
  ShapeHandle out;
  if (ok()) {
    TF_ShapeInferenceContextSubshape(ctx_, shape.get(), start, end, out.get(),
                                     status_);
  }
  return out;
}

ShapeHandle ShapeInferenceContext::Concatenate(const ShapeHandle& first,
                                               const ShapeHandle& second) {
  ShapeHandle out;
  if (ok()) {
    TF_ShapeInferenceContextConcatenateShapes(ctx_, first.get(), second.get(),
                                              out.get(), status_);
  }
  return out;
}

ShapeHandle ShapeInferenceContext::ReplaceLastDim(const ShapeHandle& shape,
                                                  int64_t new_dim) {
  ShapeHandle out;
  if (ok()) {
    ::tensorflow::signal::ReplaceLastDim(ctx_, shape.get(), new_dim, out.get(),
                                         status_);
  }
  return out;
}

void ShapeInferenceContext::SetOutput(int index, const ShapeHandle& shape) {
  if (ok()) {
    TF_ShapeInferenceContextSetOutput(ctx_, index, shape.get(), status_);
  }
}

}  // namespace signal
}  // namespace tensorflow
