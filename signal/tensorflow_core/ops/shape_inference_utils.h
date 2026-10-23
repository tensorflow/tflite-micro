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

#include <cstddef>
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

// RAII wrapper for TF_ShapeHandle*.
class ShapeHandle {
 public:
  ShapeHandle() : handle_(TF_NewShapeHandle()) {}
  explicit ShapeHandle(TF_ShapeHandle* handle) : handle_(handle) {}
  ~ShapeHandle() {
    if (handle_ != nullptr) {
      TF_DeleteShapeHandle(handle_);
    }
  }

  ShapeHandle(const ShapeHandle&) = delete;
  ShapeHandle& operator=(const ShapeHandle&) = delete;

  ShapeHandle(ShapeHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  ShapeHandle& operator=(ShapeHandle&& other) noexcept {
    if (this != &other) {
      if (handle_ != nullptr) {
        TF_DeleteShapeHandle(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  TF_ShapeHandle* get() const { return handle_; }

 private:
  TF_ShapeHandle* handle_ = nullptr;
};

// RAII wrapper for TF_DimensionHandle*.
class DimensionHandle {
 public:
  DimensionHandle() : handle_(TF_NewDimensionHandle()) {}
  ~DimensionHandle() {
    if (handle_ != nullptr) {
      TF_DeleteDimensionHandle(handle_);
    }
  }

  DimensionHandle(const DimensionHandle&) = delete;
  DimensionHandle& operator=(const DimensionHandle&) = delete;

  DimensionHandle(DimensionHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  DimensionHandle& operator=(DimensionHandle&& other) noexcept {
    if (this != &other) {
      if (handle_ != nullptr) {
        TF_DeleteDimensionHandle(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  TF_DimensionHandle* get() const { return handle_; }

 private:
  TF_DimensionHandle* handle_ = nullptr;
};

// C++ wrapper around TF_ShapeInferenceContext* and TF_Status*.
// Methods become no-ops once `status` holds an error, avoiding repeated checks.
class ShapeInferenceContext {
 public:
  ShapeInferenceContext(TF_ShapeInferenceContext* ctx, TF_Status* status)
      : ctx_(ctx), status_(status) {}

  bool ok() const { return TF_GetCode(status_) == TF_OK; }

  bool GetAttr(const char* name, int* value) {
    if (!ok()) {
      return false;
    }
    return GetAttrInt(ctx_, name, value, status_);
  }

  ShapeHandle Input(int index);
  ShapeHandle WithRank(const ShapeHandle& shape, int64_t rank);
  ShapeHandle WithRankAtLeast(const ShapeHandle& shape, int64_t rank);
  DimensionHandle Dim(const ShapeHandle& shape, int64_t index);
  bool ValueKnown(const DimensionHandle& dim) const;
  int64_t Value(const DimensionHandle& dim) const;
  bool CheckDim(const ShapeHandle& shape, int64_t dim_index, int64_t expected);
  ShapeHandle Subshape(const ShapeHandle& shape, int64_t start, int64_t end);
  ShapeHandle Scalar() {
    return ShapeHandle(TF_ShapeInferenceContextScalar(ctx_));
  }
  ShapeHandle VectorFromSize(int64_t size) {
    return ShapeHandle(TF_ShapeInferenceContextVectorFromSize(
        ctx_, static_cast<size_t>(size)));
  }
  ShapeHandle Concatenate(const ShapeHandle& first, const ShapeHandle& second);
  ShapeHandle ReplaceLastDim(const ShapeHandle& shape, int64_t new_dim);
  void SetOutput(int index, const ShapeHandle& shape);

 private:
  TF_ShapeInferenceContext* ctx_;
  TF_Status* status_;
};

// Fluent C++ wrapper around TF_OpDefinitionBuilder* (tensorflow/c/ops.h).
// Unlike internal tensorflow::OpDefBuilder, the stable TF C API does not
// expose .Doc(); user-facing op docstrings live on the public Python
// wrapper functions in python/tflite_micro/signal/ops/.
class SignalOpDefBuilder {
 public:
  explicit SignalOpDefBuilder(const char* op_name)
      : builder_(TF_NewOpDefinitionBuilder(op_name)) {}

  ~SignalOpDefBuilder() {
    if (builder_ != nullptr) {
      TF_DeleteOpDefinitionBuilder(builder_);
    }
  }

  SignalOpDefBuilder(const SignalOpDefBuilder&) = delete;
  SignalOpDefBuilder& operator=(const SignalOpDefBuilder&) = delete;

  SignalOpDefBuilder& Attr(const char* spec) {
    TF_OpDefinitionBuilderAddAttr(builder_, spec);
    return *this;
  }

  SignalOpDefBuilder& Input(const char* spec) {
    TF_OpDefinitionBuilderAddInput(builder_, spec);
    return *this;
  }

  SignalOpDefBuilder& Output(const char* spec) {
    TF_OpDefinitionBuilderAddOutput(builder_, spec);
    return *this;
  }

  SignalOpDefBuilder& SetShapeFn(void (*fn)(TF_ShapeInferenceContext*,
                                            TF_Status*)) {
    TF_OpDefinitionBuilderSetShapeInferenceFunction(builder_, fn);
    return *this;
  }

  void Register() {
    if (builder_ == nullptr) {
      return;
    }
    TF_Status* status = TF_NewStatus();
    TF_RegisterOpDefinition(builder_, status);
    builder_ = nullptr;
    TF_DeleteStatus(status);
  }

 private:
  TF_OpDefinitionBuilder* builder_ = nullptr;
};

}  // namespace signal
}  // namespace tensorflow

#endif  // SIGNAL_TENSORFLOW_CORE_OPS_SHAPE_INFERENCE_UTILS_H_
