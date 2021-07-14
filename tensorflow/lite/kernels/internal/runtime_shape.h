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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_RUNTIME_SHAPE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_RUNTIME_SHAPE_H_

class RuntimeShape {
 public:
  RuntimeShape& operator=(RuntimeShape const&) = delete;

  RuntimeShape() : size_(0) {}

  explicit RuntimeShape(int dimensions_count) : size_(dimensions_count) {}

  RuntimeShape(int shape_size, int32_t value) : size_(shape_size) {
    for (int i = 0; i < shape_size; ++i) {
      SetDim(i, value);
    }
  }

  RuntimeShape(int dimensions_count, const int32_t* dims_data) : size_(dimensions_count) {
    ReplaceWith(dimensions_count, dims_data);
  }

  bool operator==(const RuntimeShape& comp) const {
    return this->size_ == comp.size_ &&
           std::memcmp(DimsData(), comp.DimsData(), size_ * sizeof(int32_t)) ==
               0;
  }

  ~RuntimeShape() {}

  int32_t DimensionsCount() const { return size_; }
  int32_t Dims(int i) const {
    TFLITE_DCHECK_GE(i, 0);
    TFLITE_DCHECK_LT(i, size_);
    return dims_[i];
  }
  void SetDim(int i, int32_t val) {
    TFLITE_DCHECK_GE(i, 0);
    TFLITE_DCHECK_LT(i, size_);
    dims_[i] = val;
  }

  static RuntimeShape ExtendedShape(int new_shape_size,
                                           const RuntimeShape& shape) {
    return RuntimeShape(new_shape_size, shape, 1);
  }
  int32_t* DimsData() {
    return dims_;
  }
  const int32_t* DimsData() const {
    return dims_;
  }
  const int32_t* DimsDataUpTo5D() const {
    return dims_;
  }


  void ReplaceWith(int dimensions_count, const int32_t* dims_data) {
    int32_t* dst_dims = DimsData();
    std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32_t));
  }

  // Returns the total count of elements, that is the size when flattened into a
  // vector.
  int FlatSize() const {
    int buffer_size = 1;
    const int* dims_data = reinterpret_cast<const int*>(DimsData());
    for (int i = 0; i < size_; i++) {
      buffer_size *= dims_data[i];
    }
    return buffer_size;
  }

 private:
  // For use only by ExtendedShape(), written to guarantee (return-value) copy
  // elision in C++17.
  // This creates a shape padded to the desired size with the specified value.
  RuntimeShape(int new_shape_size, const RuntimeShape& shape, int pad_value)
      : size_(new_shape_size) {
    // If the following check fails, it is likely because a 4D-only kernel is
    // being used with an array of larger dimension count.
    TFLITE_CHECK_GE(new_shape_size, shape.DimensionsCount());
    const int size_increase = new_shape_size - shape.DimensionsCount();
    for (int i = 0; i < size_increase; ++i) {
      SetDim(i, pad_value);
    }
    std::memcpy(DimsData() + size_increase, shape.DimsData(),
                sizeof(int32_t) * shape.DimensionsCount());
  }

  // A maximum of 4 dimensions are supported on TFLM.
  static constexpr int kMaxSize = 4;
  int32_t size_;
  union {
    int32_t dims_[kMaxSize];
  };
};

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_RUNTIME_SHAPE_H_
