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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SCATTER_ND_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SCATTER_ND_H_

#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy

namespace tflite {

namespace reference_ops {

template <typename IndicesT, typename UpdatesT>
inline void ScatterNd(const RuntimeShape& indices_shape,
                      const IndicesT* indices_data,
                      const RuntimeShape& updates_shape,
                      const UpdatesT* updates_data,
                      const RuntimeShape& output_shape, UpdatesT* output_data) {
  ruy::profiler::ScopeLabel label("ScatterNd");

  int n_slices = 1;
  int slice_size = 1;
  const int outer_dims = indices_shape.DimensionsCount() - 1;
  const int indices_nd = indices_shape.Dims(outer_dims);
  const int updates_dims = updates_shape.DimensionsCount();
  for (int i = 0; i < outer_dims; ++i) {
    n_slices *= indices_shape.Dims(i);
  }
  for (int i = outer_dims; i < updates_dims; ++i) {
    slice_size *= updates_shape.Dims(i);
  }

  int output_flat_size = output_shape.FlatSize();
  int remain_flat_size = output_flat_size;
  std::vector<int> dims_to_count(indices_nd, 0);
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / output_shape.Dims(i);
    remain_flat_size = dims_to_count[i];
  }

  memset(output_data, 0, sizeof(UpdatesT) * output_flat_size);
  for (int i = 0; i < n_slices; ++i) {
    int to_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      IndicesT idx = indices_data[i * indices_nd + j];
      TFLITE_DCHECK(0 <= idx && idx < output_shape.Dims(j));
      to_pos += idx * dims_to_count[j];
    }
    for (int j = 0; j < slice_size; j++) {
      output_data[to_pos + j] += updates_data[i * slice_size + j];
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SCATTER_ND_H_
