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

#ifndef XCORE_ELEMENTWISE_H_
#define XCORE_ELEMENTWISE_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

template <typename TArgs>
struct ElementwiseThreadData {
  int32_t start;
  int32_t element_count;
  const TArgs* args;
};

template <typename TMultiThreadedOpData>
void* ElementwiseInit(TfLiteContext* context, const char* buffer,
                      size_t length) {
  auto job_sizes =
      CustomOptionParser(buffer, length).parseElementwiseJobSizes();
  auto* op_data = construct_persistent_object<TMultiThreadedOpData>(context);

  // in this op we have one job per thread
  auto n_threads = job_sizes.size();
  op_data->threads.allocate(context, n_threads);
  int start_idx = 0;
  for (int j{0}; j < n_threads; j++) {
    auto job_size = job_sizes[j].AsInt32();
    op_data->threads.append({start_idx, job_size, &op_data->args});
    start_idx = start_idx + job_size;
  }

  return op_data;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_ELEMENTWISE_H_
