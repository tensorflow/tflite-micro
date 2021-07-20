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

#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

//*****************************
//*****************************
//*****************************
// ExecutionPlan
//*****************************
//*****************************
//*****************************
ExecutionPlan::ExecutionPlan()
    : n_threads_(0), bias_scratch_offset_(0), bias_scratch_size_(0) {}

void ExecutionPlan::SetWeightsScratchSize(size_t size) {
  // NOTE: Weights assumes to start at scratch offset 0
  //        so we do not need to store it
  bias_scratch_offset_ = size;
}
size_t ExecutionPlan::GetWeightsScratchSize() { return bias_scratch_offset_; }

void ExecutionPlan::SetBiasScratchSize(size_t size) {
  bias_scratch_size_ = size;
}
size_t ExecutionPlan::GetBiasScratchSize() { return bias_scratch_size_; }

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
