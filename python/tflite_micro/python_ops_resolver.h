/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_PYTHON_OPS_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_PYTHON_OPS_RESOLVER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {

// PythonOpsResolver is used to register all the Ops for the TFLM Python
// interpreter. This is ok since code size is not a concern from Python and
// the goal is to be able to run any model supported by TFLM in a flexible way
class PythonOpsResolver : public MicroMutableOpResolver<200> {
 public:
  PythonOpsResolver();

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_PYTHON_OPS_RESOLVER_H_
