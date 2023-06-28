/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
/*
 * * Copyright (c) 2022 Cadence Design Systems Inc.
 * *
 * * Permission is hereby granted, free of charge, to any person obtaining
 * * a copy of this software and associated documentation files (the
 * * "Software"), to deal in the Software without restriction, including
 * * without limitation the rights to use, copy, modify, merge, publish,
 * * distribute, sublicense, and/or sell copies of the Software, and to
 * * permit persons to whom the Software is furnished to do so, subject to
 * * the following conditions:
 * *
 * * The above copyright notice and this permission notice shall be included
 * * in all copies or substantial portions of the Software.
 * *
 * * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "third_party/xtensa/examples/pytorch_to_tflite/mobilenet_v2_quantized_1x3x224x224_model_data.h"
#include "third_party/xtensa/examples/pytorch_to_tflite/pytorch_images_dog_jpg.h"
#include "third_party/xtensa/examples/pytorch_to_tflite/pytorch_op_resolver.h"

TF_LITE_MICRO_TESTS_BEGIN

#if defined(HIFI5)

TF_LITE_MICRO_TEST(TestInvoke) {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_mobilenet_v2_quantized_1x3x224x224_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the eperation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  tflite::PytorchOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  constexpr int tensor_arena_size = 3 * 1024 * 1024;

  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT(input != nullptr);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(224, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(224, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);
  // Make sure the input has the properties we expect.
  MicroPrintf("input->dims->size:%d", input->dims->size);
  MicroPrintf("input->dims->data[0]:%d", input->dims->data[0]);
  MicroPrintf("input->dims->data[1]:%d", input->dims->data[1]);
  MicroPrintf("input->dims->data[2]:%d", input->dims->data[2]);
  MicroPrintf("input->dims->data[3]:%d", input->dims->data[3]);
  MicroPrintf("input->type:%d", input->type);
  MicroPrintf("input->bytes:%d", input->bytes);

  // Copy an image with a person into the memory area used for the input.
  TFLITE_DCHECK_EQ(input->bytes,
                   static_cast<size_t>(g_pytorch_images_dog_jpg_len));
  memcpy(input->data.int8, g_pytorch_images_dog_jpg_data, input->bytes);

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1000, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output->type);
  MicroPrintf("output->dims->size:%d", output->dims->size);
  MicroPrintf("output->dims->data[0]:%d", output->dims->data[0]);
  MicroPrintf("output->dims->data[1]:%d", output->dims->data[1]);
  MicroPrintf("output->type:%d", output->type);

  int label_index = 0;
  int label_confidence = -257;
  for (int i = 0; i < (output->dims->data[0] * output->dims->data[1]); i++) {
    if (label_confidence < output->data.int8[i]) {
      label_index = i;
      label_confidence = output->data.int8[i];
    }
  }
  MicroPrintf("Label index:%d\t", label_index);
  MicroPrintf("Confidence:%d", label_confidence);

  MicroPrintf("Ran successfully\n");
}
#endif  // defined(HIFI5)

TF_LITE_MICRO_TESTS_END
