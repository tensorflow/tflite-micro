/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <memory>
#include <random>
#include <utility>

#include "flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/util.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/tools/benchmarking/op_resolver.h"
#include "tensorflow/lite/micro/tools/layer_by_layer_schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {

// Seed used for the random input. Input data shouldn't affect invocation timing
// so randomness isn't really needed.
constexpr uint32_t kRandomSeed = 0xFB;

constexpr size_t kTensorArenaSize = 3e6;
constexpr int kNumResourceVariable = 100;

bool SaveFile(const char* name, const char* buf, size_t len) {
  std::ofstream ofs(name, std::ofstream::binary);
  if (!ofs.is_open()) return false;
  ofs.write(buf, len);
  return !ofs.bad();
}

TfLiteStatus ConvertTensorType(TfLiteType type, TensorTypes& tensor_type) {
  switch (type) {
    case kTfLiteFloat16:
      tensor_type = TensorTypes_FLOAT16;
      return kTfLiteOk;
    case kTfLiteBFloat16:
      tensor_type = TensorTypes_BFLOAT16;
      return kTfLiteOk;
    case kTfLiteFloat32:
      tensor_type = TensorTypes_FLOAT32;
      return kTfLiteOk;
    case kTfLiteFloat64:
      tensor_type = TensorTypes_FLOAT64;
      return kTfLiteOk;
    case kTfLiteInt16:
      tensor_type = TensorTypes_INT16;
      return kTfLiteOk;
    case kTfLiteUInt16:
      tensor_type = TensorTypes_UINT16;
      return kTfLiteOk;
    case kTfLiteInt32:
      tensor_type = TensorTypes_INT32;
      return kTfLiteOk;
    case kTfLiteUInt32:
      tensor_type = TensorTypes_UINT32;
      return kTfLiteOk;
    case kTfLiteUInt8:
      tensor_type = TensorTypes_UINT8;
      return kTfLiteOk;
    case kTfLiteInt8:
      tensor_type = TensorTypes_INT8;
      return kTfLiteOk;
    case kTfLiteInt64:
      tensor_type = TensorTypes_INT64;
      return kTfLiteOk;
    case kTfLiteUInt64:
      tensor_type = TensorTypes_UINT64;
      return kTfLiteOk;
    case kTfLiteString:
      tensor_type = TensorTypes_STRING;
      return kTfLiteOk;
    case kTfLiteBool:
      tensor_type = TensorTypes_BOOL;
      return kTfLiteOk;
    case kTfLiteComplex64:
      tensor_type = TensorTypes_COMPLEX64;
      return kTfLiteOk;
    case kTfLiteComplex128:
      tensor_type = TensorTypes_COMPLEX128;
      return kTfLiteOk;
    case kTfLiteResource:
      tensor_type = TensorTypes_RESOURCE;
      return kTfLiteOk;
    case kTfLiteVariant:
      tensor_type = TensorTypes_VARIANT;
      return kTfLiteOk;
    case kTfLiteInt4:
      tensor_type = TensorTypes_INT4;
      return kTfLiteOk;
    case kTfLiteNoType:
      MicroPrintf("Unsupported data type %d in tensor\n", tensor_type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SetRandomInput(const uint32_t random_seed,
                            const ModelT& unpacked_model,
                            MicroInterpreter& interpreter,
                            ModelTestDataT& output_data) {
  std::mt19937 eng(random_seed);
  std::uniform_int_distribution<uint32_t> dist(0, 255);
  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input_tensor(i);
    std::unique_ptr<TensorDataT> test_data(new TensorDataT());
    test_data->input_index = i;
    test_data->layer_number = -1;
    test_data->tensor_index = -1;
    test_data->num_bytes = input->bytes;
    // make this share tensortype with tflite schema later
    TF_LITE_ENSURE_STATUS(ConvertTensorType(input->type, test_data->dtype));
    for (int x = 0; x < input->dims->size; ++x) {
      test_data->shape.push_back(input->dims->data[x]);
    }

    // Pre-populate input tensor with random values.
    uint8_t* input_values = GetTensorData<uint8_t>(input);
    for (size_t j = 0; j < input->bytes; ++j) {
      input_values[j] = dist(eng);
      test_data->data.push_back(input_values[j]);
    }
    output_data.input_data.push_back(std::move(test_data));
  }

  // Get tensor indices for all model input_tensors
  for (size_t i = 0; i < unpacked_model.subgraphs[0]->inputs.size(); ++i) {
    output_data.input_data[i]->tensor_index =
        unpacked_model.subgraphs[0]->inputs[i];
  }
  return kTfLiteOk;
}

std::unique_ptr<char[]> ReadModelFile(const char* model_file_name) {
  std::ifstream model_file(model_file_name, std::ios::binary);
  if (!model_file.is_open()) {
    MicroPrintf("could not open model file \n ");
    return nullptr;
  }

  model_file.seekg(0, std::ios::end);
  size_t num_bytes = model_file.tellg();
  model_file.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> model_data(new char[num_bytes]);
  model_file.read(model_data.get(), num_bytes);

  return model_data;
}

// Stores the Intermediate Tensor data for each layer into the unpacked
// ModelTestDataT class which is packed into the flatbuffer !
TfLiteStatus StoreLayerByLayerData(MicroInterpreter& interpreter,
                                   const ModelT& tflite_model,
                                   ModelTestDataT& output_data) {
  for (size_t i = 0; i < tflite_model.subgraphs.size(); ++i) {
    std::unique_ptr<SubgraphDataT> subgraph_data(new SubgraphDataT());
    subgraph_data->subgraph_index = i;

    for (size_t j = 0; j < tflite_model.subgraphs[i]->operators.size(); ++j) {
      for (size_t k = 0;
           k < tflite_model.subgraphs[i]->operators[j]->outputs.size(); ++k) {
        subgraph_data->outputs.emplace_back(new TensorDataT());
        std::unique_ptr<TensorDataT>& tensor_data =
            subgraph_data->outputs.back();

        // input_index
        tensor_data->input_index = -1;

        // tensor index
        tensor_data->tensor_index =
            tflite_model.subgraphs[i]->operators[j]->outputs[k];

        TfLiteEvalTensor* layer_output_tensor =
            interpreter.GetTensor(subgraph_data->outputs.back()->tensor_index,
                                  subgraph_data->subgraph_index);

        // dims
        tensor_data->shape.assign(
            layer_output_tensor->dims->data,
            layer_output_tensor->dims->data + layer_output_tensor->dims->size);

        // dtype
        TF_LITE_ENSURE_STATUS(
            ConvertTensorType(layer_output_tensor->type, tensor_data->dtype));
        // num_bytes
        tensor_data->num_bytes = EvalTensorBytes(layer_output_tensor);

        uint8_t* tensor_values =
            micro::GetTensorData<uint8_t>(layer_output_tensor);

        // data
        tensor_data->data.assign(
            tensor_values,
            tensor_values + EvalTensorBytes(layer_output_tensor));

        // layer_number
        tensor_data->layer_number = j;
      }
    }
    output_data.subgraph_data.push_back(std::move(subgraph_data));
  }

  return kTfLiteOk;
}

bool WriteToFile(const char* output_file_name, ModelTestDataT& output_data) {
  flatbuffers::DefaultAllocator allocator;
  flatbuffers::FlatBufferBuilder fbb{2048, &allocator};
  auto new_model = ModelTestData::Pack(fbb, &output_data);
  fbb.Finish(new_model);
  return SaveFile(output_file_name,
                  reinterpret_cast<char*>(fbb.GetBufferPointer()),
                  fbb.GetSize());
}

TfLiteStatus Invoke(const Model* model, ModelTestDataT& output_data) {
  const tflite::ModelT unpacked_model = *model->UnPack();
  alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

  TflmOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(CreateOpResolver(op_resolver));

  MicroAllocator* allocator = MicroAllocator::Create(
      tensor_arena, kTensorArenaSize, MemoryPlannerType::kLinear);

  MicroInterpreter interpreter(
      model, op_resolver, allocator,
      MicroResourceVariables::Create(allocator, kNumResourceVariable), nullptr);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TF_LITE_ASSERT(interpreter.preserve_all_tensors());

  MicroPrintf("");  // null MicroPrintf serves as a newline.

  // For streaming models, the interpreter will return kTfLiteAbort if the model
  // does not yet have enough data to make an inference. As such, we need to
  // invoke the interpreter multiple times until we either receive an error or
  // kTfLiteOk. This loop also works for non-streaming models, as they'll just
  // return kTfLiteOk after the first invocation.
  uint32_t seed = kRandomSeed;
  while (true) {
    TF_LITE_ENSURE_STATUS(
        SetRandomInput(seed++, unpacked_model, interpreter, output_data));
    TfLiteStatus status = interpreter.Invoke();
    if ((status != kTfLiteOk) && (static_cast<int>(status) != kTfLiteAbort)) {
      MicroPrintf("Model interpreter invocation failed: %d\n", status);
      return kTfLiteError;
    }

    if (status == kTfLiteOk) {
      break;
    }
  }
  TF_LITE_ENSURE_STATUS(
      StoreLayerByLayerData(interpreter, unpacked_model, output_data));

  return kTfLiteOk;
}
}  // namespace
}  // namespace tflite

/* Usage information:
 This binary will write a debugging flatbuffer to the path provide in 2nd arg
 using the tflite_model provided in the 1st arg :
   `bazel run tensorflow/lite/micro/tools:layer_by_layer_output_tool -- \
     </path/to/input_model.tflite>
     </path/to/output.file_name>` */

int main(int argc, char** argv) {
  if (argc < 2) {
    MicroPrintf("layer_by_layer: invalid usage!\n");
    MicroPrintf(
        "usage: layer_by_layer_output_tool  </path/to/input_model.tflite> "
        "</path/to/output.file_name>");
    return EXIT_FAILURE;
  }

  const char* model_file_name = argv[1];
  const char* output_file_name = argv[2];

  const auto model_file_content = tflite::ReadModelFile(model_file_name);

  if (!model_file_content) {
    MicroPrintf("Could not read model from file: %s", model_file_name);
    return EXIT_FAILURE;
  }

  const tflite::Model* model = tflite::GetModel(model_file_content.get());

  ModelTestDataT output_data;

  TF_LITE_ENSURE_STATUS(tflite::Invoke(model, output_data));

  if (!tflite::WriteToFile(output_file_name, output_data)) {
    MicroPrintf("Could not write to %s", output_file_name);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
