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

#include "tensorflow/lite/micro/kernels/xcore/xcore_extended_interpreter.h"

#include <iostream>
#include <vector>

#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace micro {
namespace xcore {

constexpr int max_log_len = 256;

typedef TfLiteStatus (*invoke_function_t)(TfLiteContext*, TfLiteNode*);

//****************************
//****************************
//****************************
// Callback classes
//****************************
//****************************
//****************************
class CallbackContext {
 public:
  struct Reg_CallBack {
    invoke_function_t invoke_fn;
    const TfLiteRegistration* reg;
  };

 public:
  CallbackContext()
      : current_operator(0),
        preinvoke_callback(nullptr),
        postinvoke_callback(nullptr) {}
  void Reset() {
    current_operator = 0;
    preinvoke_callback = nullptr;
    postinvoke_callback = nullptr;
    invoke_functions_and_regs.clear();
  }
  int current_operator;
  invoke_callback_t preinvoke_callback;
  invoke_callback_t postinvoke_callback;
  std::vector<Reg_CallBack> invoke_functions_and_regs;
};
static CallbackContext gCallbackContext;

TfLiteStatus CallbackInvoke(TfLiteContext* context, TfLiteNode* node) {
  int current_operator = gCallbackContext.current_operator;

  invoke_function_t invoke =
      gCallbackContext.invoke_functions_and_regs[current_operator].invoke_fn;

  if (gCallbackContext.preinvoke_callback)
    gCallbackContext.preinvoke_callback(current_operator);
  TfLiteStatus status = invoke(context, node);
  if (gCallbackContext.postinvoke_callback)
    gCallbackContext.postinvoke_callback(current_operator);
  gCallbackContext.current_operator++;

  return status;
}

//****************************
//****************************
//****************************
// BufferedErrorReporter
//****************************
//****************************
//****************************
int BufferedErrorReporter::Report(const char* format, ...) {
  va_list args;
  va_start(args, format);
  int code = Report(format, args);
  va_end(args);
  return code;
}

int BufferedErrorReporter::Report(const char* format, va_list args) {
  char log_buffer[max_log_len];
  std::vsnprintf(log_buffer, max_log_len, format, args);
  log_stream_ << log_buffer << std::endl;
  return 0;
}

std::string BufferedErrorReporter::GetError() {
  std::string error = log_stream_.str();
  Clear();
  return error;
}

void BufferedErrorReporter::Clear() { log_stream_.str(""); }

//****************************
//****************************
//****************************
// ExtendedXCoreInterpreter
//****************************
//****************************
//****************************
ExtendedXCoreInterpreter::ExtendedXCoreInterpreter(
    const tflite::Model* model, const tflite::MicroOpResolver& resolver,
    uint8_t* arena, size_t arena_size, tflite::ErrorReporter* reporter,
    XCoreProfiler* profiler)
    : XCoreInterpreter(model, resolver, arena, arena_size, reporter, profiler),
      reporter_(reporter) {}

ExtendedXCoreInterpreter::ExtendedXCoreInterpreter(
    const tflite::Model* model, const tflite::MicroOpResolver& resolver,
    tflite::MicroAllocator* allocator, tflite::ErrorReporter* reporter,
    XCoreProfiler* profiler)
    : XCoreInterpreter(model, resolver, allocator, reporter, profiler),
      reporter_(reporter) {}

size_t ExtendedXCoreInterpreter::input_tensor_index(size_t input_index) {
  const SubGraph* subgraph = model_->subgraphs()->Get(0);
  return subgraph->inputs()->Get(input_index);
}

size_t ExtendedXCoreInterpreter::output_tensor_index(size_t output_index) {
  const SubGraph* subgraph = model_->subgraphs()->Get(0);
  return subgraph->outputs()->Get(output_index);
}

TfLiteStatus ExtendedXCoreInterpreter::Invoke(
    invoke_callback_t preinvoke_callback,
    invoke_callback_t postinvoke_callback) {
  if (preinvoke_callback || postinvoke_callback) {
    gCallbackContext.preinvoke_callback = preinvoke_callback;
    gCallbackContext.postinvoke_callback = postinvoke_callback;

    size_t subgraph_idx = 0;
    const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
    TFLITE_DCHECK(subgraph != nullptr);

    auto* opcodes = model_->operator_codes();
    uint32_t operators_size = NumSubgraphOperators(subgraph);

    for (size_t i = 0; i < operators_size; ++i) {
      const auto* op = subgraph->operators()->Get(i);
      const size_t index = op->opcode_index();

      if (index >= opcodes->size()) {
        MicroPrintf("Missing registration for opcode_index %d\n", index);
        return kTfLiteError;
      }

      //[asj] cannot see a better way of getting the allocations
      const TfLiteRegistration* reg = graph_.GetAllocations()[subgraph_idx]
                                          .node_and_registrations[i]
                                          .registration;

      CallbackContext::Reg_CallBack rb = {reg->invoke, reg};
      (const_cast<TfLiteRegistration*>(reg))->invoke = CallbackInvoke;
      gCallbackContext.invoke_functions_and_regs.push_back(rb);
    }
  }

  TfLiteStatus invoke_status = XCoreInterpreter::Invoke();

  // Set back the original invoke function
  if (preinvoke_callback || postinvoke_callback) {
    for (auto cbp : gCallbackContext.invoke_functions_and_regs) {
      (const_cast<TfLiteRegistration*>(cbp.reg))->invoke = cbp.invoke_fn;
    }
    gCallbackContext.Reset();
  }
  if (invoke_status != kTfLiteOk) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::SetTensor(size_t tensor_index,
                                                 const void* value,
                                                 const int size,
                                                 const int* shape,
                                                 const int type) {
  const SubGraph* subgraph = model_->subgraphs()->Get(0);
  const Tensor* tensor_p = subgraph->tensors()->Get(tensor_index);

  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  if (tensor_p->shape()->Length() != size) {
    reporter_->Report("tensor dims size %d != %d", tensor_p->shape()->Length(),
                      size);
    return kTfLiteError;
  }

  for (int i = 0; i < size; i++) {
    if (tensor_p->shape()->Get(i) != shape[i]) {
      reporter_->Report("tensor dim %d != %d", tensor_p->shape()->Get(i),
                        shape[i]);
      return kTfLiteError;
    }
  }

  TfLiteTensor* tf_tensor_p = tensor(tensor_index);
  if (tf_tensor_p->type != type) {
    reporter_->Report("tensor type %d != %d", tf_tensor_p->type, type);
    return kTfLiteError;
  }

  std::memcpy(tf_tensor_p->data.raw, value, tf_tensor_p->bytes);
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetTensor(size_t tensor_index,
                                                 void* value, const int size,
                                                 const int* shape,
                                                 const int type) {
  const SubGraph* subgraph = model_->subgraphs()->Get(0);
  const Tensor* tensor_p = subgraph->tensors()->Get(tensor_index);

  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  if (tensor_p->shape()->Length() != size) {
    reporter_->Report("tensor dims size %d != %d", tensor_p->shape()->Length(),
                      size);
    return kTfLiteError;
  }

  for (int i = 0; i < size; i++) {
    if (tensor_p->shape()->Get(i) != shape[i]) {
      reporter_->Report("tensor dim %d != %d", tensor_p->shape()->Get(i),
                        shape[i]);
      return kTfLiteError;
    }
  }

  TfLiteTensor* tf_tensor_p = tensor(tensor_index);
  if (tf_tensor_p->type != type) {
    reporter_->Report("tensor type %d != %d", tf_tensor_p->type, type);
    return kTfLiteError;
  }

  std::memcpy(value, tf_tensor_p->data.raw, tf_tensor_p->bytes);
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetTensorDetailsBufferSizes(
    size_t tensor_index, size_t* dims, size_t* scales, size_t* zero_points) {
  const SubGraph* subgraph = model_->subgraphs()->Get(0);
  const Tensor* tensor_p = subgraph->tensors()->Get(tensor_index);

  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  *dims = 0;
  auto* shape_vector = tensor_p->shape();
  if (shape_vector) {
    *dims = shape_vector->Length();
  }

  *scales = 1;
  *zero_points = 1;
  const tflite::QuantizationParameters* quantization_params =
      tensor_p->quantization();
  if (quantization_params) {
    auto* scale_vector = quantization_params->scale();
    if (scale_vector) {
      *scales = scale_vector->Length();
    }

    auto* zero_points_vector = quantization_params->zero_point();
    if (zero_points_vector) {
      *zero_points = zero_points_vector->Length();
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetTensorDetails(
    size_t tensor_index, char* name, int name_len, int* shape, int* type,
    float* scale, int32_t* zero_point) {
  const SubGraph* subgraph = model_->subgraphs()->Get(0);
  const Tensor* tensor_p = subgraph->tensors()->Get(tensor_index);

  if (tensor_p == nullptr) {
    std::cout << "error kTfLiteError" << std::endl;
    return kTfLiteError;
  }

  if (tensor_p->name()) {
    std::strncpy(name, tensor_p->name()->c_str(), name_len);
  }

  auto* shape_vector = tensor_p->shape();
  if (shape_vector) {
    for (int i = 0; i < shape_vector->Length(); i++) {
      shape[i] = shape_vector->Get(i);
    }
  }

  scale[0] = 0.0;
  zero_point[0] = 0;

  ConvertTensorType(tensor_p->type(), (TfLiteType*)type, reporter_);
  const tflite::QuantizationParameters* quantization_params =
      tensor_p->quantization();
  if (quantization_params) {
    auto* scale_vector = quantization_params->scale();
    if (scale_vector) {
      for (int i = 0; i < scale_vector->Length(); i++) {
        scale[i] = scale_vector->Get(i);
      }
    }

    auto* zero_points_vector = quantization_params->zero_point();
    if (zero_points_vector) {
      for (int i = 0; i < zero_points_vector->Length(); i++) {
        zero_point[i] = zero_points_vector->Get(i);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetOperatorDetailsBufferSizes(
    size_t operator_index, size_t* inputs, size_t* outputs) {
  size_t subgraph_idx = 0;
  const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
  TFLITE_DCHECK(subgraph != nullptr);
  auto* opcodes = model_->operator_codes();
  uint32_t operators_size = NumSubgraphOperators(subgraph);

  if (operator_index >= operators_size) {
    reporter_->Report("Invalid operator index %d", operator_index);
    return kTfLiteError;
  }

  //[asj] cannot see a better way of getting the allocations
  tflite::NodeAndRegistration& nr = graph_.GetAllocations()[subgraph_idx]
                                        .node_and_registrations[operator_index];

  auto node = nr.node;

  *inputs = node.inputs->size;
  *outputs = node.outputs->size;

  return kTfLiteOk;
}

TfLiteStatus ExtendedXCoreInterpreter::GetOperatorDetails(
    size_t operator_index, char* name, int name_len, int* version, int* inputs,
    int* outputs) {
  size_t subgraph_idx = 0;
  const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
  TFLITE_DCHECK(subgraph != nullptr);
  auto* opcodes = model_->operator_codes();
  uint32_t operators_size = NumSubgraphOperators(subgraph);

  if (operator_index >= operators_size) {
    reporter_->Report("Invalid operator index %d", operator_index);
    return kTfLiteError;
  }

  //[asj] cannot see a better way of getting the allocations
  tflite::NodeAndRegistration& node_and_reg =
      graph_.GetAllocations()[subgraph_idx]
          .node_and_registrations[operator_index];

  const TfLiteNode& node = node_and_reg.node;
  const TfLiteRegistration* reg = node_and_reg.registration;

  if (reg->custom_name != nullptr) {
    std::strncpy(name, reg->custom_name, name_len);
  } else {
    std::strncpy(name, tflite::EnumNamesBuiltinOperator()[reg->builtin_code],
                 name_len);
  }
  *version = reg->version;
  for (int i = 0; i < node.inputs->size; i++) {
    inputs[i] = node.inputs->data[i];
  }
  for (int i = 0; i < node.outputs->size; i++) {
    outputs[i] = node.outputs->data[i];
  }

  return kTfLiteOk;
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
