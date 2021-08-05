#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace ringbuffer {

void* pointer_to_persistent_buffer = NULL;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  int32_t persistent_buffer_size;

  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);

  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

  auto values = map.Values();

  persistent_buffer_size = values[0].AsInt32();

  if (pointer_to_persistent_buffer==NULL){
    pointer_to_persistent_buffer = context->AllocatePersistentBuffer(context, persistent_buffer_size);
  };
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return  kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* prev_data = GetInput(context, node, 1);
  const TfLiteTensor* start_address_size = GetInput(context, node, 2);
  TfLiteTensor* output = GetOutput(context, node, 0);

  int start_address = start_address_size->data.i32[0];
  int prev_data_size = start_address_size->data.i32[1];
  char* prev_data_buffer = (char*)pointer_to_persistent_buffer + start_address;

  memcpy(output->data.raw ,prev_data_buffer, prev_data_size);
  memcpy(output->data.raw+prev_data_size, input->data.raw, input->bytes);
  memcpy(prev_data_buffer, output->data.raw+input->bytes, prev_data_size);

  return kTfLiteOk;
}

}  // namespace ringbuffer

TfLiteRegistration* Register_Ringbuffer() {
  static TfLiteRegistration r = {ringbuffer::Init, nullptr, ringbuffer::Prepare,
                                 ringbuffer::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
