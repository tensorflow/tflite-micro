#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

using PytorchOpsResolver = tflite::MicroMutableOpResolver<128>;

TfLiteStatus InitPytorchOpsResolver(PytorchOpsResolver& resolver);
