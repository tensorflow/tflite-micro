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

#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/kernels/xcore/xcore_utils.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char* Lookup_8_OpCode = "XC_lookup_8";
constexpr const char* MaxPool2D_OpCode = "XC_maxpool2d";
constexpr const char* AvgPool2D_OpCode = "XC_avgpool2d";
constexpr const char* AvgPool2D_Global_OpCode = "XC_avgpool2d_global";
constexpr const char* FullyConnected_8_OpCode = "XC_fc";
constexpr const char* Conv2D_Shallow_OpCode = "XC_conv2d_shallowin";
constexpr const char* Conv2D_Deep_OpCode = "XC_conv2d_deep";
constexpr const char* Conv2D_1x1_OpCode = "XC_conv2d_1x1";
constexpr const char* Conv2D_Depthwise_OpCode = "XC_conv2d_depthwise";
constexpr const char* Add_8_OpCode = "XC_add_8";
constexpr const char* Ringbuffer_OpCode = "XC_ringbuffer";
constexpr const char* Pad_OpCode = "XC_pad";

// Binarized ops
constexpr const char* Bsign_8_OpCode = "XC_bsign_8";
constexpr const char* BConv2d_Bitpacked_OpCode = "XC_bconv2d_bin";
constexpr const char* BConv2d_Bitpacked_DeepIn_OpCode = "XC_bconv2d_bin_DI";
constexpr const char* BConv2d_Int8_OpCode = "XC_bconv2d_int8";
constexpr const char* BConv2d_Int8_DeepIn_DeepOut_OpCode =
    "XC_bconv2d_int8_DIDO";

// Currently unused, may be deprecated
constexpr const char* Requantize_16_to_8_OpCode = "XC_requantize_16_to_8";
constexpr const char* ArgMax2D_OpCode = "XC_argmax_16";

TfLiteRegistration* Register_Conv2D_Shallow();
TfLiteRegistration* Register_Conv2D_Deep();
TfLiteRegistration* Register_Conv2D_1x1();
TfLiteRegistration* Register_Conv2D_Depthwise();
TfLiteRegistration* Register_FullyConnected_8();
TfLiteRegistration* Register_MaxPool2D();
TfLiteRegistration* Register_AvgPool2D();
TfLiteRegistration* Register_AvgPool2D_Global();
TfLiteRegistration* Register_Lookup_8();

// Binarized ops
TfLiteRegistration* Register_BSign_8();
TfLiteRegistration* Register_BConv2D_Bitpacked_Deepin();
TfLiteRegistration* Register_BConv2D_Bitpacked();
TfLiteRegistration* Register_BConv2D_Int8_Deepin_Deepout();
TfLiteRegistration* Register_BConv2D_Int8();

// Under development
TfLiteRegistration* Register_Pad();
TfLiteRegistration* Register_Add_8();
TfLiteRegistration* Register_Ringbuffer();

// operators not currently inserted by the XCORE converter
TfLiteRegistration* Register_Requantize_16_to_8();
TfLiteRegistration* Register_ArgMax_16();

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_H_
