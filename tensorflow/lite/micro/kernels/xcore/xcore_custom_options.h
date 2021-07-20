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

#ifndef XCORE_CUSTOM_OPTIONS_H_
#define XCORE_CUSTOM_OPTIONS_H_

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_planning.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void parse_custom_options(TfLiteContext* context, const char* buffer,
                          size_t length, ExecutionPlan* plan);

void parse_custom_options(TfLiteContext* context, const char* buffer,
                          size_t length, int32_t* stride_h = nullptr,
                          int32_t* stride_w = nullptr,
                          int32_t* pool_h = nullptr, int32_t* pool_w = nullptr,
                          ExecutionPlan* plan = nullptr);

class CustomOptionParser {
 private:
  flexbuffers::TypedVector keys_;
  flexbuffers::Vector values_;

 protected:
  template <typename T>
  void inline parseIntTuple(const flexbuffers::Vector& vec, int idx,
                            T& arg) const {
    assert(vec.size() > idx);
    arg = vec[idx].AsInt32();
  }

  template <typename T, typename... TArgs>
  void inline parseIntTuple(const flexbuffers::Vector& vec, int idx, T& arg,
                            TArgs&... args) const {
    assert(vec.size() > idx);
    arg = vec[idx].AsInt32();
    parseIntTuple(vec, idx + 1, args...);
  }

 public:
  CustomOptionParser(const flexbuffers::Map& map);
  CustomOptionParser(const char* buffer, size_t buffer_length);
  flexbuffers::Reference parseNamedCustomOption(const std::string& name) const;
  flexbuffers::Vector parseElementwiseJobSizes() const;

  template <typename... TArgs>
  void parseNamedTuple(const std::string& name, TArgs&... args) const {
    auto vec = parseNamedCustomOption(name).AsVector();
    assert(vec.size() >= sizeof...(TArgs));
    parseIntTuple(vec, 0, args...);
  }
};

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_CUSTOM_OPTIONS_H_
