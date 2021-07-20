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

#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

CustomOptionParser::CustomOptionParser(const flexbuffers::Map& map)
    : keys_(flexbuffers::TypedVector::EmptyTypedVector()),
      values_(flexbuffers::Vector::EmptyVector()) {
  keys_ = map.Keys();
  values_ = map.Values();
}

CustomOptionParser::CustomOptionParser(const char* buffer, size_t buffer_length)
    : CustomOptionParser::CustomOptionParser(
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer),
                               buffer_length)
              .AsMap()) {
  assert(buffer != nullptr);
  assert(buffer_length > 0);
}

flexbuffers::Reference CustomOptionParser::parseNamedCustomOption(
    const std::string& name) const {
  for (int i = 0; i < keys_.size(); ++i) {
    const auto& key = keys_[i].AsString().str();
    if (key.compare(name) == 0) {
      return values_[i];
    }
  }
  return flexbuffers::Reference(nullptr, 1, flexbuffers::NullPackedType());
}

flexbuffers::Vector CustomOptionParser::parseElementwiseJobSizes() const {
  auto par_parser = CustomOptionParser(parseNamedCustomOption("par").AsMap());
  auto job_sizes = par_parser.parseNamedCustomOption("eg").AsVector();
  auto n_threads = par_parser.parseNamedCustomOption("th").AsInt32();
  TFLITE_DCHECK_EQ(n_threads, job_sizes.size());  // TODO: remove this check
  return job_sizes;
}

//*****************************
// ExecutionPlan only
//*****************************
void parse_custom_options(TfLiteContext* context, const char* buffer,
                          size_t length, ExecutionPlan* plan) {
  parse_custom_options(context, buffer, length, nullptr, nullptr, nullptr,
                       nullptr, plan);
}

//*****************************
// All Params
//*****************************
void parse_custom_options(TfLiteContext* context, const char* buffer,
                          size_t length, int32_t* stride_h, int32_t* stride_w,
                          int32_t* pool_h, int32_t* pool_w,
                          ExecutionPlan* plan) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() <<
  // std::endl;
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

  auto keys = map.Keys();
  auto values = map.Values();
  for (int i = 0; i < map.size(); ++i) {
    const std::string& key = keys[i].AsString().str();

    if (key.compare("stride") == 0) {
      const auto& vec =
          values[i].AsVector();  // values represent [stride_h, stride_w]
      if (stride_h) *stride_h = vec[0].AsInt32();
      if (stride_w) *stride_w = vec[1].AsInt32();
    } else if (key.compare("stride_h") == 0) {
      if (stride_h) *stride_h = values[i].AsInt32();
    } else if (key.compare("stride_w") == 0) {
      if (stride_w) *stride_w = values[i].AsInt32();
    } else if (key.compare("pool") == 0) {
      const auto& vec =
          values[i].AsVector();  // values represent [pool_h, pool_w]
      if (pool_h) *pool_h = vec[0].AsInt32();
      if (pool_w) *pool_w = vec[1].AsInt32();
    } else if (key.compare("par") == 0) {
      if (plan) {
        const auto& plan_map = values[i].AsMap();
        auto plan_keys = plan_map.Keys();
        auto plan_values = plan_map.Values();
        for (int j = 0; j < plan_map.size(); ++j) {
          const std::string& plan_key = plan_keys[j].AsString().str();
          if (plan_key.compare("th") == 0) {
            plan->SetNumThreads(plan_values[j].AsInt32());
          } else if (plan_key.compare("cg") == 0) {
            const auto& changrps = plan_values[j].AsVector();
            plan->changrps.allocate(context, changrps.size());
            for (int k = 0; k < changrps.size(); k++) {
              auto changrp =
                  changrps[k].AsVector();  // values represent [start, end]
              plan->changrps.append(
                  {k, changrp[0].AsInt32(),
                   changrp[1].AsInt32() - changrp[0].AsInt32() + 1});
            }
          } else if (plan_key.compare("rc") == 0) {
            const auto& regions = plan_values[j].AsVector();
            plan->regions.allocate(context, regions.size());
            for (int k = 0; k < regions.size(); k++) {
              auto region =
                  regions[k]
                      .AsVector();  // values represent [top, left, rows, cols]
              plan->regions.append({region[0].AsInt32(), region[1].AsInt32(),
                                    region[2].AsInt32(), region[3].AsInt32()});
            }
          }
        }
      }
    } else if (key.compare("mem") == 0) {
      if (plan) {
        const auto& vec = values[i].AsVector();  // values represent [weights
                                                 // scratch, bias scratch]
        plan->SetWeightsScratchSize(vec[0].AsInt32());
        plan->SetBiasScratchSize(vec[1].AsInt32());
      }
    }
  }
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
