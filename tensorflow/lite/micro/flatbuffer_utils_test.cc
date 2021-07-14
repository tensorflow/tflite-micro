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

#include "tensorflow/lite/micro/flatbuffer_utils.h"

#include <string>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestFlexbufferWrapper) {
  struct TestParam {
    std::string name;
    std::string type;
    std::string value;
  };

  TestParam params[] = {
      {"xyz", "Int", "613"},
      {"Neuron", "Double", "13.22"},
      {"angle", "Int", "300"},
      {"llama", "Bool", "false"},
      {"Curl", "Float", "0.232"},
      {"aardvark", "Bool", "true"},
      {"ghost", "Double", "0.0000000001"},
      {"123stigma", "Bool", "true"},
  };
  // Index of elements sorted alphabetically by name
  int params_sorted[] = {7, 4, 1, 5, 2, 6, 3, 0};

  const int param_num = sizeof(params) / sizeof(params[0]);

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    for (int i = 0; i < param_num; i++) {
      const std::string& param_value = params[i].value;
      if (params[i].type == "Int") {
        fbb.Int(params[i].name.c_str(), std::stoi(param_value));
      } else if (params[i].type == "Bool") {
        fbb.Bool(params[i].name.c_str(), param_value == "true");
      } else if (params[i].type == "Double") {
        fbb.Double(params[i].name.c_str(), std::stod(param_value));
      } else if (params[i].type == "Float") {
        fbb.Float(params[i].name.c_str(), std::stof(param_value));
      }
    }
  });
  fbb.Finish();
  const std::vector<uint8_t> buffer = fbb.GetBuffer();
  tflite::FlexbufferWrapper wrapper(buffer.data(), buffer.size());
  for (int i = 0; i < param_num; i++) {
    std::string& param_value = params[params_sorted[i]].value;
    if (params[params_sorted[i]].type == "Int") {
      TF_LITE_MICRO_EXPECT(wrapper.ElementAsInt32(i) == std::stoi(param_value));
    } else if (params[params_sorted[i]].type == "Bool") {
      TF_LITE_MICRO_EXPECT(wrapper.ElementAsBool(i) == (param_value == "true"));
    } else if (params[params_sorted[i]].type == "Double") {
      TF_LITE_MICRO_EXPECT(wrapper.ElementAsDouble(i) ==
                           std::stod(param_value));
    } else if (params[params_sorted[i]].type == "Float") {
      TF_LITE_MICRO_EXPECT(wrapper.ElementAsFloat(i) == std::stof(param_value));
    }
  }
}

TF_LITE_MICRO_TESTS_END
