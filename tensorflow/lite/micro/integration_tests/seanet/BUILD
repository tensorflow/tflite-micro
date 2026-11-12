# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load(
    "//tensorflow/lite/micro/tools/toolchains:platform_suites.bzl",
    "tflm_platform_test_suite",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tflm_platform_test_suite(
    name = "seanet",
    profiles = [
        "cortex_m4",
        "cortex_m55",
        "xtensa_hifi3z",
        "xtensa_hifi5",
    ],
    targets = [
        "//tensorflow/lite/micro/integration_tests/seanet/add:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/conv:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/leaky_relu:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/pad:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/quantize:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/strided_slice:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/sub:integration_test",
        "//tensorflow/lite/micro/integration_tests/seanet/transpose_conv:integration_test",
    ],
    visibility = ["//visibility:public"],
)
