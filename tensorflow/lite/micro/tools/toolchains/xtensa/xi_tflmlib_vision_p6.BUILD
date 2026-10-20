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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "xi_tflmlib_vision_p6",
    srcs = glob(["**/*.cc"]),
    hdrs = glob(["**/*.h"]),
    copts = [
        "-DINCLUDE_XI_CNN",
        "-DIDMA_APP_USE_XTOS",
        "-DXI_ERROR_LEVEL=XI_ERROR_LEVEL_NO_ERROR",
        "-DCNNRT_PERF_LEVEL=CNNRT_PERF_LEVEL_NONE",
        "-Wno-shadow",
    ],
    includes = [
        "flk/include",
        "kernels/include",
        "runtime/include",
    ],
    linkopts = ["-lidma-xtos"],
)
