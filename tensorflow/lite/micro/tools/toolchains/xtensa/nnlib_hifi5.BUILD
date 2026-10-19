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
    name = "nnlib_hifi5",
    srcs = glob(
        ["algo/**/*.c"],
        exclude = [
            "algo/layers/cnn/src/xa_nn_cnn_api.c",
            "algo/layers/gru/src/xa_nn_gru_api.c",
            "algo/layers/lstm/src/xa_nn_lstm_api.c",
        ],
    ),
    hdrs = glob([
        "algo/**/*.h",
        "include/**/*.h",
    ]),
    copts = [
        "-x",
        "c",
        "-Wno-shadow",
        "-Wno-unused-function",
        "-Wno-invalid-command-line-argument",
    ],
    defines = [
        "NNLIB_V2=1",
        "MODEL_INT16=1",
        "EIGEN_NO_MALLOC=1",
        "hifi5=1",
    ],
    includes = [
        ".",
        "algo/common/include",
        "algo/kernels",
        "algo/ndsp/hifi5/include",
        "include",
        "include/nnlib",
    ],
)
