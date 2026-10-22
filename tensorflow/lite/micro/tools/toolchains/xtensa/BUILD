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

load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "tflm_sh_runner",
    "tflm_xtensa_runner_data",
    "tflm_xtensa_toolchains",
)

package(default_visibility = ["//visibility:public"])

CORES = {
    "fusion_f1": {
        "arch_def": "-DHIFI3",
        "core": "P29A_FusionF1",
    },
    "hifi3z": {
        "arch_def": "-DHIFI3",
        "core": "P29A_HiFi3z",
    },
    "hifi5": {
        "arch_def": "-DHIFI5",
        "core": "PRD_H5_RDO_07_01_2022",
    },
    "vision_p6": {
        "arch_def": "-DVISION_P6",
        "core": "P29A_VP6",
    },
}

tflm_xtensa_toolchains(cores = CORES)

[
    tflm_sh_runner(
        name = "run_%s" % name,
        srcs = ["run_xtensa.sh"],
        data = tflm_xtensa_runner_data(core = name),
    )
    for name in CORES.keys()
]

cc_test(
    name = "smoke_test",
    srcs = ["test_smoke.cc"],
)
