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

# --- 1. Default System & Shared Test Support Sources --------------------------
set(TFLM_SYSTEM_DEFAULT_SRCS
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/debug_log.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/micro_time.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/system_setup.cc"
)

file(GLOB _tflm_flexbuffers_test_data_srcs CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/*_flexbuffers_generated_data.cc"
  "${PROJECT_SOURCE_DIR}/signal/micro/kernels/*_flexbuffers_generated_data.cc")
list(FILTER _tflm_flexbuffers_test_data_srcs EXCLUDE REGEX "/\\.[^/]*$")

set(TFLM_TEST_SUPPORT_SRCS
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/fake_micro_context.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/conv_test_common.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/kernel_runner.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/mock_micro_graph.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/test_helper_custom_ops.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/test_helpers.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/testing/test_conv_model.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/testdata/conv_test_data.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/testdata/lstm_test_data.cc"
  "${PROJECT_SOURCE_DIR}/signal/testdata/fft_test_data.cc"
  ${_tflm_flexbuffers_test_data_srcs}
)

# --- 2. Core, Kernel, Backend, and Signal Sources -----------------------------
file(GLOB TFLM_MICRO_CORE_SRCS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/*.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/*.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/*.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/internal/*.cc"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/kernel_util.cc")
list(FILTER TFLM_MICRO_CORE_SRCS EXCLUDE REGEX "(/\\.[^/]*|_test\\.cc)$")
list(REMOVE_ITEM TFLM_MICRO_CORE_SRCS ${TFLM_SYSTEM_DEFAULT_SRCS} ${TFLM_TEST_SUPPORT_SRCS})

file(GLOB TFLM_MICRO_KERNEL_SRCS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/*.cc")
list(FILTER TFLM_MICRO_KERNEL_SRCS EXCLUDE
  REGEX "(/\\.[^/]*|(_test|_flexbuffers_generated_data)\\.cc)$")
list(REMOVE_ITEM TFLM_MICRO_KERNEL_SRCS
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/kernel_util.cc"
  ${TFLM_TEST_SUPPORT_SRCS})

foreach(_backend IN ITEMS cmsis_nn ethos_u hexagon xtensa)
  string(TOUPPER "${_backend}" _upper)
  set(_bdir "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/${_backend}")
  file(GLOB_RECURSE TFLM_MICRO_KERNEL_SRCS_${_upper} CONFIGURE_DEPENDS
    "${_bdir}/*.cc" "${_bdir}/*.c" "${_bdir}/*.S" "${_bdir}/*.s")
  list(FILTER TFLM_MICRO_KERNEL_SRCS_${_upper} EXCLUDE REGEX "(/\\.[^/]*|_test\\.(cc|c))$")
  file(GLOB_RECURSE TFLM_MICRO_KERNEL_HDRS_${_upper} CONFIGURE_DEPENDS "${_bdir}/*.h")
  list(FILTER TFLM_MICRO_KERNEL_HDRS_${_upper} EXCLUDE REGEX "(/\\.[^/]*|_test\\.h)$")
endforeach()

file(GLOB TFLM_SIGNAL_SRCS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/signal/micro/kernels/*.cc"
  "${PROJECT_SOURCE_DIR}/signal/src/*.cc"
  "${PROJECT_SOURCE_DIR}/signal/src/kiss_fft_wrappers/*.cc")
list(FILTER TFLM_SIGNAL_SRCS EXCLUDE
  REGEX "(/\\.[^/]*|(_test|_flexbuffers_generated_data)\\.cc)$")

# --- 3. Public Header Sets (FILE_SET HEADERS) ---------------------------------
file(GLOB TFLM_MICRO_PUBLIC_HDRS CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/signal/micro/kernels/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/c/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/internal/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/internal/reference/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/kernels/internal/reference/integer_ops/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/*.h"
  "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/schema/*.h")
list(FILTER TFLM_MICRO_PUBLIC_HDRS EXCLUDE
  REGEX "(/\\.[^/]*|(_test|_test_helpers|_flexbuffers_generated_data)\\.h)$")
foreach(_src IN LISTS TFLM_TEST_SUPPORT_SRCS)
  string(REGEX REPLACE "\\.cc$" ".h" _hdr "${_src}")
  list(REMOVE_ITEM TFLM_MICRO_PUBLIC_HDRS "${_hdr}")
endforeach()

file(GLOB TFLM_SIGNAL_PUBLIC_HDRS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/signal/src/*.h")
list(FILTER TFLM_SIGNAL_PUBLIC_HDRS EXCLUDE REGEX "(/\\.[^/]*|_test\\.h)$")

# --- 4. Unit, Example, and Integration Test Registration ----------------------
function(_tflm_add_unit_tests)
  cmake_parse_arguments(ARG "" "PREFIX" "DIRS;LABELS;DEPS" ${ARGN})
  set(_globs "")
  foreach(_dir IN LISTS ARG_DIRS)
    list(APPEND _globs "${PROJECT_SOURCE_DIR}/${_dir}/*_test.cc")
  endforeach()
  file(GLOB _test_srcs CONFIGURE_DEPENDS ${_globs})
  list(FILTER _test_srcs EXCLUDE REGEX "/\\.[^/]*$")
  foreach(_src IN LISTS _test_srcs)
    cmake_path(GET _src STEM _stem)
    set(_assets "")
    set(_deps ${ARG_DEPS})
    if(_stem STREQUAL "memory_arena_threshold_test")
      if(CMAKE_CROSSCOMPILING OR TFLM_OPTIMIZED_KERNELS OR TFLM_CO_PROCESSOR)
        continue()
      endif()
      set(_assets "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/models/keyword_scrambled.tflite")
    elseif(_stem STREQUAL "ethos_u_test")
      set(_assets "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/models/person_detect_vela.tflite")
    elseif(_stem STREQUAL "kiss_fft_wrappers_test")
      list(APPEND _deps tflm::kissfft)
    endif()
    tflm_add_cc_test(
      NAME "${ARG_PREFIX}${_stem}"
      SRCS "${_src}"
      ASSETS ${_assets}
      DEPS ${_deps}
      LABELS ${ARG_LABELS})
  endforeach()
endfunction()

function(tflm_register_tests)
  _tflm_add_unit_tests(
    PREFIX "tflm_"
    DIRS
      tensorflow/lite/micro
      tensorflow/lite/micro/arena_allocator
      tensorflow/lite/micro/memory_planner
      tensorflow/lite/micro/testing
    LABELS unit core)
  _tflm_add_unit_tests(
    PREFIX "tflm_kernel_"
    DIRS tensorflow/lite/micro/kernels
    LABELS unit kernel)
  if(TFLM_ENABLE_SIGNAL)
    _tflm_add_unit_tests(
      PREFIX "tflm_signal_"
      DIRS signal/micro/kernels signal/src signal/src/kiss_fft_wrappers
      DEPS tflm::signal
      LABELS unit signal)
  endif()

  if(TFLM_BUILD_EXAMPLES)
    set(_ex "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/examples")
    tflm_add_cc_test(NAME tflm_detection_responder_test LABELS example
      SRCS
        "${_ex}/person_detection/detection_responder_test.cc"
        "${_ex}/person_detection/detection_responder.cc")
    tflm_add_cc_test(NAME tflm_hello_world_test LABELS example
      SRCS "${_ex}/hello_world/hello_world_test.cc"
      ASSETS
        "${_ex}/hello_world/models/hello_world_float.tflite"
        "${_ex}/hello_world/models/hello_world_int8.tflite")
    tflm_add_cc_test(NAME tflm_image_provider_test LABELS example
      SRCS
        "${_ex}/person_detection/image_provider_test.cc"
        "${_ex}/person_detection/image_provider.cc"
        "${_ex}/person_detection/model_settings.cc")
    tflm_add_cc_test(NAME tflm_person_detection_test LABELS example
      SRCS
        "${_ex}/person_detection/person_detection_test.cc"
        "${_ex}/person_detection/model_settings.cc"
      ASSETS
        "${_ex}/person_detection/testdata/no_person.bmp"
        "${_ex}/person_detection/testdata/person.bmp"
        "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/models/person_detect.tflite")
    if(TFLM_ENABLE_SIGNAL)
      file(GLOB _ms_wavs CONFIGURE_DEPENDS "${_ex}/micro_speech/testdata/*.wav")
      tflm_add_cc_test(NAME tflm_micro_speech_test DEPS tflm::signal LABELS example
        SRCS "${_ex}/micro_speech/micro_speech_test.cc"
        ASSETS
          "${_ex}/micro_speech/models/audio_preprocessor_int8.tflite"
          "${_ex}/micro_speech/models/micro_speech_quantized.tflite"
          ${_ms_wavs})
    endif()
  endif()

  if(TFLM_BUILD_INTEGRATION_TESTS AND TFLM_ENABLE_SIGNAL)
    foreach(_op IN ITEMS add conv leaky_relu pad quantize strided_slice sub transpose_conv)
      set(_seanet_dir "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/integration_tests/seanet/${_op}")
      file(GLOB _seanet_assets CONFIGURE_DEPENDS
        "${_seanet_dir}/*.tflite" "${_seanet_dir}/*.csv")
      tflm_add_cc_test(
        NAME "tflm_seanet_${_op}_integration_test"
        SRCS
          "${_seanet_dir}/integration_tests.cc"
          "${PROJECT_SOURCE_DIR}/python/tflite_micro/python_ops_resolver.cc"
        ASSETS ${_seanet_assets}
        DEPS tflm::signal
        LABELS integration seanet)
    endforeach()
  endif()
endfunction()
