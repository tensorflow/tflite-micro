#!/usr/bin/env bash
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
#
# Runs CMake host builds, unit/example tests, consumer package smoke tests, and
# ecosystem interop checks in an out-of-tree temporary directory.
#
# Usage:
#   tensorflow/lite/micro/tools/ci_build/test_cmake.sh [all|ci|compression_debug|core_only|interop]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${ROOT_DIR}/tensorflow/lite/micro/tools/ci_build/helper_functions.sh"

MODE="${1:-all}"
MODEL_FILE="${ROOT_DIR}/tensorflow/lite/micro/examples/memory_footprint/models/simple_add_model.tflite"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tflm_cmake_ci.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
DEPS_DIR="${TFLM_DEPS_DIR:-${WORK_DIR}/deps}"

if [[ -z "${CMAKE_GENERATOR:-}" ]] && command -v ninja >/dev/null 2>&1; then
  export CMAKE_GENERATOR=Ninja
fi
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
export CTEST_PARALLEL_LEVEL="${JOBS}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

ensure_full_deps() {
  readable_run python3 "${ROOT_DIR}/cmake/download_deps.py" "${DEPS_DIR}"
}

run_cmake_preset() {
  local preset="$1" build_dir="$2"
  shift 2
  readable_run cmake --preset "${preset}" -S "${ROOT_DIR}" -B "${build_dir}" "$@"
  readable_run cmake --build "${build_dir}" --parallel "${JOBS}"
  readable_run ctest --test-dir "${build_dir}" \
    --output-on-failure --no-tests=error --parallel "${JOBS}"
}

verify_install_and_consumers() {
  local build_dir="$1" deps_flag_name="$2" deps_flag_val="$3" enable_signal="$4" cfg="$5"
  local tag prefix fc_dir fp_dir neg_log hdr leaked_root
  tag="$(basename "${build_dir}")"
  prefix="${WORK_DIR}/pkg-${tag}"

  # --config "${cfg}" must match CMAKE_BUILD_TYPE for tflm-targets-<cfg>.cmake.
  readable_run cmake --install "${build_dir}" --prefix "${prefix}" --config "${cfg}"

  for hdr in \
    tensorflow/lite/micro/micro_interpreter.h \
    tensorflow/lite/micro/c/common.h \
    tensorflow/lite/micro/schema/schema_generated.h \
    tflm/third_party/flatbuffers/flatbuffers.h; do
    [[ -f "${prefix}/include/${hdr}" ]] || die "Missing installed header '${hdr}'"
  done
  for hdr in \
    tensorflow/lite/micro/testing/micro_test.h \
    tensorflow/lite/micro/kernels/kernel_runner.h \
    tensorflow/lite/micro/kernels/xtensa/xtensa.h \
    signal/src/kiss_fft_wrappers/kiss_fft_int16.h; do
    [[ ! -e "${prefix}/include/${hdr}" ]] || die "Forbidden private header '${hdr}' leaked"
  done

  local cmake_files=()
  while IFS= read -r -d '' f; do
    cmake_files+=("${f}")
  done < <(find "${prefix}" \( -name 'tflm-targets*.cmake' -o -name 'tflm-config*.cmake' \) -print0)
  [[ "${#cmake_files[@]}" -gt 0 ]] || die "Missing tflm CMake config/targets under ${prefix}"
  for leaked_root in "${ROOT_DIR}" "${build_dir}" "${DEPS_DIR}"; do
    if [[ -n "${leaked_root}" ]] && grep -Fq "${leaked_root}" "${cmake_files[@]}"; then
      die "Exported CMake package leaked host path '${leaked_root}'"
    fi
  done

  fc_dir="${WORK_DIR}/smoke-fc-${tag}"
  (
    unset TFLM_DEPS_DIR TFLM_FLATBUFFERS_DIR TFLM_KISSFFT_DIR
    readable_run cmake -S "${ROOT_DIR}/cmake/tests" -B "${fc_dir}" \
      -DCMAKE_BUILD_TYPE="${cfg}" \
      -DTFLM_SOURCE_ROOT="${ROOT_DIR}" \
      -DTFLM_SMOKE_ENABLE_SIGNAL="${enable_signal}" \
      -DTFLM_TEST_MODEL_FILE="${MODEL_FILE}" \
      -DTFLM_WARNINGS_AS_ERRORS=ON \
      "-D${deps_flag_name}=${deps_flag_val}"
    readable_run cmake --build "${fc_dir}" --parallel "${JOBS}"
    readable_run "${fc_dir}/consumer_app"
  )

  fp_dir="${WORK_DIR}/smoke-fp-${tag}"
  (
    unset TFLM_DEPS_DIR TFLM_FLATBUFFERS_DIR TFLM_KISSFFT_DIR
    readable_run cmake -S "${ROOT_DIR}/cmake/tests" -B "${fp_dir}" \
      -DCMAKE_BUILD_TYPE="${cfg}" \
      -DCMAKE_PREFIX_PATH="${prefix}" \
      -DTFLM_SMOKE_ENABLE_SIGNAL="${enable_signal}" \
      -DTFLM_TEST_MODEL_FILE="${MODEL_FILE}"
    readable_run cmake --build "${fp_dir}" --parallel "${JOBS}"
    readable_run "${fp_dir}/consumer_app"
  )

  if [[ "${enable_signal}" == "OFF" ]]; then
    neg_log="${WORK_DIR}/smoke-neg-${tag}.log"
    if (
      unset TFLM_DEPS_DIR TFLM_FLATBUFFERS_DIR TFLM_KISSFFT_DIR
      cmake -S "${ROOT_DIR}/cmake/tests" -B "${WORK_DIR}/smoke-neg-${tag}" \
        -DCMAKE_PREFIX_PATH="${prefix}" \
        -DTFLM_SMOKE_ENABLE_SIGNAL=ON \
        -DTFLM_TEST_MODEL_FILE="${MODEL_FILE}"
    ) >"${neg_log}" 2>&1; then
      cat "${neg_log}" >&2
      die "Expected COMPONENTS signal check to fail on core-only package"
    fi
    tr -s '[:space:]' ' ' <"${neg_log}" | grep -q 'considered to be NOT FOUND' || {
      cat "${neg_log}" >&2
      die "Missing 'considered to be NOT FOUND' diagnostic"
    }
  fi
}

run_ci() {
  ensure_full_deps
  local build_dir="${WORK_DIR}/build-ci"
  run_cmake_preset ci "${build_dir}" -DTFLM_DEPS_DIR="${DEPS_DIR}"
  verify_install_and_consumers "${build_dir}" TFLM_DEPS_DIR "${DEPS_DIR}" ON RelWithDebInfo
  run_interop_checks
}

run_compression_debug() {
  ensure_full_deps
  local build_dir="${WORK_DIR}/build-compression-debug"
  run_cmake_preset debug "${build_dir}" -DTFLM_DEPS_DIR="${DEPS_DIR}" -DTFLM_ENABLE_COMPRESSION=ON
  readable_run ctest --test-dir "${build_dir}" \
    --output-on-failure --no-tests=error -R "^tflm_kernel_decompress_test$"
}

run_core_only() {
  if [[ ! -d "${DEPS_DIR}/flatbuffers" ]]; then
    readable_run python3 "${ROOT_DIR}/cmake/download_deps.py" --deps flatbuffers "${DEPS_DIR}"
  fi
  local build_dir="${WORK_DIR}/build-core-only"
  (
    unset TFLM_DEPS_DIR TFLM_FLATBUFFERS_DIR TFLM_KISSFFT_DIR
    run_cmake_preset core-only "${build_dir}" \
      -DTFLM_FLATBUFFERS_DIR="${DEPS_DIR}/flatbuffers" \
      -DTFLM_WARNINGS_AS_ERRORS=ON
    verify_install_and_consumers "${build_dir}" \
      TFLM_FLATBUFFERS_DIR "${DEPS_DIR}/flatbuffers" OFF MinSizeRel
  )
}

run_interop_checks() {
  ensure_full_deps

  # 1. External FlatBuffers find_package + find_dependency(flatbuffers) end-to-end.
  readable_run cmake -S "${DEPS_DIR}/flatbuffers" -B "${WORK_DIR}/fb-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLATBUFFERS_BUILD_FLATLIB=ON \
    -DFLATBUFFERS_BUILD_FLATC=OFF \
    -DFLATBUFFERS_BUILD_TESTS=OFF
  readable_run cmake --build "${WORK_DIR}/fb-build" --parallel "${JOBS}"
  readable_run cmake --install "${WORK_DIR}/fb-build" --prefix "${WORK_DIR}/fb-pkg" --config Release

  (
    unset TFLM_DEPS_DIR TFLM_FLATBUFFERS_DIR TFLM_KISSFFT_DIR
    readable_run cmake -S "${ROOT_DIR}" -B "${WORK_DIR}/tflm-extfb-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${WORK_DIR}/fb-pkg" \
      -DTFLM_ENABLE_SIGNAL=OFF \
      -DTFLM_BUILD_TESTS=OFF \
      -DTFLM_WARNINGS_AS_ERRORS=ON
    readable_run cmake --build "${WORK_DIR}/tflm-extfb-build" --parallel "${JOBS}"
    readable_run cmake --install "${WORK_DIR}/tflm-extfb-build" \
      --prefix "${WORK_DIR}/tflm-extfb-pkg" --config Release
    [[ ! -d "${WORK_DIR}/tflm-extfb-pkg/include/tflm/third_party" ]] \
      || die "Bundled FlatBuffers headers installed alongside external flatbuffers"
    readable_run cmake -S "${ROOT_DIR}/cmake/tests" -B "${WORK_DIR}/tflm-extfb-consumer" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${WORK_DIR}/tflm-extfb-pkg;${WORK_DIR}/fb-pkg" \
      -DTFLM_SMOKE_ENABLE_SIGNAL=OFF \
      -DTFLM_TEST_MODEL_FILE="${MODEL_FILE}"
    readable_run cmake --build "${WORK_DIR}/tflm-extfb-consumer" --parallel "${JOBS}"
    readable_run "${WORK_DIR}/tflm-extfb-consumer/consumer_app"
  )

  # 2. Mock Zephyr (zephyr_interface) + R-2 ABI inheritance check.
  mkdir -p "${WORK_DIR}/interop-zephyr"
  cat >"${WORK_DIR}/interop-zephyr/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.23)
project(interop_zephyr LANGUAGES CXX)
add_library(zephyr_interface INTERFACE)
add_subdirectory("${TFLM_ROOT}" tflm)
add_library(zephyr_app STATIC "${TFLM_ROOT}/cmake/tests/consumer_main.cc")
target_link_libraries(zephyr_app PRIVATE tflm::micro)
tflm_target_embed_assets(zephyr_app PRIVATE "${TFLM_MODEL_FILE}")
get_target_property(_libs tflm_micro INTERFACE_LINK_LIBRARIES)
if(NOT "$<BUILD_INTERFACE:zephyr_interface>" IN_LIST _libs OR "m" IN_LIST _libs)
  message(FATAL_ERROR "Unexpected Zephyr INTERFACE_LINK_LIBRARIES='${_libs}'")
endif()
get_target_property(_defs tflm_micro INTERFACE_COMPILE_DEFINITIONS)
foreach(_req IN ITEMS
    FLATBUFFERS_LOCALE_INDEPENDENT=0 FLATBUFFERS_PREFER_PRINTF=1
    TF_LITE_MCU_DEBUG_LOG TFLM_REMOVE_VIRTUAL_DELETE)
  if(NOT "${_req}" IN_LIST _defs)
    message(FATAL_ERROR "Missing '${_req}' in Zephyr INTERFACE_COMPILE_DEFINITIONS='${_defs}'")
  endif()
endforeach()
get_target_property(_assets zephyr_app _TFLM_EMBEDDED_ASSETS)
list(GET _assets 0 _arr_tgt)
get_target_property(_arr_libs "${_arr_tgt}" LINK_LIBRARIES)
if(NOT "tflm::micro" IN_LIST _arr_libs)
  message(FATAL_ERROR "Expected tflm::micro in ${_arr_tgt} LINK_LIBRARIES, got '${_arr_libs}'")
endif()
EOF
  readable_run cmake -S "${WORK_DIR}/interop-zephyr" -B "${WORK_DIR}/interop-zephyr/build" \
    -DCMAKE_SYSTEM_NAME=Generic \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DTFLM_ROOT="${ROOT_DIR}" \
    -DTFLM_MODEL_FILE="${MODEL_FILE}" \
    -DTFLM_FLATBUFFERS_DIR="${DEPS_DIR}/flatbuffers"

  # 3. Backend source-set specialization (R-1) & TFLM_SYSTEM_TARGET guard (R-4).
  mkdir -p "${WORK_DIR}/interop-backends"
  cat >"${WORK_DIR}/interop-backends/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.23)
project(interop_backends LANGUAGES CXX)
function(expect_match list_var suffix should_exist)
  set(_found FALSE)
  foreach(_item IN LISTS ${list_var})
    if(_item MATCHES "${suffix}$")
      set(_found TRUE)
    endif()
  endforeach()
  if(NOT _found STREQUAL should_exist)
    message(FATAL_ERROR "Mismatch for '${suffix}' (expected ${should_exist}) in ${list_var}")
  endif()
endfunction()
if(BACKEND_CASE STREQUAL "cmsis_ethos")
  add_library(cmsis-nn INTERFACE)
  add_library(ethosu_core_driver INTERFACE)
  set(TFLM_OPTIMIZED_KERNELS cmsis_nn)
  set(TFLM_CO_PROCESSOR ethos_u)
  add_subdirectory("${TFLM_ROOT}" tflm)
  get_target_property(_srcs tflm_micro SOURCES)
  expect_match(_srcs "/kernels/cmsis_nn/conv\\.cc" TRUE)
  expect_match(_srcs "/kernels/ethos_u/ethosu\\.cc" TRUE)
  expect_match(_srcs "/kernels/conv\\.cc" FALSE)
elseif(BACKEND_CASE STREQUAL "xtensa")
  add_library(xtensa_sdk INTERFACE)
  set(TFLM_OPTIMIZED_KERNELS xtensa)
  set(TFLM_BACKEND_TARGETS xtensa_sdk)
  set(TFLM_ENABLE_SIGNAL ON)
  add_subdirectory("${TFLM_ROOT}" tflm)
  get_target_property(_srcs tflm_signal SOURCES)
  expect_match(_srcs "/signal/micro/kernels/xtensa/rfft_int16\\.cc" TRUE)
  expect_match(_srcs "/signal/micro/kernels/xtensa/fft_auto_scale_kernel\\.cc" TRUE)
  expect_match(_srcs "/signal/src/rfft_int16\\.cc" FALSE)
  expect_match(_srcs "/signal/micro/kernels/fft_auto_scale_kernel\\.cc" FALSE)
  expect_match(_srcs "/signal/src/window\\.cc" TRUE)
  expect_match(_srcs "/signal/micro/kernels/window\\.cc" TRUE)
elseif(BACKEND_CASE STREQUAL "hexagon")
  add_library(hexagon_sdk INTERFACE)
  set(TFLM_OPTIMIZED_KERNELS hexagon)
  set(TFLM_BACKEND_TARGETS hexagon_sdk)
  set(TFLM_ENABLE_SIGNAL ON)
  add_subdirectory("${TFLM_ROOT}" tflm)
  get_target_property(_srcs tflm_signal SOURCES)
  expect_match(_srcs "/signal/micro/kernels/hexagon/rfft_int16\\.cc" TRUE)
  expect_match(_srcs "/signal/micro/kernels/hexagon/square_root_32\\.cc" TRUE)
  expect_match(_srcs "/signal/micro/kernels/hexagon/hexagon_square_root\\.S" TRUE)
  expect_match(_srcs "/signal/src/rfft_int16\\.cc" FALSE)
  expect_match(_srcs "/signal/src/square_root_32\\.cc" FALSE)
endif()
EOF
  local backend_case r4_log="${WORK_DIR}/interop-r4.log"
  for backend_case in cmsis_ethos xtensa hexagon; do
    readable_run cmake -S "${WORK_DIR}/interop-backends" \
      -B "${WORK_DIR}/interop-backends/build-${backend_case}" \
      -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
      -DTFLM_ROOT="${ROOT_DIR}" \
      -DTFLM_DEPS_DIR="${DEPS_DIR}" \
      -DBACKEND_CASE="${backend_case}"
  done

  if cmake -S "${ROOT_DIR}" -B "${WORK_DIR}/interop-r4-build" \
    -DTFLM_DEPS_DIR="${DEPS_DIR}" \
    -DTFLM_ENABLE_SIGNAL=OFF \
    -DTFLM_BUILD_TESTS=OFF \
    -DTFLM_SYSTEM_TARGET=kernels >"${r4_log}" 2>&1; then
    cat "${r4_log}" >&2
    die "Expected TFLM_SYSTEM_TARGET=kernels to fail configure"
  fi
  grep -q '_tflm_specialize_sources' "${r4_log}" || {
    cat "${r4_log}" >&2
    die "Missing '_tflm_specialize_sources' diagnostic for TFLM_SYSTEM_TARGET=kernels"
  }
}

case "${MODE}" in
  ci) run_ci ;;
  compression_debug|compression) run_compression_debug ;;
  core_only|core-only) run_core_only ;;
  interop) run_interop_checks ;;
  all)
    run_ci
    run_compression_debug
    run_core_only
    ;;
  *)
    echo "Unknown mode '${MODE}'. Expected: all, ci, compression_debug, core_only, or interop." >&2
    exit 2
    ;;
esac
