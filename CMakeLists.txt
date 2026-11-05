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

cmake_minimum_required(VERSION 3.23)
project(tflm VERSION 0.1.0 LANGUAGES CXX)

include(CMakeDependentOption)
include("${PROJECT_SOURCE_DIR}/cmake/tflm_codegen.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/tflm_sources.cmake")

# --- 1. Options & Validation --------------------------------------------------
option(TFLM_ENABLE_SIGNAL "Build tflm::signal (requires KissFFT)" ${PROJECT_IS_TOP_LEVEL})
option(TFLM_ENABLE_COMPRESSION "Enable TFLM weight compression support" OFF)
option(TFLM_STRIP_ERROR_STRINGS "Define TF_LITE_STRIP_ERROR_STRINGS" OFF)
option(TFLM_EXCLUDE_SYSTEM_IMPL "Exclude default debug_log/micro_time/system_setup" OFF)
option(TFLM_REMOVE_VIRTUAL_DELETE "Define TFLM_REMOVE_VIRTUAL_DELETE" ${CMAKE_CROSSCOMPILING})
option(TFLM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(_tflm_default_link_libm ON)
if(CMAKE_CXX_COMPILER_ID STREQUAL "ARMClang" OR MSVC OR TARGET zephyr_interface)
  set(_tflm_default_link_libm OFF)
endif()
option(TFLM_LINK_LIBM "Link -lm publicly on tflm::micro" ${_tflm_default_link_libm})
option(TFLM_INSTALL "Generate install and CMake package export rules" ${PROJECT_IS_TOP_LEVEL})

set(TFLM_OPTIMIZE_KERNELS_FOR "SPEED" CACHE STRING
  "Optimize TFLM kernel scratch-buffer tradeoff (SPEED or SIZE)")
set_property(CACHE TFLM_OPTIMIZE_KERNELS_FOR PROPERTY STRINGS "SPEED" "SIZE")
set(TFLM_OPTIMIZED_KERNELS "" CACHE STRING "Optimized kernel backend (cmsis_nn, xtensa, hexagon)")
set_property(CACHE TFLM_OPTIMIZED_KERNELS PROPERTY STRINGS "" cmsis_nn xtensa hexagon)
set(TFLM_CO_PROCESSOR "" CACHE STRING "Co-processor kernel backend (ethos_u)")
set_property(CACHE TFLM_CO_PROCESSOR PROPERTY STRINGS "" ethos_u)
set(TFLM_SYSTEM_TARGET "" CACHE STRING "Platform override directory under tensorflow/lite/micro/")
set(TFLM_BACKEND_TARGETS "" CACHE STRING
  "Parent CMake target(s) providing vendor DSP/NPU headers, defines, and libs")
set(TFLM_EXTERNAL_KERNEL_DIR "" CACHE STRING
  "Out-of-tree directory(ies) of custom/vendor kernel source overrides")
set(TFLM_DEPS_DIR "$ENV{TFLM_DEPS_DIR}" CACHE PATH
  "Base directory containing local FlatBuffers/KissFFT")
set(TFLM_FLATBUFFERS_DIR "" CACHE PATH "Local path to FlatBuffers root or include directory")
set(TFLM_FLATC_EXECUTABLE "" CACHE FILEPATH "Path to flatc executable to regenerate *_generated.h")
set(TFLM_KISSFFT_DIR "" CACHE PATH "Local path to KissFFT directory")

option(TFLM_BUILD_TESTS "Build TFLM unit tests" ${PROJECT_IS_TOP_LEVEL})
cmake_dependent_option(TFLM_BUILD_EXAMPLES
  "Build TFLM example tests" ${PROJECT_IS_TOP_LEVEL} "TFLM_BUILD_TESTS" OFF)
cmake_dependent_option(TFLM_BUILD_INTEGRATION_TESTS
  "Build SeaNet integration tests" OFF "TFLM_BUILD_TESTS;TFLM_ENABLE_SIGNAL" OFF)
set(TFLM_EXCLUDED_TESTS "" CACHE STRING
  "Semicolon-separated list of TFLM test target names to exclude")

if(NOT TFLM_OPTIMIZE_KERNELS_FOR MATCHES "^(SPEED|SIZE)$")
  message(FATAL_ERROR
    "Invalid TFLM_OPTIMIZE_KERNELS_FOR='${TFLM_OPTIMIZE_KERNELS_FOR}'. Expected 'SPEED' or 'SIZE'.")
endif()
foreach(_tgt IN LISTS TFLM_BACKEND_TARGETS)
  if(NOT TARGET "${_tgt}")
    message(FATAL_ERROR "TFLM_BACKEND_TARGETS entry '${_tgt}' is not an existing CMake target.")
  endif()
endforeach()
if(TFLM_OPTIMIZED_KERNELS)
  if(NOT TFLM_OPTIMIZED_KERNELS MATCHES "^(cmsis_nn|xtensa|hexagon)$")
    message(FATAL_ERROR "Unsupported TFLM_OPTIMIZED_KERNELS='${TFLM_OPTIMIZED_KERNELS}'.")
  elseif(TFLM_OPTIMIZED_KERNELS STREQUAL "cmsis_nn")
    if(NOT TFLM_BACKEND_TARGETS AND NOT TARGET cmsis-nn)
      message(FATAL_ERROR
        "TFLM_OPTIMIZED_KERNELS='cmsis_nn' requires target 'cmsis-nn' or TFLM_BACKEND_TARGETS.")
    endif()
  elseif(NOT TFLM_BACKEND_TARGETS)
    message(FATAL_ERROR
      "TFLM_OPTIMIZED_KERNELS='${TFLM_OPTIMIZED_KERNELS}' requires -DTFLM_BACKEND_TARGETS=<target>.")
  endif()
endif()
if(TFLM_CO_PROCESSOR)
  if(NOT TFLM_CO_PROCESSOR STREQUAL "ethos_u")
    message(FATAL_ERROR "Unsupported TFLM_CO_PROCESSOR='${TFLM_CO_PROCESSOR}'.")
  elseif(NOT TFLM_BACKEND_TARGETS AND NOT TARGET ethosu_core_driver)
    message(FATAL_ERROR
      "TFLM_CO_PROCESSOR='ethos_u' requires target 'ethosu_core_driver' or TFLM_BACKEND_TARGETS.")
  endif()
endif()
if(TFLM_ENABLE_COMPRESSION AND TFLM_OPTIMIZED_KERNELS
   AND NOT TFLM_OPTIMIZED_KERNELS STREQUAL "xtensa")
  message(FATAL_ERROR
    "TFLM_ENABLE_COMPRESSION=ON requires reference kernels or TFLM_OPTIMIZED_KERNELS='xtensa'.")
endif()
if(TFLM_SYSTEM_TARGET
   AND NOT IS_DIRECTORY "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/${TFLM_SYSTEM_TARGET}")
  message(FATAL_ERROR
    "Unknown TFLM_SYSTEM_TARGET='${TFLM_SYSTEM_TARGET}' under tensorflow/lite/micro/.")
endif()
set(_tflm_ext_kernel_dirs "")
foreach(_dir IN LISTS TFLM_EXTERNAL_KERNEL_DIR)
  cmake_path(ABSOLUTE_PATH _dir
    BASE_DIRECTORY "${CMAKE_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _abs_dir)
  if(NOT IS_DIRECTORY "${_abs_dir}")
    message(FATAL_ERROR "TFLM_EXTERNAL_KERNEL_DIR='${_dir}' does not exist.")
  endif()
  list(APPEND _tflm_ext_kernel_dirs "${_abs_dir}")
endforeach()

# --- 2. Internal Helpers & Dependencies ---------------------------------------
function(_tflm_apply_common_compile_options target_name)
  set(_gnu_like GNU,Clang,AppleClang,ARMClang)
  set(_clang_like Clang,AppleClang,ARMClang)
  set(_gnu_flags
    -fno-asynchronous-unwind-tables -fno-unwind-tables
    -ffunction-sections -fdata-sections
    -Wall -Wsign-compare -Wdouble-promotion -Wvla -Wno-unused-parameter)
  set(_cxx_only_flags -fno-exceptions -fno-rtti -fno-threadsafe-statics -Wnon-virtual-dtor)
  set(_msvc_werr "")
  if(TFLM_WARNINGS_AS_ERRORS)
    list(APPEND _gnu_flags -Werror)
    set(_msvc_werr $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX> $<$<COMPILE_LANG_AND_ID:C,MSVC>:/WX>)
  endif()
  target_compile_options(${target_name} PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/GR->
    ${_msvc_werr}
    "$<$<COMPILE_LANG_AND_ID:CXX,${_gnu_like}>:${_gnu_flags};${_cxx_only_flags}>"
    "$<$<COMPILE_LANG_AND_ID:C,${_gnu_like}>:${_gnu_flags};-Wimplicit-function-declaration>"
    $<$<COMPILE_LANG_AND_ID:CXX,${_clang_like}>:-fno-sanitize=alignment>
    $<$<COMPILE_LANG_AND_ID:C,${_clang_like}>:-fno-sanitize=alignment>)
endfunction()

function(_tflm_specialize_sources out_var)
  cmake_parse_arguments(ARG "REPLACE_ONLY" "OVERRIDE_DIR" "BASE_SRCS;OVERRIDE_SRCS" ${ARGN})
  set(_ovrs ${ARG_OVERRIDE_SRCS})
  if(ARG_OVERRIDE_DIR AND IS_DIRECTORY "${ARG_OVERRIDE_DIR}")
    file(GLOB _dir_srcs CONFIGURE_DEPENDS
      "${ARG_OVERRIDE_DIR}/*.cc" "${ARG_OVERRIDE_DIR}/*.c"
      "${ARG_OVERRIDE_DIR}/*.S" "${ARG_OVERRIDE_DIR}/*.s")
    list(FILTER _dir_srcs EXCLUDE REGEX "_test\\.(cc|c)$")
    list(APPEND _ovrs ${_dir_srcs})
  endif()
  if(NOT _ovrs)
    if(ARG_REPLACE_ONLY)
      message(FATAL_ERROR "_tflm_specialize_sources(${out_var}): no override sources found.")
    endif()
    set(${out_var} ${ARG_BASE_SRCS} PARENT_SCOPE)
    return()
  endif()
  set(_ovr_names "")
  foreach(_f IN LISTS _ovrs)
    cmake_path(GET _f FILENAME _name)
    if(_name IN_LIST _ovr_names)
      message(FATAL_ERROR
        "_tflm_specialize_sources(${out_var}): duplicate basename '${_name}' in overrides.")
    endif()
    list(APPEND _ovr_names "${_name}")
  endforeach()
  set(_matched "")
  set(_res ${_ovrs})
  foreach(_f IN LISTS ARG_BASE_SRCS)
    cmake_path(GET _f FILENAME _name)
    if(_name IN_LIST _ovr_names)
      if(_name IN_LIST _matched)
        message(FATAL_ERROR
          "_tflm_specialize_sources(${out_var}): override '${_name}' matches multiple files in BASE_SRCS.")
      endif()
      list(APPEND _matched "${_name}")
    else()
      list(APPEND _res "${_f}")
    endif()
  endforeach()
  if(ARG_REPLACE_ONLY)
    list(LENGTH _ovr_names _n_ovr)
    list(LENGTH _matched _n_matched)
    if(NOT _n_ovr EQUAL _n_matched)
      list(REMOVE_ITEM _ovr_names ${_matched})
      message(FATAL_ERROR
        "_tflm_specialize_sources(${out_var}): '${_ovr_names}' did not match any file in BASE_SRCS.")
    endif()
  endif()
  set(${out_var} ${_res} PARENT_SCOPE)
endfunction()

function(_tflm_resolve_dep_dir var subdir out_var)
  set(_res "")
  set(_raw "${${var}}")
  set(_deps "${TFLM_DEPS_DIR}")
  if(_raw)
    cmake_path(ABSOLUTE_PATH _raw
      BASE_DIRECTORY "${CMAKE_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _res)
    if(NOT IS_DIRECTORY "${_res}")
      message(FATAL_ERROR "${var}='${_raw}' does not exist.")
    endif()
  elseif(_deps)
    cmake_path(ABSOLUTE_PATH _deps
      BASE_DIRECTORY "${CMAKE_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _deps)
    if(NOT IS_DIRECTORY "${_deps}")
      message(FATAL_ERROR "TFLM_DEPS_DIR='${TFLM_DEPS_DIR}' does not exist.")
    elseif(IS_DIRECTORY "${_deps}/${subdir}")
      set(_res "${_deps}/${subdir}")
    endif()
  endif()
  set(${out_var} "${_res}" PARENT_SCOPE)
endfunction()

function(_tflm_find_flatbuffers_target out_var)
  foreach(_cand IN ITEMS
      flatbuffers::flatbuffers flatbuffers
      flatbuffers::flatbuffers_shared tflm_flatbuffers_headers)
    if(TARGET "${_cand}")
      set(${out_var} "${_cand}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

_tflm_find_flatbuffers_target(_tflm_fb_target)
if(NOT _tflm_fb_target)
  _tflm_resolve_dep_dir(TFLM_FLATBUFFERS_DIR flatbuffers _fb_dir)
  if(_fb_dir)
    set(_fb_inc "${_fb_dir}/include")
    if(NOT EXISTS "${_fb_inc}/flatbuffers/flatbuffers.h")
      set(_fb_inc "${_fb_dir}")
    endif()
    if(NOT EXISTS "${_fb_inc}/flatbuffers/flatbuffers.h")
      message(FATAL_ERROR
        "FlatBuffers directory '${_fb_dir}' is missing '[include/]flatbuffers/flatbuffers.h'.")
    endif()
    add_library(tflm_flatbuffers_headers INTERFACE)
    set_target_properties(tflm_flatbuffers_headers PROPERTIES
      EXPORT_NAME flatbuffers_headers
      _TFLM_FB_INC "${_fb_inc}")
    target_include_directories(tflm_flatbuffers_headers SYSTEM INTERFACE
      "$<BUILD_INTERFACE:${_fb_inc}>")
  else()
    find_package(flatbuffers CONFIG QUIET NAMES flatbuffers FlatBuffers Flatbuffers)
  endif()
  _tflm_find_flatbuffers_target(_tflm_fb_target)
endif()
if(NOT _tflm_fb_target)
  message(FATAL_ERROR
    "Missing required dependency 'flatbuffers' "
    "(provide target, TFLM_FLATBUFFERS_DIR, TFLM_DEPS_DIR, or package).")
endif()

if(TFLM_ENABLE_SIGNAL AND NOT TARGET tflm::kissfft)
  if(TARGET kissfft)
    add_library(tflm::kissfft ALIAS kissfft)
  else()
    _tflm_resolve_dep_dir(TFLM_KISSFFT_DIR kissfft _kfft_dir)
    if(NOT _kfft_dir)
      message(FATAL_ERROR
        "Missing required dependency 'kissfft' "
        "(provide target tflm::kissfft, TFLM_KISSFFT_DIR, or TFLM_DEPS_DIR).")
    endif()
    if(NOT EXISTS "${_kfft_dir}/kiss_fft.h")
      message(FATAL_ERROR
        "KissFFT directory '${_kfft_dir}' is missing 'kiss_fft.h' (KissFFT >= 131.1.0 required).")
    endif()
    add_library(tflm_kissfft INTERFACE)
    target_include_directories(tflm_kissfft SYSTEM INTERFACE "$<BUILD_INTERFACE:${_kfft_dir}>")
    add_library(tflm::kissfft ALIAS tflm_kissfft)
  endif()
endif()

# --- 3. Source Specialization & Library Targets (tflm::micro, tflm::signal) ---
set(_tflm_kernel_srcs ${TFLM_MICRO_KERNEL_SRCS})
set(_tflm_micro_hdr_files ${TFLM_MICRO_PUBLIC_HDRS})
set(_tflm_backend_defs "")
foreach(_backend IN ITEMS "${TFLM_OPTIMIZED_KERNELS}" "${TFLM_CO_PROCESSOR}")
  if(_backend)
    string(TOUPPER "${_backend}" _upper)
    _tflm_specialize_sources(_tflm_kernel_srcs
      BASE_SRCS ${_tflm_kernel_srcs} OVERRIDE_SRCS ${TFLM_MICRO_KERNEL_SRCS_${_upper}})
    list(APPEND _tflm_micro_hdr_files ${TFLM_MICRO_KERNEL_HDRS_${_upper}})
    list(APPEND _tflm_backend_defs "${_upper}")
    if(_backend STREQUAL "ethos_u")
      list(APPEND _tflm_backend_defs ETHOSU)
    endif()
  endif()
endforeach()
foreach(_ext_dir IN LISTS _tflm_ext_kernel_dirs)
  _tflm_specialize_sources(_tflm_kernel_srcs
    BASE_SRCS ${_tflm_kernel_srcs} OVERRIDE_DIR "${_ext_dir}")
endforeach()

set(_tflm_system_srcs "")
if(NOT TFLM_EXCLUDE_SYSTEM_IMPL)
  set(_tflm_system_srcs ${TFLM_SYSTEM_DEFAULT_SRCS})
  if(TFLM_SYSTEM_TARGET)
    set(_sys_dir "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/${TFLM_SYSTEM_TARGET}")
    _tflm_specialize_sources(_tflm_system_srcs REPLACE_ONLY
      BASE_SRCS ${_tflm_system_srcs} OVERRIDE_DIR "${_sys_dir}")
    file(GLOB _sys_hdrs CONFIGURE_DEPENDS "${_sys_dir}/*.h")
    list(APPEND _tflm_micro_hdr_files ${_sys_hdrs})
  endif()
endif()

set(_tflm_signal_srcs "")
if(TFLM_ENABLE_SIGNAL)
  set(_tflm_signal_srcs ${TFLM_SIGNAL_SRCS})
  set(_sig_opt_dir "${PROJECT_SOURCE_DIR}/signal/micro/kernels/${TFLM_OPTIMIZED_KERNELS}")
  if(TFLM_OPTIMIZED_KERNELS AND IS_DIRECTORY "${_sig_opt_dir}")
    _tflm_specialize_sources(_tflm_signal_srcs
      BASE_SRCS ${_tflm_signal_srcs} OVERRIDE_DIR "${_sig_opt_dir}")
  endif()
endif()

set(_selected_srcs ${_tflm_kernel_srcs} ${_tflm_system_srcs} ${_tflm_signal_srcs})
if(_selected_srcs MATCHES "\\.c(;|$)")
  enable_language(C)
endif()
if(_selected_srcs MATCHES "\\.[Ss](;|$)")
  enable_language(ASM)
endif()

add_library(tflm_micro STATIC
  ${TFLM_MICRO_CORE_SRCS} ${_tflm_system_srcs} ${_tflm_kernel_srcs})
set_target_properties(tflm_micro PROPERTIES EXPORT_NAME micro)
add_library(tflm::micro ALIAS tflm_micro)
target_compile_features(tflm_micro PUBLIC cxx_std_17)
_tflm_apply_common_compile_options(tflm_micro)

if(TFLM_FLATC_EXECUTABLE)
  set(_fb_gen "${PROJECT_BINARY_DIR}/flatbuffers_gen")
  set(_fb_hdrs "")
  foreach(_stem IN ITEMS schema/schema compression/metadata)
    set(_fbs "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/${_stem}.fbs")
    set(_hdr "${_fb_gen}/tensorflow/lite/micro/${_stem}_generated.h")
    cmake_path(GET _hdr PARENT_PATH _dir)
    add_custom_command(OUTPUT "${_hdr}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_dir}"
      COMMAND "${TFLM_FLATC_EXECUTABLE}"
        -c --no-union-value-namespacing --gen-object-api -o "${_dir}" "${_fbs}"
      DEPENDS "${_fbs}" VERBATIM)
    list(REMOVE_ITEM _tflm_micro_hdr_files
      "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/${_stem}_generated.h")
    list(APPEND _fb_hdrs "${_hdr}")
  endforeach()
  # flatbuffers_schema must precede HEADERS so regenerated *_generated.h shadow checked-in copies.
  target_sources(tflm_micro PUBLIC
    FILE_SET flatbuffers_schema TYPE HEADERS BASE_DIRS "${_fb_gen}" FILES ${_fb_hdrs})
endif()
list(REMOVE_DUPLICATES _tflm_micro_hdr_files)
target_sources(tflm_micro PUBLIC
  FILE_SET HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}" FILES ${_tflm_micro_hdr_files})
foreach(_ext_dir IN LISTS _tflm_ext_kernel_dirs)
  target_include_directories(tflm_micro PUBLIC "$<BUILD_INTERFACE:${_ext_dir}>")
endforeach()

set(_pub_defs FLATBUFFERS_LOCALE_INDEPENDENT=0 FLATBUFFERS_PREFER_PRINTF=1 ${_tflm_backend_defs})
set(_priv_defs KERNELS_OPTIMIZED_FOR_${TFLM_OPTIMIZE_KERNELS_FOR})
if(TFLM_ENABLE_COMPRESSION)
  list(APPEND _pub_defs USE_TFLM_COMPRESSION=1)
endif()
if(TFLM_STRIP_ERROR_STRINGS)
  list(APPEND _pub_defs TF_LITE_STRIP_ERROR_STRINGS)
endif()
if(TFLM_REMOVE_VIRTUAL_DELETE)
  list(APPEND _pub_defs TFLM_REMOVE_VIRTUAL_DELETE)
endif()
if(CMAKE_CROSSCOMPILING OR TFLM_SYSTEM_TARGET)
  list(APPEND _pub_defs TF_LITE_MCU_DEBUG_LOG)
else()
  list(APPEND _priv_defs TF_LITE_USE_CTIME)
endif()
target_compile_definitions(tflm_micro PUBLIC ${_pub_defs} PRIVATE ${_priv_defs})

set(_pub_libs ${_tflm_fb_target})
if(TARGET zephyr_interface)
  list(APPEND _pub_libs "$<BUILD_INTERFACE:zephyr_interface>")
endif()
if(TFLM_LINK_LIBM)
  list(APPEND _pub_libs m)
endif()
target_link_libraries(tflm_micro PUBLIC ${_pub_libs})

if(TFLM_ENABLE_SIGNAL)
  add_library(tflm_signal STATIC ${_tflm_signal_srcs})
  set_target_properties(tflm_signal PROPERTIES EXPORT_NAME signal)
  add_library(tflm::signal ALIAS tflm_signal)
  _tflm_apply_common_compile_options(tflm_signal)
  target_sources(tflm_signal PUBLIC
    FILE_SET HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}" FILES ${TFLM_SIGNAL_PUBLIC_HDRS})
  target_link_libraries(tflm_signal
    PUBLIC tflm::micro
    PRIVATE "$<BUILD_INTERFACE:tflm::kissfft>")
endif()

set(_tflm_backend_targets ${TFLM_BACKEND_TARGETS})
if(TFLM_OPTIMIZED_KERNELS STREQUAL "cmsis_nn" AND TARGET cmsis-nn)
  list(APPEND _tflm_backend_targets cmsis-nn)
endif()
if(TFLM_CO_PROCESSOR STREQUAL "ethos_u" AND TARGET ethosu_core_driver)
  list(APPEND _tflm_backend_targets ethosu_core_driver)
endif()
list(REMOVE_DUPLICATES _tflm_backend_targets)
if(_tflm_backend_targets)
  target_link_libraries(tflm_micro PRIVATE ${_tflm_backend_targets})
  if(TARGET tflm_signal)
    target_link_libraries(tflm_signal PRIVATE ${_tflm_backend_targets})
  endif()
endif()

# --- 4. Test Support & CTest Suite (tflm::test_support, tflm_add_cc_test) -----
add_library(tflm_test_support STATIC EXCLUDE_FROM_ALL ${TFLM_TEST_SUPPORT_SRCS})
add_library(tflm::test_support ALIAS tflm_test_support)
_tflm_apply_common_compile_options(tflm_test_support)
target_link_libraries(tflm_test_support PUBLIC tflm::micro)

function(tflm_add_cc_test)
  cmake_parse_arguments(ARG "" "NAME" "SRCS;ASSETS;DEPS;LABELS" ${ARGN})
  if(NOT ARG_NAME)
    message(FATAL_ERROR "tflm_add_cc_test requires NAME.")
  endif()
  string(REGEX REPLACE "^tflm_" "" _short "${ARG_NAME}")
  if(ARG_NAME IN_LIST TFLM_EXCLUDED_TESTS OR _short IN_LIST TFLM_EXCLUDED_TESTS)
    return()
  endif()
  if(TARGET "${ARG_NAME}")
    message(FATAL_ERROR "tflm_add_cc_test: duplicate test target '${ARG_NAME}'.")
  endif()
  add_executable(${ARG_NAME} ${ARG_SRCS})
  _tflm_apply_common_compile_options(${ARG_NAME})
  target_link_libraries(${ARG_NAME} PRIVATE tflm::test_support ${ARG_DEPS})
  if(ARG_ASSETS)
    tflm_target_embed_assets(${ARG_NAME} PRIVATE ${ARG_ASSETS})
  endif()
  if(NOT TFLM_STRIP_ERROR_STRINGS
     AND (NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR))
    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
    set_tests_properties(${ARG_NAME} PROPERTIES
      PASS_REGULAR_EXPRESSION "~~~ALL TESTS PASSED~~~"
      LABELS "${ARG_LABELS}")
  endif()
endfunction()

if(TFLM_BUILD_TESTS)
  enable_testing()
  if(TFLM_STRIP_ERROR_STRINGS)
    message(WARNING
      "TFLM_STRIP_ERROR_STRINGS=ON omits MicroPrintf strings; skipping CTest registration.")
  elseif(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    message(STATUS
      "Cross-compiling without CMAKE_CROSSCOMPILING_EMULATOR; skipping CTest registration.")
  endif()
  set(_py_mods "")
  if(TFLM_BUILD_EXAMPLES)
    list(APPEND _py_mods PIL)
  endif()
  if(TFLM_BUILD_INTEGRATION_TESTS)
    list(APPEND _py_mods numpy)
  endif()
  if(_py_mods)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    list(JOIN _py_mods ", " _py_imports)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import ${_py_imports}"
      RESULT_VARIABLE _py_rc OUTPUT_QUIET ERROR_QUIET)
    if(_py_rc)
      message(FATAL_ERROR
        "Python '${Python3_EXECUTABLE}' is missing (${_py_imports}); "
        "install via 'pip install Pillow numpy'.")
    endif()
  endif()
  tflm_register_tests()
endif()

# --- 5. Package Install (TFLM_INSTALL) ----------------------------------------
if(TFLM_INSTALL)
  include(CMakePackageConfigHelpers)
  include(GNUInstallDirs)
  set(_install_libs tflm_micro)
  if(TARGET tflm_signal)
    list(APPEND _install_libs tflm_signal)
  endif()
  set(_extra_sets "")
  if(TFLM_FLATC_EXECUTABLE)
    set(_extra_sets FILE_SET flatbuffers_schema)
  endif()
  install(TARGETS ${_install_libs} EXPORT tflmTargets FILE_SET HEADERS ${_extra_sets})
  set(_TFLM_BUNDLED_FLATBUFFERS FALSE)
  if(TARGET tflm_flatbuffers_headers)
    set(_TFLM_BUNDLED_FLATBUFFERS TRUE)
    get_target_property(_fb_inc tflm_flatbuffers_headers _TFLM_FB_INC)
    target_include_directories(tflm_flatbuffers_headers SYSTEM INTERFACE
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tflm/third_party>")
    install(TARGETS tflm_flatbuffers_headers EXPORT tflmTargets)
    install(DIRECTORY "${_fb_inc}/flatbuffers"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tflm/third_party"
      FILES_MATCHING PATTERN "*.h")
  endif()
  set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/tflm")
  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/tflm_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tflm-config.cmake"
    INSTALL_DESTINATION "${_cmake_dir}")
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tflm-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
  install(EXPORT tflmTargets
    FILE tflm-targets.cmake
    NAMESPACE tflm::
    DESTINATION "${_cmake_dir}")
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tflm-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tflm-config-version.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/tflm_codegen.cmake"
    "${PROJECT_SOURCE_DIR}/tensorflow/lite/micro/tools/generate_cc_arrays.py"
    DESTINATION "${_cmake_dir}")
endif()
