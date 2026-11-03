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

# Public model and test-asset C++ array codegen module for TFLM:
#   tflm_target_embed_assets(<target> <PRIVATE|PUBLIC> [FILES] <asset>...)
include_guard(GLOBAL)

function(tflm_target_embed_assets target_name visibility)
  if(TARGET "${target_name}")
    get_target_property(_type "${target_name}" TYPE)
  endif()
  if(NOT TARGET "${target_name}" OR _type STREQUAL "INTERFACE_LIBRARY"
     OR NOT "${visibility}" MATCHES "^(PRIVATE|PUBLIC)$")
    message(FATAL_ERROR "tflm_target_embed_assets(${target_name} ${visibility}): invalid args.")
  endif()
  cmake_parse_arguments(ARG "" "" "FILES" ${ARGN})
  set(_files ${ARG_FILES} ${ARG_UNPARSED_ARGUMENTS})
  if(NOT _files)
    message(FATAL_ERROR "tflm_target_embed_assets(${target_name}): no assets specified.")
  endif()
  if(NOT TARGET Python3::Interpreter)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()

  set(_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_cc_arrays.py")
  if(NOT EXISTS "${_script}")
    cmake_path(SET _script NORMALIZE
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tensorflow/lite/micro/tools/generate_cc_arrays.py")
  endif()
  if(NOT EXISTS "${_script}")
    message(FATAL_ERROR "TFLM asset generator script '${_script}' was not found.")
  endif()

  set(_tflm_suf_.tflite _model_data)
  set(_tflm_suf_.wav _audio_data)
  set(_tflm_suf_.bmp _image_data)
  set(_tflm_suf_.csv _test_data)
  set(_tflm_suf_.npy _test_data)

  # Path must end with /genfiles (required by generate_cc_arrays.py:35).
  cmake_path(SET _gen_root NORMALIZE "${CMAKE_CURRENT_BINARY_DIR}/genfiles")
  get_property(_seen TARGET "${target_name}" PROPERTY _TFLM_EMBEDDED_ASSETS)

  foreach(_item IN LISTS _files)
    cmake_path(ABSOLUTE_PATH _item
      BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _abs_in)
    cmake_path(GET _abs_in FILENAME _rel_in)
    foreach(_base IN ITEMS
        "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}")
      if(_base)
        cmake_path(SET _norm NORMALIZE "${_base}")
        cmake_path(IS_PREFIX _norm "${_abs_in}" NORMALIZE _is_pfx)
        if(_is_pfx AND NOT "${_abs_in}" STREQUAL "${_norm}")
          cmake_path(RELATIVE_PATH _abs_in BASE_DIRECTORY "${_norm}" OUTPUT_VARIABLE _rel_in)
          break()
        endif()
      endif()
    endforeach()
    if(_rel_in MATCHES "^\\.\\.(/|$)")
      cmake_path(GET _abs_in FILENAME _rel_in)
    endif()

    cmake_path(GET _rel_in EXTENSION LAST_ONLY _ext)
    if(NOT _tflm_suf_${_ext})
      message(FATAL_ERROR "tflm_target_embed_assets: unsupported extension '${_ext}' in '${_item}'.")
    endif()
    string(REGEX REPLACE "\\.[^.]*$" "${_tflm_suf_${_ext}}" _out_stem "${_rel_in}")
    cmake_path(SET _out_cc NORMALIZE "${_gen_root}/${_out_stem}.cc")
    cmake_path(SET _out_hdr NORMALIZE "${_gen_root}/${_out_stem}.h")

    string(SHA1 _asset_hash "${_out_cc}")
    set(_asset_tgt "_tflm_cc_array_${_asset_hash}")
    if(TARGET "${_asset_tgt}")
      get_target_property(_prev_in "${_asset_tgt}" _TFLM_ASSET_INPUT)
      if(NOT "${_prev_in}" STREQUAL "${_abs_in}")
        message(FATAL_ERROR
          "tflm_target_embed_assets: '${_abs_in}' collides on '${_out_cc}' with '${_prev_in}'.")
      endif()
    else()
      add_custom_command(OUTPUT "${_out_cc}" "${_out_hdr}"
        COMMAND Python3::Interpreter "${_script}" "${_out_cc}" "${_abs_in}"
        COMMAND Python3::Interpreter "${_script}" "${_out_hdr}" "${_abs_in}"
        DEPENDS "${_abs_in}" "${_script}"
        COMMENT "Generating C++ array for ${_rel_in}"
        VERBATIM)
      add_library(${_asset_tgt} OBJECT EXCLUDE_FROM_ALL "${_out_cc}" "${_out_hdr}")
      set_target_properties(${_asset_tgt} PROPERTIES _TFLM_ASSET_INPUT "${_abs_in}")
      if(TARGET tflm::micro AND NOT "${target_name}" STREQUAL "tflm_micro")
        target_link_libraries(${_asset_tgt} PRIVATE tflm::micro)
      else()
        target_compile_features(${_asset_tgt} PRIVATE cxx_std_17)
      endif()
      target_include_directories(${_asset_tgt} PRIVATE "${_gen_root}")
      if(COMMAND _tflm_apply_common_compile_options)
        _tflm_apply_common_compile_options(${_asset_tgt})
      endif()
    endif()

    if(NOT _asset_tgt IN_LIST _seen)
      list(APPEND _seen "${_asset_tgt}")
      set_property(TARGET "${target_name}" APPEND PROPERTY _TFLM_EMBEDDED_ASSETS "${_asset_tgt}")
      target_sources("${target_name}" PRIVATE $<TARGET_OBJECTS:${_asset_tgt}>)
      add_dependencies("${target_name}" "${_asset_tgt}")
    endif()
  endforeach()

  target_include_directories("${target_name}" ${visibility} "$<BUILD_INTERFACE:${_gen_root}>")
endfunction()
