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
project(tflm_consumer LANGUAGES CXX)
option(TFLM_SMOKE_ENABLE_SIGNAL "Test linking tflm::signal" OFF)
if(DEFINED TFLM_SOURCE_ROOT)
  if(TFLM_SMOKE_ENABLE_SIGNAL)
    set(TFLM_ENABLE_SIGNAL ON)
  endif()
  include(FetchContent)
  FetchContent_Declare(tflm SOURCE_DIR "${TFLM_SOURCE_ROOT}")
  FetchContent_MakeAvailable(tflm)
else()
  set(_comps micro)
  if(TFLM_SMOKE_ENABLE_SIGNAL)
    list(APPEND _comps signal)
  endif()
  find_package(tflm CONFIG REQUIRED COMPONENTS ${_comps})
endif()
configure_file("${TFLM_TEST_MODEL_FILE}"
  "${CMAKE_CURRENT_BINARY_DIR}/models/simple_add_model.tflite" COPYONLY)
add_executable(consumer_app "${CMAKE_CURRENT_SOURCE_DIR}/consumer_main.cc")
tflm_target_embed_assets(
  consumer_app PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/models/simple_add_model.tflite")
target_link_libraries(consumer_app PRIVATE tflm::micro)
if(TFLM_SMOKE_ENABLE_SIGNAL)
  target_compile_definitions(consumer_app PRIVATE TFLM_SMOKE_TEST_SIGNAL=1)
  target_link_libraries(consumer_app PRIVATE tflm::signal)
endif()
