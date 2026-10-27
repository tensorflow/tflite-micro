/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/cc/litert_api_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/tflite_support/custom_op/custom_op.h"
#include "signal/lite/kernels/signal_ops.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace tflm_signal {
namespace lite {
namespace {

using ::testing::ElementsAreArray;

struct TensorSpec {
  TensorType type;
  std::vector<int32_t> shape;
  const char* name;
};

struct OpSpec {
  const char* custom_name;
  std::vector<int32_t> inputs;
  std::vector<int32_t> outputs;
  std::vector<uint8_t> custom_options;
};

// Builds a minimal in-memory .tflite FlatBuffer with custom ops.
flatbuffers::DetachedBuffer BuildModelBuffer(
    const std::vector<TensorSpec>& tensor_specs,
    const std::vector<int32_t>& subgraph_inputs,
    const std::vector<int32_t>& subgraph_outputs,
    const std::vector<OpSpec>& op_specs) {
  flatbuffers::FlatBufferBuilder fbb;

  // Buffer 0 must always be an empty buffer in TFLite schema.
  std::vector<flatbuffers::Offset<Buffer>> buffers;
  buffers.push_back(
      CreateBuffer(fbb, fbb.CreateVector(std::vector<uint8_t>{})));

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.reserve(tensor_specs.size());
  for (const auto& spec : tensor_specs) {
    tensors.push_back(CreateTensor(fbb, fbb.CreateVector(spec.shape), spec.type,
                                   /*buffer=*/0, fbb.CreateString(spec.name),
                                   /*quantization=*/0, /*is_variable=*/false));
  }

  std::vector<flatbuffers::Offset<OperatorCode>> opcodes;
  std::vector<flatbuffers::Offset<Operator>> operators;
  opcodes.reserve(op_specs.size());
  operators.reserve(op_specs.size());
  for (size_t i = 0; i < op_specs.size(); ++i) {
    const auto& op = op_specs[i];
    opcodes.push_back(CreateOperatorCodeDirect(fbb, BuiltinOperator_CUSTOM,
                                               op.custom_name, /*version=*/1));
    operators.push_back(CreateOperator(
        fbb, /*opcode_index=*/i, fbb.CreateVector(op.inputs),
        fbb.CreateVector(op.outputs), BuiltinOptions_NONE,
        /*builtin_options=*/0, fbb.CreateVector(op.custom_options),
        CustomOptionsFormat_FLEXBUFFERS));
  }

  std::vector<flatbuffers::Offset<SubGraph>> subgraphs;
  subgraphs.push_back(CreateSubGraph(
      fbb, fbb.CreateVector(tensors), fbb.CreateVector(subgraph_inputs),
      fbb.CreateVector(subgraph_outputs), fbb.CreateVector(operators),
      fbb.CreateString("main")));

  auto model = CreateModel(fbb, /*version=*/3, fbb.CreateVector(opcodes),
                           fbb.CreateVector(subgraphs),
                           fbb.CreateString("signal_litert_test"),
                           fbb.CreateVector(buffers));
  FinishModelBuffer(fbb, model);
  return fbb.Release();
}

litert::Options CreateCpuBuiltinOptions() {
  auto options_expected = litert::Options::Create();
  EXPECT_TRUE(options_expected.HasValue());
  auto options = std::move(options_expected.Value());
  options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);

  auto cpu_options_expected = options.GetCpuOptions();
  EXPECT_TRUE(cpu_options_expected.HasValue());
  cpu_options_expected.Value().SetKernelMode(kLiteRtCpuKernelModeBuiltin);
  return options;
}

TEST(SignalLiteRtTest, WindowOpWithTfLiteOperator) {
  auto env_expected = litert::Environment::Create({});
  ASSERT_TRUE(env_expected.HasValue());
  auto env = std::move(env_expected.Value());

  auto options = CreateCpuBuiltinOptions();
  ASSERT_TRUE(
      litert::tflite_support::AddCustomOp(options, CreateWindowOperator())
          .HasValue());

  flexbuffers::Builder flex;
  flex.Map([&]() { flex.Int("shift", 12); });
  flex.Finish();

  auto model_buf = BuildModelBuffer(
      {
          {TensorType_INT16, {4}, "input"},
          {TensorType_INT16, {4}, "weights"},
          {TensorType_INT16, {4}, "output"},
      },
      /*subgraph_inputs=*/{0, 1},
      /*subgraph_outputs=*/{2},
      {{"SignalWindow", {0, 1}, {2}, flex.GetBuffer()}});

  auto compiled_expected = litert::CompiledModel::Create(
      env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      options);
  ASSERT_TRUE(compiled_expected.HasValue());
  auto compiled = std::move(compiled_expected.Value());

  auto inputs_expected = compiled.CreateInputBuffers();
  ASSERT_TRUE(inputs_expected.HasValue());
  auto inputs = std::move(inputs_expected.Value());

  auto outputs_expected = compiled.CreateOutputBuffers();
  ASSERT_TRUE(outputs_expected.HasValue());
  auto outputs = std::move(outputs_expected.Value());

  std::vector<int16_t> in_data = {4096, 4096, 4096, 4096};
  std::vector<int16_t> weights_data = {1024, 2048, 3072, 4096};
  ASSERT_TRUE(inputs[0]
                  .Write<int16_t>(litert::Span<const int16_t>(in_data))
                  .HasValue());
  ASSERT_TRUE(inputs[1]
                  .Write<int16_t>(litert::Span<const int16_t>(weights_data))
                  .HasValue());

  ASSERT_TRUE(compiled.Run(inputs, outputs).HasValue());

  auto lock_expected = litert::TensorBufferScopedLock::Create<const int16_t>(
      outputs[0], litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(lock_expected.HasValue());
  litert::Span<const int16_t> out_span(lock_expected.Value().second, 4);
  EXPECT_THAT(out_span, ElementsAreArray({1024, 2048, 3072, 4096}));
}

TEST(SignalLiteRtTest, StatefulDelayOpPersistsStateAcrossRuns) {
  auto env_expected = litert::Environment::Create({});
  ASSERT_TRUE(env_expected.HasValue());
  auto env = std::move(env_expected.Value());

  auto options = CreateCpuBuiltinOptions();
  ASSERT_TRUE(
      litert::tflite_support::AddCustomOp(options, CreateDelayOperator())
          .HasValue());

  flexbuffers::Builder flex;
  flex.Map([&]() { flex.Int("delay_length", 2); });
  flex.Finish();

  auto model_buf = BuildModelBuffer(
      {
          {TensorType_INT16, {3}, "input"},
          {TensorType_INT16, {3}, "output"},
      },
      /*subgraph_inputs=*/{0},
      /*subgraph_outputs=*/{1}, {{"SignalDelay", {0}, {1}, flex.GetBuffer()}});

  auto compiled_expected = litert::CompiledModel::Create(
      env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      options);
  ASSERT_TRUE(compiled_expected.HasValue());
  auto compiled = std::move(compiled_expected.Value());

  auto inputs_expected = compiled.CreateInputBuffers();
  ASSERT_TRUE(inputs_expected.HasValue());
  auto inputs = std::move(inputs_expected.Value());

  auto outputs_expected = compiled.CreateOutputBuffers();
  ASSERT_TRUE(outputs_expected.HasValue());
  auto outputs = std::move(outputs_expected.Value());

  // Run 1: input [10, 20, 30] -> delayed by 2 -> [0, 0, 10]
  std::vector<int16_t> in_run1 = {10, 20, 30};
  ASSERT_TRUE(inputs[0]
                  .Write<int16_t>(litert::Span<const int16_t>(in_run1))
                  .HasValue());
  ASSERT_TRUE(compiled.Run(inputs, outputs).HasValue());
  {
    auto lock_expected = litert::TensorBufferScopedLock::Create<const int16_t>(
        outputs[0], litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_expected.HasValue());
    litert::Span<const int16_t> out_span(lock_expected.Value().second, 3);
    EXPECT_THAT(out_span, ElementsAreArray({0, 0, 10}));
  }

  // Run 2: input [40, 50, 60] -> carries over [20, 30] -> [20, 30, 40]
  std::vector<int16_t> in_run2 = {40, 50, 60};
  ASSERT_TRUE(inputs[0]
                  .Write<int16_t>(litert::Span<const int16_t>(in_run2))
                  .HasValue());
  ASSERT_TRUE(compiled.Run(inputs, outputs).HasValue());
  {
    auto lock_expected = litert::TensorBufferScopedLock::Create<const int16_t>(
        outputs[0], litert::TensorBuffer::LockMode::kRead);
    ASSERT_TRUE(lock_expected.HasValue());
    litert::Span<const int16_t> out_span(lock_expected.Value().second, 3);
    EXPECT_THAT(out_span, ElementsAreArray({20, 30, 40}));
  }
}

TEST(SignalLiteRtTest, MultiOpFramerWindowRfftPipeline) {
  auto env_expected = litert::Environment::Create({});
  ASSERT_TRUE(env_expected.HasValue());
  auto env = std::move(env_expected.Value());

  auto options = CreateCpuBuiltinOptions();
  ASSERT_TRUE(
      litert::tflite_support::AddCustomOp(options, CreateFramerOperator())
          .HasValue());
  ASSERT_TRUE(
      litert::tflite_support::AddCustomOp(options, CreateWindowOperator())
          .HasValue());
  ASSERT_TRUE(litert::tflite_support::AddCustomOp(options, CreateRfftOperator())
                  .HasValue());

  flexbuffers::Builder framer_flex;
  framer_flex.Map([&]() {
    framer_flex.Int("frame_size", 64);
    framer_flex.Int("frame_step", 64);
    framer_flex.Bool("prefill", true);
  });
  framer_flex.Finish();

  flexbuffers::Builder window_flex;
  window_flex.Map([&]() { window_flex.Int("shift", 12); });
  window_flex.Finish();

  flexbuffers::Builder rfft_flex;
  rfft_flex.Map([&]() {
    rfft_flex.Int("fft_length", 64);
    rfft_flex.Int("T", TensorType_INT16);
  });
  rfft_flex.Finish();

  auto model_buf = BuildModelBuffer(
      {
          {TensorType_INT16, {64}, "audio_in"},
          {TensorType_INT16, {64}, "weights"},
          {TensorType_INT16, {1, 64}, "framed"},
          {TensorType_BOOL, {}, "valid"},
          {TensorType_INT16, {1, 64}, "windowed"},
          {TensorType_INT16, {1, 66}, "spectrum"},
      },
      /*subgraph_inputs=*/{0, 1},
      /*subgraph_outputs=*/{5, 3},
      {
          {"SignalFramer", {0}, {2, 3}, framer_flex.GetBuffer()},
          {"SignalWindow", {2, 1}, {4}, window_flex.GetBuffer()},
          {"SignalRfft", {4}, {5}, rfft_flex.GetBuffer()},
      });

  auto compiled_expected = litert::CompiledModel::Create(
      env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      options);
  ASSERT_TRUE(compiled_expected.HasValue());
  auto compiled = std::move(compiled_expected.Value());

  auto inputs_expected = compiled.CreateInputBuffers();
  ASSERT_TRUE(inputs_expected.HasValue());
  auto inputs = std::move(inputs_expected.Value());

  auto outputs_expected = compiled.CreateOutputBuffers();
  ASSERT_TRUE(outputs_expected.HasValue());
  auto outputs = std::move(outputs_expected.Value());

  // Impulse at index 0 with unit window weights (4096 >> 12 = 1.0).
  std::vector<int16_t> audio_in(64, 0);
  audio_in[0] = 6400;
  std::vector<int16_t> weights(64, 4096);
  ASSERT_TRUE(inputs[0]
                  .Write<int16_t>(litert::Span<const int16_t>(audio_in))
                  .HasValue());
  ASSERT_TRUE(inputs[1]
                  .Write<int16_t>(litert::Span<const int16_t>(weights))
                  .HasValue());

  ASSERT_TRUE(compiled.Run(inputs, outputs).HasValue());

  auto spec_lock = litert::TensorBufferScopedLock::Create<const int16_t>(
      outputs[0], litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(spec_lock.HasValue());
  litert::Span<const int16_t> spec_span(spec_lock.Value().second, 66);
  // Integer RFFT scales by 1/fft_length (6400 / 64 = 100 real, 0 imag for all
  // 33 bins).
  for (int i = 0; i < 33; ++i) {
    EXPECT_EQ(spec_span[2 * i], 100) << "bin " << i;
    EXPECT_EQ(spec_span[2 * i + 1], 0) << "bin " << i;
  }

  auto valid_lock = litert::TensorBufferScopedLock::Create<const bool>(
      outputs[1], litert::TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(valid_lock.HasValue());
  EXPECT_TRUE(valid_lock.Value().second[0]);
}

}  // namespace
}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
