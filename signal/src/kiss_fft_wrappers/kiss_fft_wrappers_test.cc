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

#include <cstddef>
#include <cstdint>
#include <type_traits>

// clang-format off
// Intentionally include int16 -> float -> int16 (re-include) -> int32 in a
// single translation unit to verify multi-precision header composability.
#include "signal/src/kiss_fft_wrappers/kiss_fft_int16.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_int16.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_int32.h"
// clang-format on

#include "tensorflow/lite/micro/testing/micro_test_v2.h"

static_assert(
    std::is_same<decltype(kiss_fft_fixed16::kiss_fft_cpx{}.r), int16_t>::value,
    "kiss_fft_fixed16::kiss_fft_cpx::r must be int16_t");
static_assert(
    std::is_same<decltype(kiss_fft_fixed16::kiss_fft_cpx{}.i), int16_t>::value,
    "kiss_fft_fixed16::kiss_fft_cpx::i must be int16_t");
static_assert(
    std::is_same<decltype(kiss_fft_float::kiss_fft_cpx{}.r), float>::value,
    "kiss_fft_float::kiss_fft_cpx::r must be float");
static_assert(
    std::is_same<decltype(kiss_fft_float::kiss_fft_cpx{}.i), float>::value,
    "kiss_fft_float::kiss_fft_cpx::i must be float");
static_assert(
    std::is_same<decltype(kiss_fft_fixed32::kiss_fft_cpx{}.r), int32_t>::value,
    "kiss_fft_fixed32::kiss_fft_cpx::r must be int32_t");
static_assert(
    std::is_same<decltype(kiss_fft_fixed32::kiss_fft_cpx{}.i), int32_t>::value,
    "kiss_fft_fixed32::kiss_fft_cpx::i must be int32_t");

TEST(KissFftWrappersTest, FloatMultiPrecisionAndSafety) {
  constexpr int kN = 8;
  constexpr int kBins = kN / 2 + 1;

  EXPECT_EQ(8, kiss_fft_float::kiss_fft_next_fast_size(7));

  size_t odd_size = 0;
  EXPECT_EQ(nullptr, kiss_fft_float::kiss_fftr_alloc(7, 0, nullptr, &odd_size));

  alignas(16) uint8_t fwd_arena[2048];
  alignas(16) uint8_t inv_arena[2048];
  size_t fwd_size = 0;
  size_t inv_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_float::kiss_fftr_alloc(kN, 0, nullptr, &fwd_size));
  EXPECT_EQ(nullptr,
            kiss_fft_float::kiss_fftr_alloc(kN, 1, nullptr, &inv_size));
  ASSERT_GT(fwd_size, static_cast<size_t>(0));
  ASSERT_LE(fwd_size, sizeof(fwd_arena));
  ASSERT_GT(inv_size, static_cast<size_t>(0));
  ASSERT_LE(inv_size, sizeof(inv_arena));

  kiss_fft_float::kiss_fftr_cfg fwd_cfg =
      kiss_fft_float::kiss_fftr_alloc(kN, 0, fwd_arena, &fwd_size);
  kiss_fft_float::kiss_fftr_cfg inv_cfg =
      kiss_fft_float::kiss_fftr_alloc(kN, 1, inv_arena, &inv_size);
  ASSERT_NE(nullptr, fwd_cfg);
  ASSERT_NE(nullptr, inv_cfg);

  const float impulse[kN] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  kiss_fft_float::kiss_fft_cpx freq[kBins] = {};
  kiss_fft_float::kiss_fftr(fwd_cfg, impulse, freq);
  for (int k = 0; k < kBins; ++k) {
    EXPECT_NEAR(1.0f, freq[k].r, 1e-5f);
    EXPECT_NEAR(0.0f, freq[k].i, 1e-5f);
  }

  // Mismatched direction calls must return cleanly without exit or mutation.
  float sentinel_time[kN];
  for (int i = 0; i < kN; ++i) sentinel_time[i] = -42.0f;
  kiss_fft_float::kiss_fftri(fwd_cfg, freq, sentinel_time);
  EXPECT_EQ(-42.0f, sentinel_time[0]);

  kiss_fft_float::kiss_fft_cpx sentinel_freq[kBins];
  for (int k = 0; k < kBins; ++k) sentinel_freq[k] = {-42.0f, -42.0f};
  kiss_fft_float::kiss_fftr(inv_cfg, impulse, sentinel_freq);
  EXPECT_EQ(-42.0f, sentinel_freq[0].r);

  const float signal_in[kN] = {1.0f, 0.5f,    -0.25f, 0.125f,
                               0.0f, -0.125f, 0.25f,  -0.5f};
  float signal_out[kN] = {};
  kiss_fft_float::kiss_fftr(fwd_cfg, signal_in, freq);
  kiss_fft_float::kiss_fftri(inv_cfg, freq, signal_out);
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(signal_in[i] * kN, signal_out[i], 1e-4f);
  }

  // Complex FFT + stride + cleanup round-trip.
  alignas(16) uint8_t cpx_fwd_arena[2048];
  alignas(16) uint8_t cpx_inv_arena[2048];
  size_t cpx_fwd_size = 0;
  size_t cpx_inv_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_float::kiss_fft_alloc(kN, 0, nullptr, &cpx_fwd_size));
  EXPECT_EQ(nullptr,
            kiss_fft_float::kiss_fft_alloc(kN, 1, nullptr, &cpx_inv_size));
  ASSERT_LE(cpx_fwd_size, sizeof(cpx_fwd_arena));
  ASSERT_LE(cpx_inv_size, sizeof(cpx_inv_arena));

  kiss_fft_float::kiss_fft_cfg cpx_fwd =
      kiss_fft_float::kiss_fft_alloc(kN, 0, cpx_fwd_arena, &cpx_fwd_size);
  kiss_fft_float::kiss_fft_cfg cpx_inv =
      kiss_fft_float::kiss_fft_alloc(kN, 1, cpx_inv_arena, &cpx_inv_size);
  ASSERT_NE(nullptr, cpx_fwd);
  ASSERT_NE(nullptr, cpx_inv);

  kiss_fft_float::kiss_fft_cpx cpx_in[kN] = {};
  kiss_fft_float::kiss_fft_cpx cpx_strided[2 * kN] = {};
  for (int i = 0; i < kN; ++i) {
    cpx_in[i] = {signal_in[i], -0.5f * signal_in[i]};
    cpx_strided[2 * static_cast<size_t>(i)] = cpx_in[i];
  }
  kiss_fft_float::kiss_fft_cpx cpx_mid[kN] = {};
  kiss_fft_float::kiss_fft_cpx cpx_stride_mid[kN] = {};
  kiss_fft_float::kiss_fft_cpx cpx_out[kN] = {};
  kiss_fft_float::kiss_fft(cpx_fwd, cpx_in, cpx_mid);
  kiss_fft_float::kiss_fft_stride(cpx_fwd, cpx_strided, cpx_stride_mid, 2);
  kiss_fft_float::kiss_fft(cpx_inv, cpx_mid, cpx_out);
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(cpx_mid[i].r, cpx_stride_mid[i].r, 1e-5f);
    EXPECT_NEAR(cpx_mid[i].i, cpx_stride_mid[i].i, 1e-5f);
    EXPECT_NEAR(cpx_in[i].r * kN, cpx_out[i].r, 1e-4f);
    EXPECT_NEAR(cpx_in[i].i * kN, cpx_out[i].i, 1e-4f);
  }
  kiss_fft_float::kiss_fft_cleanup();
}

TEST(KissFftWrappersTest, Fixed16MultiPrecisionAndSafety) {
  constexpr int kN = 8;
  constexpr int kBins = kN / 2 + 1;

  EXPECT_EQ(8, kiss_fft_fixed16::kiss_fft_next_fast_size(7));

  size_t odd_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_fixed16::kiss_fftr_alloc(7, 0, nullptr, &odd_size));

  alignas(16) uint8_t fwd_arena[2048];
  alignas(16) uint8_t inv_arena[2048];
  size_t fwd_size = 0;
  size_t inv_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_fixed16::kiss_fftr_alloc(kN, 0, nullptr, &fwd_size));
  EXPECT_EQ(nullptr,
            kiss_fft_fixed16::kiss_fftr_alloc(kN, 1, nullptr, &inv_size));
  ASSERT_GT(fwd_size, static_cast<size_t>(0));
  ASSERT_LE(fwd_size, sizeof(fwd_arena));
  ASSERT_GT(inv_size, static_cast<size_t>(0));
  ASSERT_LE(inv_size, sizeof(inv_arena));

  kiss_fft_fixed16::kiss_fftr_cfg fwd_cfg =
      kiss_fft_fixed16::kiss_fftr_alloc(kN, 0, fwd_arena, &fwd_size);
  kiss_fft_fixed16::kiss_fftr_cfg inv_cfg =
      kiss_fft_fixed16::kiss_fftr_alloc(kN, 1, inv_arena, &inv_size);
  ASSERT_NE(nullptr, fwd_cfg);
  ASSERT_NE(nullptr, inv_cfg);

  const int16_t impulse[kN] = {8192, 0, 0, 0, 0, 0, 0, 0};
  kiss_fft_fixed16::kiss_fft_cpx freq[kBins] = {};
  kiss_fft_fixed16::kiss_fftr(fwd_cfg, impulse, freq);
  for (int k = 0; k < kBins; ++k) {
    EXPECT_NEAR(1024, freq[k].r, 2);
    EXPECT_NEAR(0, freq[k].i, 2);
  }

  int16_t sentinel_time[kN];
  for (int i = 0; i < kN; ++i) sentinel_time[i] = -42;
  kiss_fft_fixed16::kiss_fftri(fwd_cfg, freq, sentinel_time);
  EXPECT_EQ(-42, sentinel_time[0]);

  kiss_fft_fixed16::kiss_fft_cpx sentinel_freq[kBins];
  for (int k = 0; k < kBins; ++k) sentinel_freq[k] = {-42, -42};
  kiss_fft_fixed16::kiss_fftr(inv_cfg, impulse, sentinel_freq);
  EXPECT_EQ(-42, sentinel_freq[0].r);

  const int16_t signal_in[kN] = {8192, 4096,  -2048, 1024,
                                 0,    -1024, 2048,  -4096};
  int16_t signal_out[kN] = {};
  kiss_fft_fixed16::kiss_fftr(fwd_cfg, signal_in, freq);
  kiss_fft_fixed16::kiss_fftri(inv_cfg, freq, signal_out);
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(signal_in[i] / kN, signal_out[i], 4);
  }

  alignas(16) uint8_t cpx_fwd_arena[2048];
  alignas(16) uint8_t cpx_inv_arena[2048];
  size_t cpx_fwd_size = 0;
  size_t cpx_inv_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_fixed16::kiss_fft_alloc(kN, 0, nullptr, &cpx_fwd_size));
  EXPECT_EQ(nullptr,
            kiss_fft_fixed16::kiss_fft_alloc(kN, 1, nullptr, &cpx_inv_size));
  ASSERT_LE(cpx_fwd_size, sizeof(cpx_fwd_arena));
  ASSERT_LE(cpx_inv_size, sizeof(cpx_inv_arena));

  kiss_fft_fixed16::kiss_fft_cfg cpx_fwd =
      kiss_fft_fixed16::kiss_fft_alloc(kN, 0, cpx_fwd_arena, &cpx_fwd_size);
  kiss_fft_fixed16::kiss_fft_cfg cpx_inv =
      kiss_fft_fixed16::kiss_fft_alloc(kN, 1, cpx_inv_arena, &cpx_inv_size);
  ASSERT_NE(nullptr, cpx_fwd);
  ASSERT_NE(nullptr, cpx_inv);

  kiss_fft_fixed16::kiss_fft_cpx cpx_in[kN] = {};
  kiss_fft_fixed16::kiss_fft_cpx cpx_strided[2 * kN] = {};
  for (int i = 0; i < kN; ++i) {
    cpx_in[i] = {signal_in[i], static_cast<int16_t>(-signal_in[i] / 2)};
    cpx_strided[2 * static_cast<size_t>(i)] = cpx_in[i];
  }
  kiss_fft_fixed16::kiss_fft_cpx cpx_mid[kN] = {};
  kiss_fft_fixed16::kiss_fft_cpx cpx_stride_mid[kN] = {};
  kiss_fft_fixed16::kiss_fft_cpx cpx_out[kN] = {};
  kiss_fft_fixed16::kiss_fft(cpx_fwd, cpx_in, cpx_mid);
  kiss_fft_fixed16::kiss_fft_stride(cpx_fwd, cpx_strided, cpx_stride_mid, 2);
  kiss_fft_fixed16::kiss_fft(cpx_inv, cpx_mid, cpx_out);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(cpx_mid[i].r, cpx_stride_mid[i].r);
    EXPECT_EQ(cpx_mid[i].i, cpx_stride_mid[i].i);
    EXPECT_NEAR(cpx_in[i].r / kN, cpx_out[i].r, 4);
    EXPECT_NEAR(cpx_in[i].i / kN, cpx_out[i].i, 4);
  }
  kiss_fft_fixed16::kiss_fft_cleanup();
}

TEST(KissFftWrappersTest, Fixed32MultiPrecisionAndSafety) {
  constexpr int kN = 8;
  constexpr int kBins = kN / 2 + 1;

  EXPECT_EQ(8, kiss_fft_fixed32::kiss_fft_next_fast_size(7));

  size_t odd_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_fixed32::kiss_fftr_alloc(7, 0, nullptr, &odd_size));

  alignas(16) uint8_t fwd_arena[2048];
  alignas(16) uint8_t inv_arena[2048];
  size_t fwd_size = 0;
  size_t inv_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_fixed32::kiss_fftr_alloc(kN, 0, nullptr, &fwd_size));
  EXPECT_EQ(nullptr,
            kiss_fft_fixed32::kiss_fftr_alloc(kN, 1, nullptr, &inv_size));
  ASSERT_GT(fwd_size, static_cast<size_t>(0));
  ASSERT_LE(fwd_size, sizeof(fwd_arena));
  ASSERT_GT(inv_size, static_cast<size_t>(0));
  ASSERT_LE(inv_size, sizeof(inv_arena));

  kiss_fft_fixed32::kiss_fftr_cfg fwd_cfg =
      kiss_fft_fixed32::kiss_fftr_alloc(kN, 0, fwd_arena, &fwd_size);
  kiss_fft_fixed32::kiss_fftr_cfg inv_cfg =
      kiss_fft_fixed32::kiss_fftr_alloc(kN, 1, inv_arena, &inv_size);
  ASSERT_NE(nullptr, fwd_cfg);
  ASSERT_NE(nullptr, inv_cfg);

  const int32_t impulse[kN] = {16777216, 0, 0, 0, 0, 0, 0, 0};
  kiss_fft_fixed32::kiss_fft_cpx freq[kBins] = {};
  kiss_fft_fixed32::kiss_fftr(fwd_cfg, impulse, freq);
  for (int k = 0; k < kBins; ++k) {
    EXPECT_NEAR(2097152, freq[k].r, 4);
    EXPECT_NEAR(0, freq[k].i, 4);
  }

  int32_t sentinel_time[kN];
  for (int i = 0; i < kN; ++i) sentinel_time[i] = -42;
  kiss_fft_fixed32::kiss_fftri(fwd_cfg, freq, sentinel_time);
  EXPECT_EQ(-42, sentinel_time[0]);

  kiss_fft_fixed32::kiss_fft_cpx sentinel_freq[kBins];
  for (int k = 0; k < kBins; ++k) sentinel_freq[k] = {-42, -42};
  kiss_fft_fixed32::kiss_fftr(inv_cfg, impulse, sentinel_freq);
  EXPECT_EQ(-42, sentinel_freq[0].r);

  const int32_t signal_in[kN] = {16777216, 8388608,  -4194304, 2097152,
                                 0,        -2097152, 4194304,  -8388608};
  int32_t signal_out[kN] = {};
  kiss_fft_fixed32::kiss_fftr(fwd_cfg, signal_in, freq);
  kiss_fft_fixed32::kiss_fftri(inv_cfg, freq, signal_out);
  for (int i = 0; i < kN; ++i) {
    EXPECT_NEAR(signal_in[i] / kN, signal_out[i], 16);
  }

  alignas(16) uint8_t cpx_fwd_arena[2048];
  alignas(16) uint8_t cpx_inv_arena[2048];
  size_t cpx_fwd_size = 0;
  size_t cpx_inv_size = 0;
  EXPECT_EQ(nullptr,
            kiss_fft_fixed32::kiss_fft_alloc(kN, 0, nullptr, &cpx_fwd_size));
  EXPECT_EQ(nullptr,
            kiss_fft_fixed32::kiss_fft_alloc(kN, 1, nullptr, &cpx_inv_size));
  ASSERT_LE(cpx_fwd_size, sizeof(cpx_fwd_arena));
  ASSERT_LE(cpx_inv_size, sizeof(cpx_inv_arena));

  kiss_fft_fixed32::kiss_fft_cfg cpx_fwd =
      kiss_fft_fixed32::kiss_fft_alloc(kN, 0, cpx_fwd_arena, &cpx_fwd_size);
  kiss_fft_fixed32::kiss_fft_cfg cpx_inv =
      kiss_fft_fixed32::kiss_fft_alloc(kN, 1, cpx_inv_arena, &cpx_inv_size);
  ASSERT_NE(nullptr, cpx_fwd);
  ASSERT_NE(nullptr, cpx_inv);

  kiss_fft_fixed32::kiss_fft_cpx cpx_in[kN] = {};
  kiss_fft_fixed32::kiss_fft_cpx cpx_strided[2 * kN] = {};
  for (int i = 0; i < kN; ++i) {
    cpx_in[i] = {signal_in[i], -signal_in[i] / 2};
    cpx_strided[2 * static_cast<size_t>(i)] = cpx_in[i];
  }
  kiss_fft_fixed32::kiss_fft_cpx cpx_mid[kN] = {};
  kiss_fft_fixed32::kiss_fft_cpx cpx_stride_mid[kN] = {};
  kiss_fft_fixed32::kiss_fft_cpx cpx_out[kN] = {};
  kiss_fft_fixed32::kiss_fft(cpx_fwd, cpx_in, cpx_mid);
  kiss_fft_fixed32::kiss_fft_stride(cpx_fwd, cpx_strided, cpx_stride_mid, 2);
  kiss_fft_fixed32::kiss_fft(cpx_inv, cpx_mid, cpx_out);
  for (int i = 0; i < kN; ++i) {
    EXPECT_EQ(cpx_mid[i].r, cpx_stride_mid[i].r);
    EXPECT_EQ(cpx_mid[i].i, cpx_stride_mid[i].i);
    EXPECT_NEAR(cpx_in[i].r / kN, cpx_out[i].r, 16);
    EXPECT_NEAR(cpx_in[i].i / kN, cpx_out[i].i, 16);
  }
  kiss_fft_fixed32::kiss_fft_cleanup();
}

TF_LITE_MICRO_TESTS_MAIN
