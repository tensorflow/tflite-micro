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

#include "signal/lite/kernels/signal_ops.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "signal/lite/register/signal_ops_registerer.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace tflm_signal {
namespace lite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::NotNull;

TEST(SignalOpsRegistererTest, RegistersAll14Ops) {
  MutableOpResolver resolver;
  SignalOpsRegisterer(&resolver);

  const char* kOpNames[] = {
      "SignalDelay",
      "SignalEnergy",
      "SignalFftAutoScale",
      "SignalFilterBank",
      "SignalFilterBankLog",
      "SignalFilterBankSpectralSubtraction",
      "SignalFilterBankSquareRoot",
      "SignalFramer",
      "SignalIrfft",
      "SignalOverlapAdd",
      "SignalPCAN",
      "SignalRfft",
      "SignalStacker",
      "SignalWindow",
  };
  for (const char* op_name : kOpNames) {
    const TfLiteRegistration* reg = resolver.FindOp(op_name, 1);
    ASSERT_THAT(reg, NotNull()) << "Missing op: " << op_name;
    EXPECT_THAT(reg->registration_external, NotNull())
        << "Missing registration_external for: " << op_name;
  }
}

class DelayOpModel : public SingleOpModel {
 public:
  DelayOpModel(const TensorData& input, const TensorData& output,
               int delay_length) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Int("delay_length", delay_length); });
    fbb.Finish();
    SetCustomOp("SignalDelay", fbb.GetBuffer(), Register_DELAY);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }

 private:
  int input_;
  int output_;
};

TEST(SignalOpsTest, DelayOp) {
  DelayOpModel m({TensorType_INT16, {3}}, {TensorType_INT16, {3}}, 2);
  m.PopulateTensor<int16_t>(m.input(), {10, 20, 30});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 10}));

  m.PopulateTensor<int16_t>(m.input(), {40, 50, 60});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({20, 30, 40}));
}

class EnergyOpModel : public SingleOpModel {
 public:
  EnergyOpModel(const TensorData& input, const TensorData& output,
                int start_index, int end_index) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("start_index", start_index);
      fbb.Int("end_index", end_index);
    });
    fbb.Finish();
    SetCustomOp("SignalEnergy", fbb.GetBuffer(), Register_ENERGY);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<uint32_t> GetOutput() { return ExtractVector<uint32_t>(output_); }

 private:
  int input_;
  int output_;
};

TEST(SignalOpsTest, EnergyOp) {
  EnergyOpModel m({TensorType_INT16, {6}}, {TensorType_UINT32, {3}}, 0, 2);
  m.PopulateTensor<int16_t>(m.input(), {3, 4, 5, 12, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(25, 169, 0));
}

class FftAutoScaleOpModel : public SingleOpModel {
 public:
  FftAutoScaleOpModel() {
    input_ = AddInput({TensorType_INT16, {4}});
    output_ = AddOutput({TensorType_INT16, {4}});
    scale_bit_ = AddOutput({TensorType_INT32, {}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {});
    fbb.Finish();
    SetCustomOp("SignalFftAutoScale", fbb.GetBuffer(), Register_FFT_AUTO_SCALE);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }
  int32_t GetScaleBit() { return ExtractVector<int32_t>(scale_bit_)[0]; }

 private:
  int input_;
  int output_;
  int scale_bit_;
};

TEST(SignalOpsTest, FftAutoScaleOp) {
  FftAutoScaleOpModel m;
  m.PopulateTensor<int16_t>(m.input(), {0, 0x1000, 0x2000, 0x1000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 0x2000, 0x4000, 0x2000));
  EXPECT_EQ(m.GetScaleBit(), 1);
}

class FilterBankOpModel : public SingleOpModel {
 public:
  FilterBankOpModel() {
    energy_ = AddInput({TensorType_UINT32, {4}});
    weights_ = AddInput({TensorType_INT16, {4}});
    unweights_ = AddInput({TensorType_INT16, {4}});
    freq_starts_ = AddInput({TensorType_INT16, {3}});
    weight_starts_ = AddInput({TensorType_INT16, {3}});
    widths_ = AddInput({TensorType_INT16, {3}});
    output_ = AddOutput({TensorType_UINT64, {2}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Int("num_channels", 2); });
    fbb.Finish();
    SetCustomOp("SignalFilterBank", fbb.GetBuffer(), Register_FILTER_BANK);
    BuildInterpreter({GetShape(energy_), GetShape(weights_),
                      GetShape(unweights_), GetShape(freq_starts_),
                      GetShape(weight_starts_), GetShape(widths_)});
  }
  int energy() const { return energy_; }
  int weights() const { return weights_; }
  int unweights() const { return unweights_; }
  int freq_starts() const { return freq_starts_; }
  int weight_starts() const { return weight_starts_; }
  int widths() const { return widths_; }
  std::vector<uint64_t> GetOutput() { return ExtractVector<uint64_t>(output_); }

 private:
  int energy_;
  int weights_;
  int unweights_;
  int freq_starts_;
  int weight_starts_;
  int widths_;
  int output_;
};

TEST(SignalOpsTest, FilterBankOp) {
  FilterBankOpModel m;
  m.PopulateTensor<uint32_t>(m.energy(), {10, 20, 30, 40});
  m.PopulateTensor<int16_t>(m.weights(), {1024, 1024, 1024, 1024});
  m.PopulateTensor<int16_t>(m.unweights(), {3072, 3072, 3072, 3072});
  m.PopulateTensor<int16_t>(m.freq_starts(), {0, 1, 2});
  m.PopulateTensor<int16_t>(m.weight_starts(), {0, 1, 2});
  m.PopulateTensor<int16_t>(m.widths(), {1, 1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput().size(), 2);
}

class FilterBankLogOpModel : public SingleOpModel {
 public:
  FilterBankLogOpModel() {
    input_ = AddInput({TensorType_UINT32, {2}});
    output_ = AddOutput({TensorType_INT16, {2}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("output_scale", 16);
      fbb.Int("input_correction_bits", 0);
    });
    fbb.Finish();
    SetCustomOp("SignalFilterBankLog", fbb.GetBuffer(),
                Register_FILTER_BANK_LOG);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }

 private:
  int input_;
  int output_;
};

TEST(SignalOpsTest, FilterBankLogOp) {
  FilterBankLogOpModel m;
  m.PopulateTensor<uint32_t>(m.input(), {1, 100});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput()[0], 0);
  EXPECT_GT(m.GetOutput()[1], 0);
}

class FilterBankSpectralSubtractionOpModel : public SingleOpModel {
 public:
  FilterBankSpectralSubtractionOpModel() {
    input_ = AddInput({TensorType_UINT32, {2}});
    output_ = AddOutput({TensorType_UINT32, {2}});
    noise_ = AddOutput({TensorType_UINT32, {2}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("num_channels", 2);
      fbb.Int("smoothing", 1638);
      fbb.Int("one_minus_smoothing", 14746);
      fbb.Int("alternate_smoothing", 0);
      fbb.Int("alternate_one_minus_smoothing", 16384);
      fbb.Int("smoothing_bits", 0);
      fbb.Int("min_signal_remaining", 819);
      fbb.Bool("clamping", true);
      fbb.Int("spectral_subtraction_bits", 14);
    });
    fbb.Finish();
    SetCustomOp("SignalFilterBankSpectralSubtraction", fbb.GetBuffer(),
                Register_FILTER_BANK_SPECTRAL_SUBTRACTION);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<uint32_t> GetOutput() { return ExtractVector<uint32_t>(output_); }

 private:
  int input_;
  int output_;
  int noise_;
};

TEST(SignalOpsTest, FilterBankSpectralSubtractionOp) {
  FilterBankSpectralSubtractionOpModel m;
  m.PopulateTensor<uint32_t>(m.input(), {1000, 2000});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_GT(m.GetOutput()[0], 0);
  EXPECT_GT(m.GetOutput()[1], 0);
}

class FilterBankSquareRootOpModel : public SingleOpModel {
 public:
  FilterBankSquareRootOpModel() {
    input_ = AddInput({TensorType_UINT64, {2}});
    scale_bits_ = AddInput({TensorType_INT32, {}});
    output_ = AddOutput({TensorType_UINT32, {2}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {});
    fbb.Finish();
    SetCustomOp("SignalFilterBankSquareRoot", fbb.GetBuffer(),
                Register_FILTER_BANK_SQUARE_ROOT);
    BuildInterpreter({GetShape(input_), GetShape(scale_bits_)});
  }
  int input() const { return input_; }
  int scale_bits() const { return scale_bits_; }
  std::vector<uint32_t> GetOutput() { return ExtractVector<uint32_t>(output_); }

 private:
  int input_;
  int scale_bits_;
  int output_;
};

TEST(SignalOpsTest, FilterBankSquareRootOp) {
  FilterBankSquareRootOpModel m;
  m.PopulateTensor<uint64_t>(m.input(), {16, 64});
  m.PopulateTensor<int32_t>(m.scale_bits(), {0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(4, 8));
}

class FramerOpModel : public SingleOpModel {
 public:
  FramerOpModel() {
    input_ = AddInput({TensorType_INT16, {2}});
    output_ = AddOutput({TensorType_INT16, {1, 3}});
    valid_ = AddOutput({TensorType_BOOL, {}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("frame_size", 3);
      fbb.Int("frame_step", 2);
      fbb.Bool("prefill", true);
    });
    fbb.Finish();
    SetCustomOp("SignalFramer", fbb.GetBuffer(), Register_FRAMER);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }
  bool GetValid() { return ExtractVector<bool>(valid_)[0]; }

 private:
  int input_;
  int output_;
  int valid_;
};

TEST(SignalOpsTest, FramerOp) {
  FramerOpModel m;
  m.PopulateTensor<int16_t>(m.input(), {10, 20});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetValid());
  EXPECT_THAT(m.GetOutput(), ElementsAre(0, 10, 20));
}

class RfftOpModel : public SingleOpModel {
 public:
  RfftOpModel() {
    input_ = AddInput({TensorType_FLOAT32, {4}});
    output_ = AddOutput({TensorType_FLOAT32, {6}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("T", TensorType_FLOAT32);
      fbb.Int("fft_length", 4);
    });
    fbb.Finish();
    SetCustomOp("SignalRfft", fbb.GetBuffer(), Register_RFFT);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(SignalOpsTest, RfftOp) {
  RfftOpModel m;
  m.PopulateTensor<float>(m.input(), {1.0f, 0.0f, 0.0f, 0.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f));
}

class IrfftOpModel : public SingleOpModel {
 public:
  IrfftOpModel() {
    input_ = AddInput({TensorType_FLOAT32, {6}});
    output_ = AddOutput({TensorType_FLOAT32, {4}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("T", TensorType_FLOAT32);
      fbb.Int("fft_length", 4);
    });
    fbb.Finish();
    SetCustomOp("SignalIrfft", fbb.GetBuffer(), Register_IRFFT);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(SignalOpsTest, IrfftOp) {
  IrfftOpModel m;
  m.PopulateTensor<float>(m.input(), {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(1.0f, 0.0f, 0.0f, 0.0f));
}

class OverlapAddOpModel : public SingleOpModel {
 public:
  OverlapAddOpModel() {
    input_ = AddInput({TensorType_INT16, {1, 4}});
    output_ = AddOutput({TensorType_INT16, {2}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("T", TensorType_INT16);
      fbb.Int("frame_step", 2);
    });
    fbb.Finish();
    SetCustomOp("SignalOverlapAdd", fbb.GetBuffer(), Register_OVERLAP_ADD);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }

 private:
  int input_;
  int output_;
};

TEST(SignalOpsTest, OverlapAddOp) {
  OverlapAddOpModel m;
  m.PopulateTensor<int16_t>(m.input(), {1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(1, 2));

  m.PopulateTensor<int16_t>(m.input(), {10, 20, 30, 40});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(13, 24));
}

class PcanOpModel : public SingleOpModel {
 public:
  PcanOpModel() {
    input_ = AddInput({TensorType_UINT32, {2}});
    noise_ = AddInput({TensorType_UINT32, {2}});
    gain_lut_ = AddInput({TensorType_INT16, {140}});
    output_ = AddOutput({TensorType_UINT32, {2}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Int("snr_shift", 6); });
    fbb.Finish();
    SetCustomOp("SignalPCAN", fbb.GetBuffer(), Register_PCAN);
    BuildInterpreter({GetShape(input_), GetShape(noise_), GetShape(gain_lut_)});
  }
  int input() const { return input_; }
  int noise() const { return noise_; }
  int gain_lut() const { return gain_lut_; }
  std::vector<uint32_t> GetOutput() { return ExtractVector<uint32_t>(output_); }

 private:
  int input_;
  int noise_;
  int gain_lut_;
  int output_;
};

TEST(SignalOpsTest, PcanOp) {
  PcanOpModel m;
  m.PopulateTensor<uint32_t>(m.input(), {1000, 2000});
  m.PopulateTensor<uint32_t>(m.noise(), {100, 200});
  std::vector<int16_t> lut(140, 64);
  m.PopulateTensor<int16_t>(m.gain_lut(), lut);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput().size(), 2);
}

class StackerOpModel : public SingleOpModel {
 public:
  StackerOpModel() {
    input_ = AddInput({TensorType_INT16, {2}});
    output_ = AddOutput({TensorType_INT16, {4}});
    valid_ = AddOutput({TensorType_BOOL, {}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("num_channels", 2);
      fbb.Int("stacker_left_context", 1);
      fbb.Int("stacker_right_context", 0);
      fbb.Int("stacker_step", 1);
    });
    fbb.Finish();
    SetCustomOp("SignalStacker", fbb.GetBuffer(), Register_STACKER);
    BuildInterpreter({GetShape(input_)});
  }
  int input() const { return input_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }
  bool GetValid() { return ExtractVector<bool>(valid_)[0]; }

 private:
  int input_;
  int output_;
  int valid_;
};

TEST(SignalOpsTest, StackerOp) {
  StackerOpModel m;
  m.PopulateTensor<int16_t>(m.input(), {7, 8});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_TRUE(m.GetValid());
  EXPECT_THAT(m.GetOutput(), ElementsAre(7, 8, 7, 8));
}

class WindowOpModel : public SingleOpModel {
 public:
  WindowOpModel() {
    input_ = AddInput({TensorType_INT16, {4}});
    weights_ = AddInput({TensorType_INT16, {4}});
    output_ = AddOutput({TensorType_INT16, {4}});
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Int("shift", 2); });
    fbb.Finish();
    SetCustomOp("SignalWindow", fbb.GetBuffer(), Register_WINDOW);
    BuildInterpreter({GetShape(input_), GetShape(weights_)});
  }
  int input() const { return input_; }
  int weights() const { return weights_; }
  std::vector<int16_t> GetOutput() { return ExtractVector<int16_t>(output_); }

 private:
  int input_;
  int weights_;
  int output_;
};

TEST(SignalOpsTest, WindowOp) {
  WindowOpModel m;
  m.PopulateTensor<int16_t>(m.input(), {8, 16, 24, 32});
  m.PopulateTensor<int16_t>(m.weights(), {4, 4, 4, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput(), ElementsAre(8, 16, 24, 32));
}

}  // namespace
}  // namespace lite
}  // namespace tflm_signal
}  // namespace tflite
