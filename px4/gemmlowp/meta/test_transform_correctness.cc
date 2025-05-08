// Copyright 2016 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unistd.h>
#ifdef __APPLE__
#include <sys/time.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "single_thread_transform.h"
#include "transform_kernels.h"

#define EPSILON (0.0001)

using namespace gemmlowp::meta;

typedef Transform1DParams<std::int32_t, std::uint8_t, Requantize> RequantizeParams;
typedef Transform1DParams<float, std::uint8_t, Quantize> QuantizeParams;
typedef Transform1DParams<std::uint8_t, float, Dequantize> DequantizeParams;
typedef Transform1DParams<std::uint8_t, std::uint8_t, MinMax<std::uint8_t>> MinMaxParams;
typedef Transform1DParams<std::uint8_t, std::int32_t, BiasAdd<std::uint8_t>> BiasAddParams;

void prepare_data_requantize(int count, std::int32_t* data) {
  float scale = 4000000000.0f / static_cast<float>(count - 1);
  for (int i = 0; i < count; ++i) {
    float temp = -2000000000.0f + scale * i;
    data[i] = static_cast<std::int32_t>(temp);
  }
}

void prepare_data_quantize(int count, float* data) {
  float scale = 200.0f / static_cast<float>(count - 1);
  for (int i = 0; i < count; ++i) {
    data[i] = -100 + scale * i;
  }
}

void prepare_data_dequantize(int count, std::uint8_t* data) {
  for (int i = 0; i < count; ++i) {
    data[i] = static_cast<std::uint8_t>(i % 256);
  }
}

void prepare_data_minmax(int count, std::uint8_t* data) {
  for (int i = 0; i < count; ++i) {
    data[i] = static_cast<std::uint8_t>(i % 256);
  }
}

void prepare_data_biasadd(int count, std::uint8_t* data) {
  for (int i = 0; i < count; ++i) {
    data[i] = static_cast<std::uint8_t>(i % 256);
  }
}

void verify_requantize(const RequantizeParams& params) {
  for (int i = 0; i < params.kernel.count; ++i) {
    std::uint8_t actual = params.output[i];
    float expected = static_cast<float>(params.input[i]);
    expected -= params.kernel.input_range_offset;
    expected *= params.kernel.input_range_scale;
    expected += params.kernel.input_range_min;
    expected -= params.kernel.output_range_min;
    expected *= params.kernel.one_over_output_range_scale;
    expected += params.kernel.output_range_offset;
    std::uint8_t expected_uint8 = static_cast<std::uint8_t>(expected);

    if (actual != expected_uint8) {
      std::cout << "Wrong: " << i << " : " << actual << " vs. "
                << expected_uint8 << std::endl;
      std::exit(1);
    }
  }
  std::cout << "Requantize: OK" << std::endl;
}

void verify_quantize(const QuantizeParams& params) {
  for (int i = 0; i < params.kernel.count; ++i) {
    std::uint8_t actual = params.output[i];
    float expected = params.input[i];
    expected -= params.kernel.range_min;
    expected *= params.kernel.range_scale;
    expected += params.kernel.range_offset;
    std::uint8_t expected_uint8 = static_cast<std::uint8_t>(expected);

    if (actual != expected_uint8) {
      std::cout << "Wrong: " << i << " : " << actual << " vs. "
                << expected_uint8 << std::endl;
      std::exit(1);
    }
  }
  std::cout << "Quantize: OK" << std::endl;
}

void verify_dequantize(const DequantizeParams& params) {
  for (int i = 0; i < params.kernel.count; ++i) {
    float actual = params.output[i];
    float expected = static_cast<float>(params.input[i]);
    expected -= params.kernel.range_offset;
    expected *= params.kernel.range_scale;
    expected += params.kernel.range_min;
    if (std::abs(actual - expected) > EPSILON) {
      std::cout << std::setprecision(9) << "Wrong: " << i << " : " << actual
                << " vs. " << expected << std::endl;
      std::exit(1);
    }
  }
  std::cout << "Dequantize: OK" << std::endl;
}

void verify_minmax(const MinMaxParams& params) {
  for (int i = 0; i < params.kernel.count; ++i) {
    std::uint8_t actual = params.output[i];
    std::uint8_t expected = params.input[i];
    expected = std::min(expected, params.kernel.max);
    expected = std::max(expected, params.kernel.min);

    if (actual != expected) {
      std::cout << "Wrong: " << i << " : " << actual << " vs. " << expected
                << std::endl;
      std::exit(1);
    }
  }
  std::cout << "MinMax: OK" << std::endl;
}

void verify_biasadd(const BiasAddParams& params) {
  for (int i = 0; i < params.kernel.rows * params.kernel.count; ++i) {
    std::int32_t actual = params.output[i];
    std::uint8_t input = params.input[i];
    std::uint8_t bias = params.kernel.bias[i % params.kernel.count];
    float input_float = static_cast<float>(input);
    input_float -= params.kernel.input_range_offset;
    input_float *= params.kernel.input_range_scale;
    input_float += params.kernel.input_range_min;
    float bias_float = static_cast<float>(bias);
    bias_float -= params.kernel.bias_range_offset;
    bias_float *= params.kernel.bias_range_scale;
    bias_float += params.kernel.bias_range_min;
    float sum = input_float + bias_float;
    sum -= params.kernel.output_range_min;
    sum *= params.kernel.one_over_output_range_scale;
    sum += params.kernel.output_range_offset;
    std::int32_t expected = static_cast<std::int32_t>(sum);
    if (std::abs(actual - expected) > 1024) {
      std::cout << "Wrong: " << i << " : " << actual << " vs. " << expected
                << std::endl;
      std::exit(1);
    }
  }
  std::cout << "BiasAdd: OK" << std::endl;
}

int main() {
  std::unique_ptr<std::int32_t[]> array_int32(new std::int32_t[128 * 1024]);
  std::unique_ptr<std::uint8_t[]> array_uint8(new std::uint8_t[128 * 1024]);
  std::unique_ptr<std::uint8_t[]> array_uint8_2(new std::uint8_t[128 * 1024]);
  std::unique_ptr<float[]> array_float(new float[128 * 1024]);

  {
    RequantizeParams requantize_params;
    requantize_params.input = array_int32.get();
    requantize_params.output = array_uint8.get();
    requantize_params.kernel.count = 12345;
    requantize_params.kernel.input_range_min = -100.0f;
    requantize_params.kernel.input_range_scale =
        200.0f / ((static_cast<std::int64_t>(1) << 32) - 1);
    requantize_params.kernel.input_range_offset =
        static_cast<float>(std::numeric_limits<std::int32_t>::lowest());
    requantize_params.kernel.output_range_min = -100.f;
    requantize_params.kernel.one_over_output_range_scale =
        static_cast<float>((static_cast<std::int64_t>(1) << 8) - 1) / 200.0f;
    requantize_params.kernel.output_range_offset =
        static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());

    prepare_data_requantize(12345, array_int32.get());

    Transform1D<RequantizeParams, 16>(requantize_params);

    verify_requantize(requantize_params);
  }

  {
    QuantizeParams quantize_params;
    quantize_params.input = array_float.get();
    quantize_params.output = array_uint8.get();
    quantize_params.kernel.count = 12345;
    quantize_params.kernel.range_min = -100.0f;
    quantize_params.kernel.range_scale =
        static_cast<float>((static_cast<std::int64_t>(1) << 8) - 1) / 200.0f;
    quantize_params.kernel.range_offset =
        static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());

    prepare_data_quantize(12345, array_float.get());

    Transform1D<QuantizeParams, 16>(quantize_params);

    verify_quantize(quantize_params);
  }

  {
    DequantizeParams dequantize_params;
    dequantize_params.input = array_uint8.get();
    dequantize_params.output = array_float.get();
    dequantize_params.kernel.count = 12345;
    dequantize_params.kernel.range_min = -100.0f;
    dequantize_params.kernel.range_scale =
        200.0f / ((static_cast<std::int64_t>(1) << 8) - 1);
    dequantize_params.kernel.range_offset =
        static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());

    prepare_data_dequantize(12345, array_uint8.get());

    Transform1D<DequantizeParams, 16>(dequantize_params);

    verify_dequantize(dequantize_params);
  }

  {
    MinMaxParams minmax_params;
    minmax_params.input = array_uint8.get();
    minmax_params.output = array_uint8_2.get();
    minmax_params.kernel.count = 12345;
    minmax_params.kernel.min = 64;
    minmax_params.kernel.max = 192;

    prepare_data_minmax(12345, array_uint8.get());

    Transform1D<MinMaxParams, 16>(minmax_params);

    verify_minmax(minmax_params);
  }

  {
    BiasAddParams biasadd_params;
    biasadd_params.input = array_uint8.get();
    biasadd_params.output = array_int32.get();
    biasadd_params.kernel.count = 1234;
    biasadd_params.kernel.rows = 11;
    biasadd_params.kernel.input_range_min = -100.0f;
    biasadd_params.kernel.bias_range_min = -100.0f;
    biasadd_params.kernel.output_range_min = -250.0f;
    biasadd_params.kernel.input_range_offset =
        static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());
    biasadd_params.kernel.bias_range_offset =
        static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());
    biasadd_params.kernel.output_range_offset =
        static_cast<float>(std::numeric_limits<std::int32_t>::lowest());
    biasadd_params.kernel.input_range_scale =
        200.0f / ((static_cast<std::int64_t>(1) << 8) - 1);
    biasadd_params.kernel.bias_range_scale =
        200.0f / ((static_cast<std::int64_t>(1) << 8) - 1);
    biasadd_params.kernel.one_over_output_range_scale =
        static_cast<float>((static_cast<std::int64_t>(1) << 32) - 1) / 500.0f;
    biasadd_params.kernel.bias = array_uint8_2.get();

    prepare_data_biasadd(1234 * 11, array_uint8.get());
    prepare_data_biasadd(1234, array_uint8_2.get());

    Transform1D<BiasAddParams, 16>(biasadd_params);

    verify_biasadd(biasadd_params);
  }

  return 0;
}
