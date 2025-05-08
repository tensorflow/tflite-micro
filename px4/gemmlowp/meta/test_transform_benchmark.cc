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

#include "multi_thread_transform.h"
#include "transform_kernels.h"

using namespace gemmlowp::meta;

double time() {
#ifdef __APPLE__
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + 1e-6 * t.tv_usec;
#else
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
}

#define kernel_size (16)

template <typename Context, typename Params>
void run_benchmark(const std::string& name, int repetitions, int elements,
                   Context* context, const Params& params) {
  std::cout << "Benchmark: " << name << std::endl;
  std::cout << "Warmup single." << std::endl;

  for (int i = 0; i < 10; ++i) {
    Transform1D<Params, kernel_size>(params);
  }

  std::cout << "Benchmark single." << std::endl;

  double start = time();

  for (int i = 0; i < repetitions; ++i) {
    Transform1D<Params, kernel_size>(params);
  }

  double wall_time = time() - start;
  double ops = static_cast<double>(elements) * repetitions;
  std::cout << "Avg: " << (wall_time / repetitions) << std::endl;
  std::cout << "Perf: " << static_cast<std::int64_t>(ops / wall_time) << "/s."
            << std::endl;

  std::cout << "Warmup single." << std::endl;

  for (int i = 0; i < 10; ++i) {
    MultiThreadTransform1D<Context, Params, kernel_size>(context, params);
  }

  std::cout << "Benchmark multi." << std::endl;

  start = time();

  for (int i = 0; i < repetitions; ++i) {
    MultiThreadTransform1D<Context, Params, kernel_size>(context, params);
  }

  wall_time = time() - start;
  ops = static_cast<double>(elements) * repetitions;
  std::cout << "Avg: " << (wall_time / repetitions) << std::endl;
  std::cout << "Perf: " << static_cast<std::int64_t>(ops / wall_time) << "/s."
            << std::endl;
}

int main() {
  const int repetitions = 500;
  const int elements = 4 * 1024 * 1024;

  std::unique_ptr<std::int32_t[]> int32_array(new std::int32_t[elements]);
  std::unique_ptr<std::uint8_t[]> uint8_array(new std::uint8_t[elements]);
  std::unique_ptr<float[]> float_array(new float[elements]);

  typedef SimpleContext<gemmlowp::WorkersPool> Context;
  Context context(4, new gemmlowp::WorkersPool());

  typedef Transform1DParams<std::int32_t, std::uint8_t, Requantize> RequantizeParams;
  RequantizeParams requantize_params;
  requantize_params.input = int32_array.get();
  requantize_params.output = uint8_array.get();
  requantize_params.kernel.count = elements;
  requantize_params.kernel.input_range_min = -100.0f;
  requantize_params.kernel.input_range_scale =
      200.0f / ((static_cast<std::int64_t>(1) << 32) - 1);
  requantize_params.kernel.input_range_offset =
      static_cast<float>(std::numeric_limits<std::int32_t>::lowest());
  requantize_params.kernel.output_range_min = -200.0f;
  requantize_params.kernel.one_over_output_range_scale =
      static_cast<float>((static_cast<std::int64_t>(1) << 8) - 1) / 500.0f;
  requantize_params.kernel.output_range_offset =
      static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());

  run_benchmark("Requantize", repetitions, elements, &context,
                requantize_params);

  typedef Transform1DParams<std::uint8_t, float, Dequantize> DequantizeParams;
  DequantizeParams dequantize_params;
  dequantize_params.input = uint8_array.get();
  dequantize_params.output = float_array.get();
  dequantize_params.kernel.count = elements;
  dequantize_params.kernel.range_min = -100.0f;
  dequantize_params.kernel.range_scale =
      static_cast<float>((static_cast<std::int64_t>(1) << 8) - 1) / 200.0f;
  dequantize_params.kernel.range_offset =
      static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());

  run_benchmark("Dequantize", repetitions, elements, &context,
                dequantize_params);

  typedef Transform1DParams<float, std::uint8_t, Quantize> QuantizeParams;
  QuantizeParams quantize_params;
  quantize_params.input = float_array.get();
  quantize_params.output = uint8_array.get();
  quantize_params.kernel.count = elements;
  quantize_params.kernel.range_min = -100.0f;
  quantize_params.kernel.range_scale =
      200.0f / ((static_cast<std::int64_t>(1) << 8) - 1);
  quantize_params.kernel.range_offset =
      static_cast<float>(std::numeric_limits<std::uint8_t>::lowest());

  run_benchmark("Quantize", repetitions, elements, &context, quantize_params);

  return 0;
}
