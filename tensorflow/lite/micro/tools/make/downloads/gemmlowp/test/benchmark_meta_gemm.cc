// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
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
#include <vector>

#include "../eight_bit_int_gemm/eight_bit_int_gemm.h"
#include "test.h"

#if defined(__arm__) && !defined(GEMMLOWP_NEON)
#warning "Building without NEON support on ARM, check your compiler setup!"
#endif

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

const std::int32_t MIN_WORKING_SET_SIZE = 2 * 1024 * 1024;
const double MIN_OPS = 1000.0 * 1000000.0;

struct WorkingSet {
  WorkingSet() : lhs(nullptr), rhs(nullptr), result(nullptr) {}

  void init(std::int32_t n, std::int32_t m, std::int32_t k) {
    lhs = new std::uint8_t[n * k];
    rhs = new std::uint8_t[k * m];
    result = new std::uint8_t[m * n];
  }

  std::uint8_t* lhs;
  std::uint8_t* rhs;
  std::uint8_t* result;
};

struct Shape {
  std::int32_t n;
  std::int32_t m;
  std::int32_t k;

  std::int32_t repetitions;
  std::int32_t current_set;
  std::vector<WorkingSet> working_sets;

  Shape(std::int32_t n, std::int32_t m, std::int32_t k)
      : n(n), m(m), k(k), repetitions(1), current_set(0), working_sets() {}

  void init() {
    const std::int32_t size = n * k + k * m + n * m;
    const std::int32_t count = MIN_WORKING_SET_SIZE / size + 1;
    const double ops = static_cast<double>(n) * static_cast<double>(m) *
                       static_cast<double>(k);
    for (int i = 0; i < count; ++i) {
      working_sets.push_back(WorkingSet());
      working_sets.back().init(n, m, k);
    }
    current_set = 0;
    repetitions = MIN_OPS / ops + 20;
  }

  WorkingSet& working_set() { return working_sets[current_set]; }

  void next_working_set() {
    current_set = (current_set + 1) % working_sets.size();
  }
};

double run_gemm(std::int32_t n, std::int32_t m, std::int32_t k,
                std::uint8_t* lhs, std::uint8_t* rhs, std::uint8_t* result) {
  gemmlowp::eight_bit_int_gemm::EightBitIntGemm(
      true, false, false, m, n, k, rhs, -100, k, lhs, -100, k, result, 10000,
      10, 3, m, gemmlowp::eight_bit_int_gemm::BitDepthSetting::A8B8);
  return static_cast<double>(n * m * k * 2);
}

double run_gemms(std::vector<Shape>* shapes) {
  double ops = 0.0;
  for (auto& shape : *shapes) {
    ops += run_gemm(shape.n, shape.m, shape.k, shape.working_set().lhs,
                    shape.working_set().rhs, shape.working_set().result);
  }
  return ops;
}

void print_summary(std::vector<double>* times, bool full) {
  std::sort(times->begin(), times->end());

  double sum_times = 0;
  double sum_times_trimmed = 0;
  int count_times_trimmed = 0;
  const float trim_ratio = 0.25;
  const size_t count_trimmed = times->size() * trim_ratio;
  double sum_times_best = 0;
  int count_times_best = 0;
  const float best_ratio = 0.1;
  const size_t count_best = times->size() * best_ratio;

  for (size_t i = 0; i < times->size(); i++) {
    sum_times += (*times)[i];
    if (i >= count_trimmed && i < times->size() - count_trimmed) {
      sum_times_trimmed += (*times)[i];
      count_times_trimmed++;
    }
    if (i < count_best) {
      sum_times_best += (*times)[i];
      count_times_best++;
    }
  }

  const double min_latency = times->front();
  const double max_latency = times->back();
  const double mean_latency = sum_times / times->size();
  const double trimmed_mean_latency = sum_times_trimmed / count_times_trimmed;
  const double best_mean_latency = sum_times_best / count_times_best;

  if (full) {
    std::cout << "Graph latency (over " << times->size()
              << " iterations):" << std::endl;
    std::cout << "  Best:             " << min_latency << "s" << std::endl;
    std::cout << "  Worst:            " << max_latency << "s" << std::endl;
    std::cout << "  Mean:             " << mean_latency << "s" << std::endl;
    std::cout << "  " << 100 * trim_ratio
              << "% trimmed mean: " << trimmed_mean_latency << "s" << std::endl;
    std::cout << "  Mean of " << 100 * best_ratio
              << "% best: " << best_mean_latency << "s" << std::endl;
  } else {
    std::cout << (mean_latency * 1000.0) << std::endl;
  }
}

void time_all(std::vector<Shape>* shapes, std::int32_t repetitions,
              double max_time) {
  std::vector<double> times;
  double ops = 0.0;
  double sum_time = 0.0;

  while (sum_time < max_time) {
    double start = time();

    for (int i = 0; i < repetitions; ++i) {
      ops += run_gemms(shapes);
    }
    double delta_time = (time() - start);
    times.push_back(delta_time / repetitions);
    sum_time += delta_time;
  }

  print_summary(&times, true);
}

void time_one(Shape* shape, double max_time) {
  std::vector<double> times;
  double ops = 0.0;
  double sum_time = 0.0;

  std::cout << std::setprecision(6) << std::fixed << shape->n << ", "
            << shape->m << ", " << shape->k << ", " << std::flush;

  while (sum_time < max_time) {
    double start = time();

    for (int i = 0; i < shape->repetitions; ++i) {
      ops += run_gemm(shape->n, shape->m, shape->k, shape->working_set().lhs,
                      shape->working_set().rhs, shape->working_set().result);
      shape->next_working_set();
    }
    double delta_time = (time() - start);
    times.push_back(delta_time / shape->repetitions);
    sum_time += delta_time;
  }

  print_summary(&times, false);
}

int main() {
  std::vector<Shape> googlenet_gemms;
  googlenet_gemms.push_back(Shape(12544, 64, 147));
  googlenet_gemms.push_back(Shape(3136, 64, 64));
  googlenet_gemms.push_back(Shape(3136, 192, 576));
  googlenet_gemms.push_back(Shape(784, 64, 192));
  googlenet_gemms.push_back(Shape(784, 96, 192));
  googlenet_gemms.push_back(Shape(784, 128, 864));
  googlenet_gemms.push_back(Shape(784, 16, 192));
  googlenet_gemms.push_back(Shape(784, 32, 400));
  googlenet_gemms.push_back(Shape(784, 32, 192));
  googlenet_gemms.push_back(Shape(784, 128, 256));
  googlenet_gemms.push_back(Shape(784, 128, 256));
  googlenet_gemms.push_back(Shape(784, 192, 1152));
  googlenet_gemms.push_back(Shape(784, 32, 256));
  googlenet_gemms.push_back(Shape(784, 96, 800));
  googlenet_gemms.push_back(Shape(784, 64, 256));
  googlenet_gemms.push_back(Shape(196, 192, 480));
  googlenet_gemms.push_back(Shape(196, 96, 480));
  googlenet_gemms.push_back(Shape(196, 204, 864));
  googlenet_gemms.push_back(Shape(196, 16, 480));
  googlenet_gemms.push_back(Shape(196, 48, 400));
  googlenet_gemms.push_back(Shape(196, 64, 480));
  googlenet_gemms.push_back(Shape(196, 160, 508));
  googlenet_gemms.push_back(Shape(196, 112, 508));
  googlenet_gemms.push_back(Shape(196, 224, 1008));
  googlenet_gemms.push_back(Shape(196, 24, 508));
  googlenet_gemms.push_back(Shape(196, 64, 600));
  googlenet_gemms.push_back(Shape(196, 64, 508));
  googlenet_gemms.push_back(Shape(196, 128, 512));
  googlenet_gemms.push_back(Shape(196, 128, 512));
  googlenet_gemms.push_back(Shape(196, 256, 1152));
  googlenet_gemms.push_back(Shape(196, 24, 512));
  googlenet_gemms.push_back(Shape(196, 64, 600));
  googlenet_gemms.push_back(Shape(196, 64, 512));
  googlenet_gemms.push_back(Shape(196, 112, 512));
  googlenet_gemms.push_back(Shape(196, 144, 512));
  googlenet_gemms.push_back(Shape(196, 288, 1296));
  googlenet_gemms.push_back(Shape(196, 32, 512));
  googlenet_gemms.push_back(Shape(196, 64, 800));
  googlenet_gemms.push_back(Shape(196, 64, 512));
  googlenet_gemms.push_back(Shape(196, 256, 528));
  googlenet_gemms.push_back(Shape(196, 160, 528));
  googlenet_gemms.push_back(Shape(196, 320, 1440));
  googlenet_gemms.push_back(Shape(196, 32, 528));
  googlenet_gemms.push_back(Shape(196, 128, 800));
  googlenet_gemms.push_back(Shape(196, 128, 528));
  googlenet_gemms.push_back(Shape(49, 256, 832));
  googlenet_gemms.push_back(Shape(49, 160, 832));
  googlenet_gemms.push_back(Shape(49, 320, 1440));
  googlenet_gemms.push_back(Shape(49, 48, 832));
  googlenet_gemms.push_back(Shape(49, 128, 1200));
  googlenet_gemms.push_back(Shape(49, 128, 832));
  googlenet_gemms.push_back(Shape(49, 384, 832));
  googlenet_gemms.push_back(Shape(49, 192, 832));
  googlenet_gemms.push_back(Shape(49, 384, 1728));
  googlenet_gemms.push_back(Shape(49, 48, 832));
  googlenet_gemms.push_back(Shape(49, 128, 1200));
  googlenet_gemms.push_back(Shape(49, 128, 832));
  googlenet_gemms.push_back(Shape(16, 128, 508));
  googlenet_gemms.push_back(Shape(1, 1024, 2048));
  googlenet_gemms.push_back(Shape(1, 1008, 1024));
  googlenet_gemms.push_back(Shape(16, 128, 528));
  googlenet_gemms.push_back(Shape(1, 1024, 2048));
  googlenet_gemms.push_back(Shape(1, 1008, 1024));
  googlenet_gemms.push_back(Shape(1, 1008, 1024));

  for (auto& shape : googlenet_gemms) {
    shape.init();
  }

  std::vector<Shape> small_gemms;
  small_gemms.push_back(Shape(29232, 16, 25));
  small_gemms.push_back(Shape(7308, 6, 400));
  small_gemms.push_back(Shape(203, 3002, 216));

  for (auto& shape : small_gemms) {
    shape.init();
  }

  std::vector<Shape> others;
  others.push_back(Shape(100, 100, 100));
  others.push_back(Shape(1000, 1000, 1000));
  others.push_back(Shape(2000, 1000, 1000));

  for (auto& shape : others) {
    shape.init();
  }

  std::vector<Shape> lstm;
  lstm.push_back(Shape(1, 500, 320));
  lstm.push_back(Shape(1, 100, 500));
  lstm.push_back(Shape(1, 500, 500));
  lstm.push_back(Shape(1, 500, 100));
  lstm.push_back(Shape(1, 2000, 100));

  for (auto& shape : lstm) {
    shape.init();
  }

  gemmlowp::eight_bit_int_gemm::SetMaxNumThreads(4);

  std::cout << "Warmup run." << std::endl;
  time_all(&googlenet_gemms, 10, 1.0);
  time_all(&small_gemms, 50, 1.0);

  std::cout << "Timing all." << std::endl;
  time_all(&googlenet_gemms, 10, 10.0);
  time_all(&small_gemms, 50, 10.0);

  std::cout << "Timing separate." << std::endl;

  for (auto& shape : googlenet_gemms) {
    time_one(&shape, 0.10);
  }

  for (auto& shape : small_gemms) {
    time_one(&shape, 0.10);
  }

  for (auto& shape : others) {
    time_one(&shape, 0.10);
  }

  for (auto& shape : lstm) {
    time_one(&shape, 0.10);
  }

  return 0;
}
