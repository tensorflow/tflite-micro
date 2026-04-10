// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");

// fuzz_model_load.cc
//
// OSS-Fuzz harness for TensorFlow Lite Micro (tflite-micro).
//
// Pipeline exercised:
//   GetModel(data) -> MicroInterpreter(model, resolver, arena, size)
//   -> AllocateTensors() -> fill inputs -> Invoke()



#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


constexpr int kArenaSize = 200000;
alignas(16) static uint8_t arena[kArenaSize];

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {

  if (size < 8 || size > 4 * 1024 * 1024) {
    return 0;
  }


  const tflite::Model *model = tflite::GetModel(data);
  if (model == nullptr) {
    return 0;
  }

  // Basic null checks to avoid trivial nullptr crashes in the parser
  if (model->subgraphs() == nullptr || model->subgraphs()->size() == 0) {
    return 0;
  }
  if (model->buffers() == nullptr) {
    return 0;
  }


  tflite::MicroMutableOpResolver<20> resolver;
  // Integer overflow PoC ops
  resolver.AddFullyConnected();
  resolver.AddDequantize();
  resolver.AddQuantize();
  resolver.AddReshape();
  // Gather OOB read PoC ops
  resolver.AddGather();             
  resolver.AddGatherNd();           
  resolver.AddEmbeddingLookup();    
  // Common TFLM ops for broader coverage
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddSoftmax();
  resolver.AddLogistic();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddMean();


  tflite::MicroInterpreter interp(model, resolver, arena, kArenaSize);

  TfLiteStatus alloc_status = interp.AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    return 0;
  }


  for (size_t i = 0; i < interp.inputs_size(); ++i) {
    TfLiteTensor *inp = interp.input(i);
    if (inp == nullptr || inp->data.raw == nullptr || inp->bytes == 0) {
      continue;
    }
    if (inp->bytes < (size_t)kArenaSize) {
      for (size_t byte_idx = 0; byte_idx < inp->bytes; ++byte_idx) {
        inp->data.raw[byte_idx] = data[byte_idx % size];
      }
    }
  }

  interp.Invoke();

  return 0;
}