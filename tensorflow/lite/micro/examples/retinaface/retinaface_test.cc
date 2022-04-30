/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the specific model to run
#include "tensorflow/lite/micro/examples/retinaface/retinaface_model_data.h"
#include "tensorflow/lite/micro/examples/retinaface/retinaface_utils.h"

constexpr int IMG_H = 240;
constexpr int IMG_W = 320;
constexpr int N_ANCHORS = 3160;

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {

  tflite::MicroErrorReporter micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(g_retinaface_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  tflite::MicroMutableOpResolver<13> resolver;
  resolver.AddPad();
  resolver.AddTranspose();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddLeakyRelu();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddResizeNearestNeighbor();
  resolver.AddAdd();
  resolver.AddSoftmax();

  constexpr int kTensorArenaSize = 1.2 * 1024.0 * 1024.0; // MB
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* input = interpreter.input(0);

  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);     // N
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[1]);     // C
  TF_LITE_MICRO_EXPECT_EQ(IMG_H, input->dims->data[2]); // H
  TF_LITE_MICRO_EXPECT_EQ(IMG_W, input->dims->data[3]); // W

  std::string src_path = __FILE__;
  std::string src_dir = src_path.substr(0, src_path.rfind("/")); 
  std::string img_path = cv::samples::findFile(
      src_dir + "/images/Argentina.jpeg");
  cv::Mat raw_img = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat img;
  cv::resize(raw_img, img, cv::Size(IMG_W, IMG_H), 0.0, 0.0, cv::INTER_LINEAR);

  // img is in HWC format -> need to convert to NCHW
  // also unroll image 
  float means[3] = {104.0f, 117.0f, 123.0f};
  for (int ch = 0; ch < 3; ch++) {
    for (int vH = 0; vH < IMG_H; vH++) {
      for (int vW = 0; vW < IMG_W; vW++) {
        int vv = img.at<uchar>(vH*3*IMG_W + vW*3 + ch);
        float v = vv - means[ch];
        input->data.f[vW + vH*IMG_W + ch*IMG_W*IMG_H] = v;
      }
    }
  }

  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  std::cout << "model executed!" << std::endl;

  TfLiteTensor* output0 = interpreter.output(0);
  TfLiteTensor* output1 = interpreter.output(1);
  TfLiteTensor* output2 = interpreter.output(2);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output0->type);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output1->type);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output2->type);

  TF_LITE_MICRO_EXPECT_EQ(3, output0->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(3, output1->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(3, output2->dims->size);

  TF_LITE_MICRO_EXPECT_EQ(1, output0->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output1->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output2->dims->data[0]);

  TF_LITE_MICRO_EXPECT_EQ(10, output0->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(2 , output1->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(4 , output2->dims->data[2]);

  TF_LITE_MICRO_EXPECT_EQ(N_ANCHORS, output0->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(N_ANCHORS, output1->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(N_ANCHORS, output2->dims->data[1]);

  // POST processing
  
  BoxList loc;
  LandmList landm;
  std::vector<FloatPair> conf;
  convertOutputsToVectors(loc,
                          landm,
                          conf,
                          output0->data.f,
                          output1->data.f,
                          output2->data.f,
                          N_ANCHORS);

  PriorBox priorbox = PriorBox({{16, 32}, {64, 128}, {256, 512}},
                      {8, 16, 32},
                      false, IMG_H, IMG_W);
  BoxList priors = priorbox.forward();
  BoxList boxes = decode(loc, priors, {0.1f, 0.2f});
  std::array<int, 4> scale = {IMG_W, IMG_H, IMG_W, IMG_H};
  for (int i = 0; i < N_ANCHORS; i++) {
    for (int j = 0; j < 4; j++) {
      boxes[i][j] *= scale[j];
    }
  }

  LandmList landms = decode_landm(landm, priors, {0.1f, 0.2f});
  std::array<int, 10> scale1 = {IMG_W, IMG_H,
                               IMG_W, IMG_H,
                               IMG_W, IMG_H,
                               IMG_W, IMG_H,
                               IMG_W, IMG_H};
  for (int i = 0; i < N_ANCHORS; i++) {
    for (int j = 0; j < 10; j++) {
      landms[i][j] *= scale1[j];
    }
  }

  // Threshold filtering
  constexpr float threshold = 0.7;
  DetList dets;
  filterAndGetDetections(conf,
                         boxes,
                         landms,
                         dets,
                         threshold,
                         N_ANCHORS);

  // Non-max suppression 
  constexpr float nms_threshold = 0.4;
  nonMaxSuppression(&dets, nms_threshold);

  // Should detect 11 facts with `threshold`
  TF_LITE_MICRO_EXPECT_EQ(11, (int)dets.size());
}

TF_LITE_MICRO_TESTS_END
