/*
 * Implements helper functions and classes for running the retinaface face 
 * detection pipeline. Mostly consists of utilities needed for post-processing
 * of the results of the model invocation.
 */

#include <iostream>
#include <unordered_set>

using namespace std;

// The "Box" type represents a box in the image by the top-left and top-right
// corner of the box.
typedef array<float, 4> Box;
typedef vector<Box> BoxList;

// "Landm" represents the coordinates of the five facial landmarks.
typedef array<float, 10> Landm;
typedef vector<Landm> LandmList;

// "Det" represents a detection and contains the box and landmark info combined
// along with a confidence score.
typedef array<float, 15> Det;
typedef vector<Det> DetList;

// Just a pair of floats
typedef array<float, 2> FloatPair;

/* TODO: what does this class represent ?
 */
class PriorBox {
public:
  PriorBox(vector<array<int, 2>> min_sizes,
           vector<int> steps,
           bool clip,
           int image_height,
           int image_width) :
           _min_sizes(min_sizes),
           _steps(steps),
           _clip(clip),
           _image_size({image_height, image_width}) {
    for (int step : _steps) {
      float a = (float)_image_size[0] / (step);
      float b = (float)_image_size[1] / (step);
      _feature_maps.push_back({a, b});
    }
  }

  BoxList forward() {
    BoxList anchors;
    for (int k = 0; k < (int)_feature_maps.size(); k++) {
      auto f = _feature_maps.at(k);
      auto min_sizes = _min_sizes[k];

      for (int i = 0; i < f[0]; i++) {
        for (int j = 0; j < f[1]; j++) {
          for (auto min_size : min_sizes) {
            float s_kx = (float)min_size / _image_size[1];
            float s_ky = (float)min_size / _image_size[0];
            float dense_cx = (j + 0.5f) * _steps[k] / _image_size[1];
            float dense_cy = (i + 0.5f) * _steps[k] / _image_size[0];
            anchors.push_back({dense_cx, dense_cy, s_kx, s_ky});
          }
        }
      }
    }
    return anchors;
  }

private:
  vector<array<int, 2>> _min_sizes;
  vector<int> _steps;
  bool _clip;
  array<int, 2> _image_size;
  vector<FloatPair> _feature_maps;
};

/* Decodes the locations of bounding boxes in the image.
 * loc: output of the model invocation.
 * priors: TODO: what is this?
 * variances: TODO: what is this?
 */
BoxList decode(BoxList& loc, BoxList& priors, FloatPair variances) {

  BoxList boxes;
  for (int i = 0; i < (int)loc.size(); i++) {
    auto el0 = priors[i][0] + loc[i][0] * variances[0] * priors[i][2]; 
    auto el1 = priors[i][1] + loc[i][1] * variances[0] * priors[i][3]; 

    auto el2 = priors[i][2] * std::exp(loc[i][2] * variances[1]);
    auto el3 = priors[i][3] * std::exp(loc[i][3] * variances[1]);

    boxes.push_back({el0, el1, el2, el3});
  }
  for (int i = 0; i < (int)loc.size(); i++) {
    boxes[i][0] -= boxes[i][2] / 2.0f;
    boxes[i][1] -= boxes[i][3] / 2.0f;

    boxes[i][2] += boxes[i][0];
    boxes[i][3] += boxes[i][1];
  }
  return boxes;
}

/* Decodes the landmarks.
 * pre: model invocation landmark outputs.
 * priors: TODO: what is this?
 * variances: TODO: what is this?
 */
LandmList
decode_landm(LandmList& pre, BoxList& priors, array<float, 2> variances) {
  LandmList r;
  for (int i = 0; i < (int)pre.size(); i++) {
    Landm landm;
    for (int j = 0; j < 10; j++) {
      landm[j] = priors[i][j%2] + pre[i][j]*variances[0]*priors[i][j%2 + 2];
    }
    r.push_back(landm);
  }
  return r;
}


/* Converts the raw output buffers from the model invocations to vectors of
 * processed box locations, landmark locations, and confidence scores.
 */
void convertOutputsToVectors(BoxList& loc,
                             LandmList& landm,
                             std::vector<FloatPair>& conf,
                             float output0_f[],
                             float output1_f[],
                             float output2_f[],
                             int N_ANCHORS) {
  for (int i = 0; i < N_ANCHORS; i++) {
    Box loc_i;
    for (int j = 0; j < 4; j++) {
      loc_i[j] = output2_f[i*4 + j];
    }
    loc.push_back(loc_i);
  }
  for (int i = 0; i < N_ANCHORS; i++) {
    Landm landm_i;
    for (int j = 0; j < 10; j++) {
      landm_i[j] = output0_f[i*10 + j];
    }
    landm.push_back(landm_i);
  }
  for (int i = 0; i < N_ANCHORS; i++) {
    FloatPair conf_i;
    for (int j = 0; j < 2; j++) {
      conf_i[j] = output1_f[i*2 + j];
    }
    conf.push_back(conf_i);
  }
}

void filterAndGetDetections(std::vector<FloatPair>& conf,
                            BoxList& boxes,
                            LandmList& landms, 
                            DetList& dets,
                            float threshold,
                            int N_ANCHORS) {
  for (int i = 0; i < N_ANCHORS; i++) {
    if (conf[i][1] > threshold) {
      Det det;
      int j = 0;
      while (j < 4) {
        det[j] = boxes[i][j];
        j++;
      }
      det[j++] = conf[i][1];
      while(j < 15) {
        det[j] = landms[i][j - 5];
        j++;
      }
      dets.push_back(det);
    }
  }
}

void nonMaxSuppression(DetList* dets,
                       float nms_threshold) {
  DetList dets2;
  int finds = dets->size();
  sort(dets->begin(), dets->end(),
            [](Det& a, Det& b) { return a[4] > b[4]; });
  int curr_proposal = 0;

  bool skip_indices[finds];
  for (int i = 0; i < finds; i++) {
    skip_indices[i] = false;
  }
  skip_indices[curr_proposal] = true;
  int nProcessedIndices = 1;

  while (nProcessedIndices < finds) {
    dets2.push_back((*dets)[curr_proposal]);

    float x1 = (*dets)[curr_proposal][0];
    float y1 = (*dets)[curr_proposal][1];
    float x2 = (*dets)[curr_proposal][2];
    float y2 = (*dets)[curr_proposal][3];
    float area =  (x2 - x1 + 1) * (y2 - y1 + 1);

    bool found_next_proposal = false;
    // Go over all other possibilities that have not yet been added or skipped
    for (int i = 0; i < finds; i++) {
      if (skip_indices[i]) {
        continue;
      }
      float area2 = ((*dets)[i][2] - (*dets)[i][0] + 1) *
                    ((*dets)[i][3] - (*dets)[i][1] + 1);

      float int_x1 = max(x1, (*dets)[i][0]);
      float int_y1 = max(y1, (*dets)[i][1]);
      float int_x2 = min(x2, (*dets)[i][2]);
      float int_y2 = min(y2, (*dets)[i][3]);

      float inter_area = max(0.0f, int_x2 - int_x1 + 1) *
                         max(0.0f, int_y2 - int_y1 + 1);
      float union_area = area + area2 - inter_area;

      if ((inter_area / union_area) > nms_threshold) {
        skip_indices[i] = true;
        nProcessedIndices += 1;

      } else if (!found_next_proposal) {
        found_next_proposal = true;
        curr_proposal = i;

        skip_indices[i] = true;
        nProcessedIndices += 1;
      }
    }
  }
  // replace the old detection list with the new list
  *dets = dets2;
}
