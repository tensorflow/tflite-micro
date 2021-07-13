# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import sys

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("pr_files", help="File with list of files modified by the Pull Request", default="")
  args = parser.parse_args()

  tflite_files = set(line.strip() for line in open("ci/tflite_files.txt"))
  pr_files = set(line.strip() for line in open(args.pr_files))

  tflite_files_in_pr = tflite_files.intersection(pr_files)

  if len(tflite_files_in_pr) != 0:
    print("The following files should be modified in the upstream Tensorflow repo:")
    print("\n".join(tflite_files_in_pr))
    sys.exit(1)
  else:
    print("No TfLite files are modified in the PR. We can proceed.")
