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
"""Script to whether the size exceeds a threshold and also plot size history as a graph.
"""

import argparse
import pandas as pd
from matplotlib import pyplot as plt

# Limit the size history check for the past 60 days
SIZE_HISTORY_DEPTH = 60
# If a section of size log exceeds the below threshold, an error will be raised
SIZE_THRESHOLD_SETTING = {
    "text": 512,
    "total": 512,
}


def _plot_and_detect_size_increase_for_binary(input_dir, output_dir,
                                              binary_name, threshold):
  csv_path = '%s/%s.csv' % (input_dir, binary_name)
  size_log = pd.read_csv(csv_path, index_col=False).iloc[-SIZE_HISTORY_DEPTH:]
  size_log.reset_index(drop=True, inplace=True)
  start_date = size_log.iloc[0, 0][0:10]
  end_date = size_log.iloc[-1, 0][0:10]

  fig, axs = plt.subplots(3, 2)
  fig.suptitle('Source: %s\n%s - %s' % (binary_name, start_date, end_date))

  threshold_messages = []

  for index, name in enumerate(['text', 'data', 'total']):
    err_msg_or_none = _subplot_and_detect_size_increase(
        axs, size_log, name, index, threshold)
    if err_msg_or_none is not None:
      threshold_messages.append('%s failure: %s' %
                                (binary_name, err_msg_or_none))

  fig_path = '%s/%s.png' % (output_dir, binary_name)
  fig.tight_layout()
  plt.savefig(fig_path)
  plt.clf()

  return threshold_messages


def _subplot_and_detect_size_increase(subplot_axs, size_log, section_name, row,
                                      threshold):
  subplot_axs[row, 0].set_title(section_name)
  subplot_axs[row, 0].plot(size_log[section_name], 'o-')
  subplot_axs[row, 0].set_ylabel('Abs Sz(bytes)')
  increased_size = size_log[section_name].diff()
  subplot_axs[row, 1].plot(increased_size, 'o-')
  subplot_axs[row, 1].set_ylabel('Incr Sz (bytes)')

  if section_name in threshold and len(increased_size) > 1:
    if increased_size[1] > threshold[section_name]:
      return '%s size increases by %d and exceeds threshold %d' % (
          section_name, increased_size[1], threshold[section_name])

  # By default there is no size increase that exceeds the threshold
  return None


def _detect_size_increase_and_plot_history(input_dir, output_dir, binary_list,
                                           threshold_setting):
  threshold_messages = []

  for binary_name in binary_list:
    threshold_messages += _plot_and_detect_size_increase_for_binary(
        input_dir, output_dir, binary_name, threshold_setting)

  if len(threshold_messages) != 0:
    raise RuntimeError(str(threshold_messages))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  default_binary_list_string = 'keyword_benchmark,baseline_memory_footprint,interpreter_memory_footprint'
  parser.add_argument(
      '--binary_list',
      nargs='?',
      const=default_binary_list_string,
      default=default_binary_list_string,
      help=
      'binary list separated by comma (e.g. keyword_benchmark,baseline_memory_footprint)'
  )
  parser.add_argument('--input_dir',
                      help='Path to the size log file (e.g. ~/size_log')
  parser.add_argument('--output_dir', help='Path to save plot to (e.g. /tmp/)')

  args = parser.parse_args()

  binary_names = args.binary_list.split(',')

  _detect_size_increase_and_plot_history(args.input_dir, args.output_dir,
                                         binary_names, SIZE_THRESHOLD_SETTING)
