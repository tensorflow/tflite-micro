#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Script to build the required binaries, profile their size and generate log.
"""

import argparse
import datetime
import os
import pandas as pd
import subprocess

def _build_a_binary(root_dir, binary_name, makefile_options, toolchain, additional_make_flags, cmsis_path, optimized_kernel_dir):
  os.chdir(root_dir)
  params_list = [
      "make", "-f", "tensorflow/lite/micro/tools/make/Makefile", binary_name
  ] + ["%s=%s" % (key, value) for (key, value) in makefile_options.items()]
  if toolchain is not None:
      params_list += ["TOOLCHAIN=" + toolchain]
  if cmsis_path is not None:
      params_list += ["CMSIS_PATH=" + cmsis_path]
  if optimized_kernel_dir is not None:
      params_list += ["OPTIMIZED_KERNEL_DIR=" + optimized_kernel_dir]

  params_list += additional_make_flags

  process = subprocess.Popen(params_list,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  _, stderr = process.communicate()
  if process.returncode != 0:
    raise RuntimeError("Building %s failed with \n\n %s" %
                       (" ".join(params_list), stderr.decode()))


def _profile_a_binary(root_dir, binary_name, makefile_options, build_info, save_top_path):
  target_dir = "%s_%s_%s" % (makefile_options["TARGET"],
                             makefile_options["TARGET_ARCH"],
                             makefile_options["BUILD_TYPE"])
  binary_path = os.path.join(root_dir, 'tensorflow/lite/micro/tools/make/gen/',
                             target_dir, 'bin', binary_name)
  if save_top_path is None:
    csv_folder_path = os.path.join(root_dir, 'data/continuous_builds/size_profiling',
                          target_dir)
  else:
    csv_folder_path = os.path.join(save_top_path, target_dir)

  if not os.path.exists(csv_folder_path):
    os.makedirs(csv_folder_path)
  csv_path = os.path.join(csv_folder_path, "%s.csv" % binary_name)


  # Run size command and extract the output
  process = subprocess.Popen(["size", binary_path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  if process.returncode != 0:
    raise RuntimeError("size %s failed with \n\n %s" %
                       (binary_name, stderr.decode()))

  output_str = stdout.decode()
  df = pd.DataFrame([line.split() for line in output_str.split('\n')[1:]],
                    columns=list(output_str.split('\n')[0].split()))

  # Append the output from the size to the CSV file
  report = _create_or_read_csv(csv_path)
  report.loc[len(report.index)] = [
      build_info["date"], build_info['sha'], df['text'][0], df['data'][0],
      df['bss'][0], df['dec'][0]
  ]
  report.to_csv(csv_path, index=False, header=False, mode='a')


def _create_or_read_csv(csv_file_name):
  if os.path.exists(csv_file_name) is not True:
    csv_df = pd.DataFrame(
        columns=['date', 'sha', 'text', 'data', 'bss', 'total'])
    csv_df.to_csv(csv_file_name, index=False, mode='w')

  csv_head = pd.read_csv(csv_file_name, index_col=False, nrows=0)
  return csv_head


def _get_build_info(root_dir, git_dir):
  os.chdir(root_dir)

  git_process = subprocess.Popen(["git", "rev-parse", "HEAD"],
                                 stdout=subprocess.PIPE,
                                 cwd=root_dir)
  sha, err = git_process.communicate()
  if git_process.returncode != 0:
    raise RuntimeError("Git failed with %s" % err.decode())

  if git_dir is not None:
    git_commit_date_process = subprocess.Popen(["git", "show", "-s", "--format=%ci"],
                                                stdout=subprocess.PIPE,
                                                cwd=git_dir)
    datestr, err = git_commit_date_process.communicate()
    if git_commit_date_process.returncode != 0:
      raise RuntimeError("Git failed with %s" % err.decode())
    datestr = datestr.decode().strip('\n')
    current_time = datestr[:20]
  else:
    current_time = str(datetime.datetime.now())

  return {'date': current_time, 'sha': sha.decode().strip('\n')}


def _build_and_profile(root_dir, makefile_options, binary_names, toolchain, additional_make_flags, cmsis_path, \
  optimized_kernel_dir, save_top_path, git_dir):
  build_info = _get_build_info(root_dir, git_dir)
  for binary_name in binary_names:
    _build_a_binary(root_dir, binary_name, makefile_options, toolchain, additional_make_flags, cmsis_path, optimized_kernel_dir)
    _profile_a_binary(root_dir, binary_name, makefile_options, build_info, save_top_path)


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
  parser.add_argument('--build_type',
                      nargs='?',
                      const='release',
                      default='release',
                      help='build type (e.g. release)')
  parser.add_argument('--target',
                      nargs='?',
                      const='linux',
                      default='linux',
                      help='host target (e.g. linux)')
  parser.add_argument('--target_arch',
                      nargs='?',
                      const='x86_64',
                      default='x86_64',
                      help='target architecture (e.g x86_64)')
  parser.add_argument('--toolchain',
                      nargs='?',
                      default=None,
                      help='toolchain (optional)')
  parser.add_argument('--make_flags',
                      nargs=1,
                      help='additional make flags (include leading dash if necessary)')
  parser.add_argument('--relative_root_dir',
                      nargs='?',
                      default='../../../../..',
                      help='root directory relative to this pythin script\'s dir')
  parser.add_argument('--optimized_kernel_dir',
                      nargs='?',
                      default=None,
                      help='select another kernel than the default')
  parser.add_argument('--cmsis_path',
                      nargs='?',
                      default=None,
                      help='CMSIS-specific setting')
  parser.add_argument('--save_top_path',
                      nargs='?',
                      default=None,
                      help='save directory for memory summary')
  parser.add_argument('--git_dir',
                      nargs='?',
                      default=None,
                      type=str,
                      help='Path from where git commit date is logged')

  args = parser.parse_args()

  makefile_options = {
      "BUILD_TYPE": args.build_type,
      "TARGET": args.target,
      "TARGET_ARCH": args.target_arch
  }
  binary_names = args.binary_list.split(',')
  toolchain = args.toolchain
  additional_make_flags=args.make_flags

  script_path = os.path.dirname(os.path.realpath(__file__))
  root_dir = os.path.join(script_path, args.relative_root_dir)

  cmsis_path = args.cmsis_path
  optimized_kernel_dir = args.optimized_kernel_dir
  save_top_path = args.save_top_path

  git_dir = args.git_dir

  _build_and_profile(root_dir, makefile_options, binary_names, toolchain, additional_make_flags, \
    cmsis_path, optimized_kernel_dir, save_top_path, git_dir)
