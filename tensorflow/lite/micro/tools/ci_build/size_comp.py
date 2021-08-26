#!/usr/bin/env python3
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
import json
import sys


def berkeley_size_format_to_dict(berkeley_size_format):
  lines = berkeley_size_format.split('\n')
  labels = lines[0].split()
  values = lines[1].split()
  outdict = {labels[i]: values[i] for i in range(len(labels) - 2)}
  return (outdict)


def json_to_dict(some_json):
  outdict = json.loads(some_json)
  return (outdict)


def file_to_dict(a_file):
  with open(a_file) as the_file:
    contents = the_file.read()
  if contents[0] == "{":
    retdict = json_to_dict(contents)
  else:
    retdict = berkeley_size_format_to_dict(contents)

  return (retdict)


def compare_val_in_files(old_file, new_file, val='bss'):
  old_dict = file_to_dict(old_file)
  new_dict = file_to_dict(new_file)

  if int(new_dict[val]) > int(old_dict[val]):
    print(val, " larger than previous value")
    print("old: ", old_dict[val])
    print("new: ", new_dict[val])
    print("=====Check failed=====")
    sys.exit(1)

  print(val)
  print("old: ", old_dict[val])
  print("new: ", new_dict[val])
  print("Check Passed")

  return ()


def compare_all_val_in_files(old_file, new_file, error_on_mem_increase):
  old_dict = file_to_dict(old_file)
  new_dict = file_to_dict(new_file)
  any_mem_increase = False
  for section, val in old_dict.items():
    if int(new_dict[section]) > int(old_dict[section]):
      print(section, " larger than previous value")
      print("old: ", old_dict[section])
      print("new: ", new_dict[section])
      any_mem_increase = True
    else:
      print(section)
      print("old: ", old_dict[section])
      print("new: ", new_dict[section])

  if any_mem_increase:
    print("Warning: memory footprint increases!")
    if error_on_mem_increase:
      print("Error on memory footprint increase!!")
      sys.exit(1)

  return ()


def berkeley_size_format_to_json_file(input_file, output_file):
  output_dict = file_to_dict(input_file)
  with open(output_file, 'w') as outfile:
    json.dump(output_dict, outfile)

  return ()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-t",
      "--transform",
      help="transform a berkeley size format file to a json file",
      nargs=2)
  parser.add_argument("-c",
                      "--compare",
                      help="compare value in old file to new file",
                      nargs=2)
  parser.add_argument("-v",
                      "--value",
                      default="bss",
                      help="value to be compared")
  parser.add_argument("-a",
                      "--compare_all",
                      help="compare all value in old file to new file",
                      nargs=2)
  parser.add_argument("-e",
                      "--error_on_mem_increase",
                      default=False,
                      action="store_true",
                      help="error exit on memory footprint increase")
  args = parser.parse_args()

  if args.transform:
    berkeley_size_format_to_json_file(args.transform[0], args.transform[1])

  if args.compare:
    compare_val_in_files(args.compare[0], args.compare[1], args.value)

  if args.compare_all:
    compare_all_val_in_files(args.compare_all[0], args.compare_all[1],
                             args.error_on_mem_increase)
