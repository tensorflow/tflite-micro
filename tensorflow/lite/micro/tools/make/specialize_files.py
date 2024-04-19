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
import os


# Selects the more specialized files in directory in favor of the file with the
# same name in base_file_list and returns a list containing all the files as a
# result of this specialization merge.
def _specialize_files(base_file_list, directory, slash_type):
  # If the specialized directory is not a valid path, then return the
  # base_file_list.
  if not os.path.isdir(directory):
    return base_file_list

  specialize_files = os.listdir(directory)
  specialized_list = []
  for fpath in base_file_list:
    fname = os.path.basename(fpath)
    if fname in specialize_files:
      if slash_type == "posix":
        import posixpath
        p = posixpath.join(directory, fname)
      elif slash_type == "native":
        p = os.path.join(directory, fname)
      else:
        raise ValueError("Unknown slash type: " + slash_type)
      specialized_list.append(p)
    else:
      specialized_list.append(fpath)
  return specialized_list


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Helper functions used during the Makefile build")

  parser.add_argument(
      "--base_files",
      default="",
      help="String with (space separated) list of all the files "
      "to attempt to specialize.")

  parser.add_argument("--specialize_directory",
                      default="",
                      help="Directory containing the more specialized files.")
  
  parser.add_argument("--slash_type", default="posix" if os.name == "nt" else "native", help="Slash type, native or posix")

  args = parser.parse_args()

  assert(args.slash_type.lower() in ["native", "posix"])

  if args.base_files != "" and args.specialize_directory != "":
    print(" ".join(
        _specialize_files(args.base_files.split(), args.specialize_directory, args.slash_type.lower())))
