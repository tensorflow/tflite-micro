#!/usr/bin/env python3

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# ----
"""
 A filter that expands Bazel workspace stamp variables.

 For example, the input steam:

     This build was compiled at {BUILD_DATE}.

 is expanded into the output stream:

     This build was compiled at 2023-02-10T14:15.

 Stamp variable key-value pairs are read from all files passed as positional
 arguments. These files are typically bazel-out/stable-status.txt and
 bazel-out/volatile-status.txt. See the Bazel documentation for the option
 --workspace_status_command.
"""

import sys


def read_stamps(file):
  """Return a dictionary of key-value pairs read from a stamp file.

  These files are typically bazel-out/stable-status.txt and
  bazel-out/volatile-status.txt. See the Bazel documentation for the option
  --workspace_status_command."""

  stamps = {}
  for line in file:
    try:
      key, value = line.split(" ", maxsplit=1)
      stamps[key] = value.strip()
    except ValueError:
      pass  # Skip blank lines, etc.

  return stamps


def expand(istream, ostream, stamps):
  """Write istream to ostream, expanding placeholders like {KEY}."""
  for line in istream:
    for key, value in stamps.items():
      line = line.replace(f"{{{key}}}", value)
    ostream.write(line)


def _main():
  """Stamp variables are read from all files passed as positional arguments."""
  stamps = {}
  for name in sys.argv[1:]:
    with open(name) as f:
      stamps.update(read_stamps(f))

  expand(sys.stdin, sys.stdout, stamps)

  sys.exit(0)


if __name__ == "__main__":
  _main()
