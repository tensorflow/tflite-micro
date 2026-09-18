#!/usr/bin/env python3
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Checks Apache 2.0 license notice on source files."""

from pathlib import Path
import re
import subprocess
import sys

TARGET_DIRS = [
  "tensorflow/lite/kernels/internal/reference",
  "tensorflow/lite/micro",
  "third_party",
]

EXCLUDES = (
  "kernels/internal/reference/integer_ops/",
  "kernels/internal/reference/reference_ops.h",
  "python/schema_py_generated.py",
  "python_requirements.in",
  "tensorflow/lite/micro/compression/metadata_saved.h",
  "tools/make/downloads",
  "tools/make/targets/ecm3531",
  "BUILD",
  "leon_commands",
  "LICENSE",
  ".gitignore",
  ".bmp",
  ".bzl",
  ".csv",
  ".h5",
  ".inc",
  ".ipynb",
  ".md",
  ".patch",
  ".properties",
  ".tflite",
  ".tpl",
  ".txt",
  ".wav",
  ".png",
  ".jpg",
  ".json",
  ".lock",
  ".toml",
)

COPYRIGHT_REGEX = re.compile(
  r"Copyright 20\d\d The TensorFlow Authors\. All Rights Reserved\."
)
LICENSE_KEYWORD = "Apache License, Version 2.0"


def main():
  repo_root = Path(__file__).resolve().parents[5]
  cmd = ["git", "-C", str(repo_root), "ls-files"] + TARGET_DIRS
  files = subprocess.check_output(cmd, text=True).splitlines()

  missing = []
  for rel_path in files:
    if any(ex in rel_path or rel_path.endswith(ex) for ex in EXCLUDES):
      continue

    file_path = repo_root / rel_path
    try:
      content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
      continue

    header = "\n".join(content.splitlines()[:30])
    if not (COPYRIGHT_REGEX.search(header) and LICENSE_KEYWORD in header):
      missing.append(rel_path)

  if missing:
    print(f"FAILED: License header missing in {len(missing)} file(s):")
    for f in missing:
      print(f"  {f}")
    return 1

  print(f"PASSED: License check ({len(files)} files scanned)")
  return 0


if __name__ == "__main__":
  sys.exit(main())
