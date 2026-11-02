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
"""Checks Apache 2.0 license notice on source files and BUILD license metadata."""

import ast
from pathlib import Path
import re
import subprocess
import sys

TARGET_DIRS = [
  "tensorflow/lite/micro",
  "third_party",
]

EXCLUDES = (
  "python_requirements.in",
  "tools/make/downloads",
  "BUILD",
  "LICENSE",
  "_generated.h",
  "_generated.py",
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
REQUIRED_BUILD_LICENSE = "//:license"


def has_default_applicable_license(content):
  try:
    tree = ast.parse(content)
  except SyntaxError:
    return False

  for node in tree.body:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
      continue
    func = node.value.func
    if not (isinstance(func, ast.Name) and func.id == "package"):
      continue
    for kw in node.value.keywords:
      if kw.arg == "default_applicable_licenses" and isinstance(
        kw.value, (ast.List, ast.Tuple)
      ):
        labels = [
          elt.value
          for elt in kw.value.elts
          if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
        if REQUIRED_BUILD_LICENSE in labels:
          return True
  return False


def check_build_licenses(repo_root):
  cmd = [
    "git",
    "-C",
    str(repo_root),
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
  ]
  all_files = subprocess.check_output(cmd, text=True).splitlines()

  build_files = []
  missing_license = []
  for rel_path in sorted(set(all_files)):
    if (
      rel_path.startswith("third_party/") or "tools/make/downloads" in rel_path
    ):
      continue
    if Path(rel_path).name not in ("BUILD", "BUILD.bazel"):
      continue
    file_path = repo_root / rel_path
    if not file_path.is_file():
      continue
    build_files.append(rel_path)
    content = file_path.read_text(encoding="utf-8", errors="replace")
    if not has_default_applicable_license(content):
      missing_license.append(rel_path)

  return build_files, missing_license


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

  build_files, missing_build_license = check_build_licenses(repo_root)

  failed = False
  if missing:
    print(f"FAILED: License header missing in {len(missing)} file(s):")
    for f in missing:
      print(f"  {f}")
    failed = True

  if missing_build_license:
    print(
      "FAILED: Missing package(default_applicable_licenses = "
      f'["{REQUIRED_BUILD_LICENSE}"]) in {len(missing_build_license)} BUILD '
      "file(s):"
    )
    for f in missing_build_license:
      print(f"  {f}")
    failed = True

  if failed:
    return 1

  print(
    f"PASSED: License check ({len(files)} files scanned, "
    f"{len(build_files)} BUILD files verified)"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
