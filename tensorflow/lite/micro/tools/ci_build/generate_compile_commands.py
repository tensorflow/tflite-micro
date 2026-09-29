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
"""Generates compile_commands.json using Bazel aquery."""

import argparse
import json
import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.environ.get(
  "BUILD_WORKSPACE_DIRECTORY",
  os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../..")),
)


def parse_args():
  parser = argparse.ArgumentParser(
    description="Generate compile_commands.json from Bazel aquery."
  )
  parser.add_argument(
    "--output",
    "-o",
    default="compile_commands.json",
    help="Output file path for compilation database.",
  )
  parser.add_argument(
    "--targets",
    "-t",
    default="//...",
    help="Bazel targets expression to query.",
  )
  parser.add_argument(
    "--build-generated",
    action="store_true",
    default=True,
    help="Build generated schema and flatbuffers headers required by Clang.",
  )
  parser.add_argument(
    "--no-build-generated",
    dest="build_generated",
    action="store_false",
    help="Do not build generated headers.",
  )
  return parser.parse_args()


def build_generated_headers(repo_root):
  """Build generated targets needed by Clang tooling."""
  targets = [
    "//tensorflow/lite/micro:micro_framework",
    "//tensorflow/lite/micro/tools:layer_by_layer_schema",
  ]
  print(f"Building targets for headers: {' '.join(targets)}...")
  subprocess.run(["bazel", "build"] + targets, cwd=repo_root, check=True)


def setup_external_symlink(repo_root):
  """Symlinks 'external' to Bazel's execution_root/external.

  In Bazel 8 (bzlmod), external headers are referenced via paths like
  '-iquote external/+_repo_rules+gemmlowp'. Bazel creates this external/
  directory in execution_root, not the workspace root. Creating this symlink
  allows Clang tooling running from workspace root to resolve external headers.
  """
  try:
    result = subprocess.run(
      ["bazel", "info", "execution_root"],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=repo_root,
      check=True,
    )
    exec_root = result.stdout.strip()
    exec_external = os.path.join(exec_root, "external")
    target_link = os.path.join(repo_root, "external")
    if os.path.exists(exec_external):
      if os.path.islink(target_link) or os.path.lexists(target_link):
        try:
          if (
            os.path.islink(target_link)
            and os.readlink(target_link) == exec_external
          ):
            return
          os.unlink(target_link)
        except OSError as e:
          print(f"Warning: Could not remove old external link: {e}")
          return
      elif os.path.isdir(target_link):
        print(f"Warning: {target_link} is a directory, not a symlink.")
        return
      os.symlink(exec_external, target_link)
      print(f"Symlinked {target_link} -> {exec_external}")
  except Exception as e:
    print(f"Warning: Failed to set up external symlink: {e}")


def generate_compilation_database(output_path, targets, repo_root):
  """Queries Bazel aquery and outputs compile_commands.json."""
  query = f"mnemonic(CppCompile, {targets})"
  cmd = ["bazel", "aquery", query, "--output=jsonproto"]
  print(f"Running bazel aquery: {' '.join(cmd)}...")
  result = subprocess.run(
    cmd, stdout=subprocess.PIPE, cwd=repo_root, check=True
  )

  aquery_data = json.loads(result.stdout)
  compile_commands = []
  seen_sources = set()

  for action in aquery_data.get("actions", []):
    args = action.get("arguments", [])
    source = None
    for i, arg in enumerate(args):
      if arg == "-c" and i + 1 < len(args):
        source = args[i + 1]
        break

    if not source:
      continue

    # Deduplicate sources if compiled multiple times (e.g. pic vs non-pic)
    if source in seen_sources:
      continue
    seen_sources.add(source)

    # Filter out GCC-specific flags that Clang doesn't recognize
    unsupported_flags = {
      "-fno-canonical-system-headers",
      "-Wno-free-nonheap-object",
    }
    filtered_args = [a for a in args[1:] if a not in unsupported_flags]

    # Select compiler based on file type and add warning suppression flag
    compiler = "clang" if source.endswith(".c") else "clang++"
    clang_args = [compiler] + filtered_args
    clang_args.append("-Wno-unknown-warning-option")

    compile_commands.append(
      {
        "directory": repo_root,
        "file": source,
        "arguments": clang_args,
      }
    )

  print(
    f"Writing {len(compile_commands)} compilation entries to {output_path}..."
  )
  if os.path.lexists(output_path):
    try:
      os.remove(output_path)
    except OSError:
      pass
  with open(output_path, "w") as f:
    json.dump(compile_commands, f, indent=2)


def main():
  args = parse_args()
  output_path = args.output
  if not os.path.isabs(output_path):
    output_path = os.path.join(ROOT_DIR, output_path)

  if args.build_generated:
    build_generated_headers(ROOT_DIR)
  setup_external_symlink(ROOT_DIR)
  generate_compilation_database(output_path, args.targets, ROOT_DIR)
  print("Done generating compilation database.")


if __name__ == "__main__":
  main()
