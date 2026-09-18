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
"""Downloads, verifies, extracts, and patches third-party dependencies for TFLM.

This script ensures robust, atomic, and race-free dependency management
for Makefile builds, compatible with Python 3.6+.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile


class Dependency:
  def __init__(
    self,
    name,
    url="",
    md5=None,
    patch=None,
    git_repo=None,
    git_commit=None,
    dep_type="archive",
  ):
    self.name = name
    self.url = url
    self.md5 = md5
    self.patch = patch
    self.git_repo = git_repo
    self.git_commit = git_commit
    self.type = dep_type


DEPENDENCIES = {
  "gemmlowp": Dependency(
    name="gemmlowp",
    url="https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip",
    md5="7e8191b24853d75de2af87622ad293ba",
    dep_type="archive",
  ),
  "ruy": Dependency(
    name="ruy",
    url="https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip",
    md5="abf7a91eb90d195f016ebe0be885bb6e",
    dep_type="archive",
  ),
  "flatbuffers": Dependency(
    name="flatbuffers",
    url="https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.zip",
    md5="023eca1e211d64007124420cd6be29c7",
    patch="tensorflow/lite/micro/tools/make/flatbuffers.patch",
    dep_type="archive",
  ),
  "kissfft": Dependency(
    name="kissfft",
    url="https://github.com/mborgerding/kissfft/archive/refs/tags/v130.zip",
    md5="438ba1fef5783cc5f5f201395cc477ca",
    patch="third_party/kissfft/kissfft.patch",
    dep_type="archive",
  ),
  "eyalroz_printf": Dependency(
    name="eyalroz_printf",
    url="https://github.com/eyalroz/printf/archive/f8ed5a9bd9fa8384430973465e94aa14c925872d.zip",
    md5="5772534c1d6f718301bca1fefaba28f3",
    dep_type="archive",
  ),
}

DEFAULT_DEPENDENCIES = [
  "gemmlowp",
  "ruy",
  "flatbuffers",
  "kissfft",
  "eyalroz_printf",
]
STAMP_FILENAME = ".download_complete"


def compute_file_hash(filepath):
  hasher = hashlib.md5()
  with open(str(filepath), "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def compute_stamp_metadata(dep, tensorflow_root):
  metadata = {
    "name": dep.name,
    "type": dep.type,
    "url": dep.url,
    "md5": dep.md5,
    "git_repo": dep.git_repo,
    "git_commit": dep.git_commit,
  }
  if dep.patch:
    patch_path = tensorflow_root / dep.patch
    if patch_path.exists():
      metadata["patch_md5"] = compute_file_hash(patch_path)
    else:
      metadata["patch_md5"] = None
  return metadata


def is_already_downloaded(dep, dest_dir, tensorflow_root):
  stamp_path = dest_dir / STAMP_FILENAME
  if not dest_dir.is_dir() or not stamp_path.exists():
    return False
  try:
    with open(str(stamp_path), "r") as f:
      existing_metadata = json.load(f)
    expected_metadata = compute_stamp_metadata(dep, tensorflow_root)
    return existing_metadata == expected_metadata
  except Exception:
    return False


def delete_build_files(directory):
  for root, _, files in os.walk(str(directory)):
    for file in files:
      if file in ("BUILD", "BUILD.bazel"):
        try:
          os.remove(os.path.join(root, file))
        except OSError:
          pass


def download_url_with_retry(url, output_path, max_retries=5):
  for attempt in range(1, max_retries + 1):
    try:
      req = urllib.request.Request(
        url,
        headers={"User-Agent": "TFLM-Downloader/1.0"},
      )
      with urllib.request.urlopen(req, timeout=30) as response:
        with open(str(output_path), "wb") as out_file:
          shutil.copyfileobj(response, out_file)
      return
    except Exception as e:
      if attempt == max_retries:
        raise RuntimeError(
          "Failed to download {} after {} attempts: {}".format(
            url, max_retries, e
          )
        )
      time.sleep(min(2**attempt, 8))


def extract_archive(archive_path, extract_dir):
  temp_extract = extract_dir.parent / (str(extract_dir.name) + "_unpacked")
  temp_extract.mkdir(parents=True, exist_ok=True)
  try:
    if zipfile.is_zipfile(str(archive_path)):
      with zipfile.ZipFile(str(archive_path), "r") as zf:
        zf.extractall(str(temp_extract))
    elif tarfile.is_tarfile(str(archive_path)):
      with tarfile.open(str(archive_path), "r:*") as tf:
        tf.extractall(str(temp_extract))
    else:
      raise ValueError("Unsupported archive format: {}".format(archive_path))

    entries = [p for p in temp_extract.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
      src_dir = entries[0]
    else:
      src_dir = temp_extract

    extract_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
      shutil.move(str(item), str(extract_dir))
  finally:
    shutil.rmtree(str(temp_extract), ignore_errors=True)


def apply_patch(patch_path, target_dir):
  if not patch_path.exists():
    raise FileNotFoundError("Patch file not found: {}".format(patch_path))

  patch_err = ""
  try:
    with open(str(patch_path), "rb") as patch_in:
      res = subprocess.run(
        ["patch", "-p1", "-d", str(target_dir)],
        stdin=patch_in,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
      )
    if res.returncode == 0:
      return
    patch_err = res.stderr.decode("utf-8", errors="replace")
  except FileNotFoundError:
    patch_err = "'patch' command not found"

  # Fallback to git apply if patch command failed or was not found
  git_cmd = [
    "git",
    "apply",
    "--ignore-space-change",
    "--ignore-whitespace",
    str(patch_path),
  ]
  try:
    res2 = subprocess.run(
      git_cmd,
      cwd=str(target_dir),
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
    )
    if res2.returncode != 0:
      raise RuntimeError(
        "Failed to apply patch {}:\npatch output:\n{}\ngit output:\n{}".format(
          patch_path,
          patch_err,
          res2.stderr.decode("utf-8", errors="replace"),
        )
      )
  except FileNotFoundError:
    raise RuntimeError(
      "Failed to apply patch {}: neither 'patch' nor 'git' commands were found.".format(
        patch_path
      )
    )


def download_and_prepare(dep, downloads_dir, tensorflow_root):
  dest_dir = downloads_dir / dep.name
  if is_already_downloaded(dep, dest_dir, tensorflow_root):
    return

  sys.stderr.write("Downloading and setting up {}...\n".format(dep.name))
  downloads_dir.mkdir(parents=True, exist_ok=True)
  staging_dir = Path(
    tempfile.mkdtemp(prefix=".tmp_{}_".format(dep.name), dir=str(downloads_dir))
  )

  try:
    if dep.type == "archive":
      temp_archive = downloads_dir / ".dl_{}_{}".format(dep.name, os.getpid())
      try:
        download_url_with_retry(dep.url, temp_archive)

        if dep.md5 and dep.md5 != "SKIP_MD5_CHECK":
          actual_md5 = compute_file_hash(temp_archive)
          if actual_md5 != dep.md5:
            raise ValueError(
              "MD5 mismatch for {}: expected {}, got {}".format(
                dep.name, dep.md5, actual_md5
              )
            )

        extract_archive(temp_archive, staging_dir)
      finally:
        if temp_archive.exists():
          temp_archive.unlink()

    elif dep.type == "git":
      git_clone_cmd = [
        "git",
        "clone",
        dep.git_repo,
        str(staging_dir / dep.name),
      ]
      try:
        subprocess.run(
          git_clone_cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          check=True,
        )
      except subprocess.CalledProcessError as e:
        raise RuntimeError(
          "Failed to clone git repo {}:\n{}".format(
            dep.git_repo, e.stderr.decode("utf-8", errors="replace")
          )
        )
      cloned_dir = staging_dir / dep.name
      if dep.git_commit:
        try:
          subprocess.run(
            ["git", "checkout", dep.git_commit],
            cwd=str(cloned_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
          )
        except subprocess.CalledProcessError as e:
          raise RuntimeError(
            "Failed to checkout commit {} in {}:\n{}".format(
              dep.git_commit,
              dep.git_repo,
              e.stderr.decode("utf-8", errors="replace"),
            )
          )
      git_dir = cloned_dir / ".git"
      if git_dir.exists():
        shutil.rmtree(str(git_dir), ignore_errors=True)
      # Move contents of cloned_dir to staging_dir
      for item in cloned_dir.iterdir():
        shutil.move(str(item), str(staging_dir))
      cloned_dir.rmdir()

    delete_build_files(staging_dir)

    if dep.patch:
      patch_path = tensorflow_root / dep.patch
      apply_patch(patch_path, staging_dir)

    metadata = compute_stamp_metadata(dep, tensorflow_root)
    with open(str(staging_dir / STAMP_FILENAME), "w") as f:
      json.dump(metadata, f, indent=2)

    # Check if another process completed the download in the meantime
    if is_already_downloaded(dep, dest_dir, tensorflow_root):
      shutil.rmtree(str(staging_dir), ignore_errors=True)
      return

    # Atomic promotion
    if dest_dir.exists():
      shutil.rmtree(str(dest_dir), ignore_errors=True)
    staging_dir.rename(dest_dir)
    sys.stderr.write("Successfully installed {}.\n".format(dep.name))

  except Exception:
    shutil.rmtree(str(staging_dir), ignore_errors=True)
    raise


def main():
  parser = argparse.ArgumentParser(
    description="Download third-party dependencies for TFLM."
  )
  parser.add_argument(
    "--downloads_dir",
    default="tensorflow/lite/micro/tools/make/downloads",
    help="Directory to unpack dependencies into",
  )
  parser.add_argument(
    "--tensorflow_root",
    default="",
    help="Root directory of TensorFlow / TFLM tree",
  )
  parser.add_argument(
    "dependencies",
    nargs="*",
    help="Specific dependencies to download (defaults to all core deps)",
  )
  args = parser.parse_args()

  downloads_dir = Path(args.downloads_dir).resolve()
  tensorflow_root = Path(args.tensorflow_root).resolve()

  target_deps = args.dependencies if args.dependencies else DEFAULT_DEPENDENCIES

  # Validate dependency names
  for name in target_deps:
    if name not in DEPENDENCIES:
      sys.stderr.write(
        "Unknown dependency: {}. Available: {}\n".format(
          name, list(DEPENDENCIES.keys())
        )
      )
      return 1

  # Filter out dependencies that are already satisfied
  to_download = [
    name
    for name in target_deps
    if not is_already_downloaded(
      DEPENDENCIES[name], downloads_dir / name, tensorflow_root
    )
  ]

  if to_download:
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=min(len(to_download), 4)
    ) as executor:
      futures = {
        executor.submit(
          download_and_prepare,
          DEPENDENCIES[name],
          downloads_dir,
          tensorflow_root,
        ): name
        for name in to_download
      }
      for future in concurrent.futures.as_completed(futures):
        dep_name = futures[future]
        try:
          future.result()
        except Exception as e:
          sys.stderr.write("Error downloading {}: {}\n".format(dep_name, e))
          return 1

  # Print SUCCESS on stdout as expected by Makefile convention
  print("SUCCESS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
