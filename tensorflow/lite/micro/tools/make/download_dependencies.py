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

Provides race-free, atomic, SHA-256-verified dependency management for TFLM
Makefile builds using only the Python standard library.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

try:
  import fcntl
except ImportError:
  fcntl = None  # type: ignore[assignment]

try:
  import msvcrt
except ImportError:
  msvcrt = None  # type: ignore[assignment]


def _query_umask():
  current = os.umask(0o022)
  os.umask(current)
  return current


UMASK = _query_umask()
STAMP_FILENAME = ".download_complete"
STAMP_SCHEMA_VERSION = 2
_FORCE_SYMLINK_COPY_FALLBACK = False
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS = {}


@dataclasses.dataclass(frozen=True)
class SubArchiveSpec:
  urls: tuple[str, ...]
  sha256: str
  subdir: str


@dataclasses.dataclass(frozen=True)
class ArchiveSpec:
  urls: tuple[str, ...]
  sha256: str
  sub_archives: tuple[SubArchiveSpec, ...] = ()


@dataclasses.dataclass(frozen=True)
class DependencySpec:
  name: str
  archives: ArchiveSpec | dict[tuple[str, str], ArchiveSpec]
  patch: str | None = None
  hook: str | None = None
  hook_version: int = 1
  ndsp_core_name: str | None = None

  def resolve_archive(self, host_os, host_arch):
    if isinstance(self.archives, ArchiveSpec):
      return self.archives
    for key in ((host_os, host_arch), (host_os, "*"), ("*", "*")):
      if key in self.archives:
        return self.archives[key]
    raise RuntimeError(
      "Dependency '{}' is not supported on host platform ({}, {})".format(
        self.name, host_os, host_arch
      )
    )


_ARM_GNU_REL = "14.3.rel1"
_ARM_GNU_PRIMARY = (
  f"https://developer.arm.com/-/media/Files/downloads/gnu/{_ARM_GNU_REL}/binrel"
)
_ARM_GNU_MIRROR = (
  "https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu/"
  f"{_ARM_GNU_REL}/binrel"
)
_CORSTONE_BASE = (
  "https://developer.arm.com/-/media/Arm%20Developer%20Community/"
  "Downloads/OSS/FVP/Corstone-300"
)
_TF_GCS_MIRROR = "https://storage.googleapis.com/mirror.tensorflow.org"

DEPENDENCIES = {
  "flatbuffers": DependencySpec(
    name="flatbuffers",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.tar.gz",
      ),
      sha256="9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada",
    ),
  ),
  "kissfft": DependencySpec(
    name="kissfft",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/mborgerding/kissfft/archive/refs/tags/131.2.0.tar.gz",
      ),
      sha256="205a8f6a448ef12b091f8ac6a514b5091bb5f6b0b543431ed75f673116cf5cbf",
    ),
  ),
  "eyalroz_printf": DependencySpec(
    name="eyalroz_printf",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/eyalroz/printf/archive/f8ed5a9bd9fa8384430973465e94aa14c925872d.zip",
      ),
      sha256="5c7cffe6f83734c2cb226a58bbd2467273334e6df7ba64c5cd79b020d6d607df",
    ),
  ),
  "cmsis": DependencySpec(
    name="cmsis",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/ARM-software/CMSIS_6/archive/5782d6f8057906d360f4b95ec08a2354afe5c9b9.zip",
      ),
      sha256="7d1de644243e370e8d1d798a0dfa8c8d67ab1121082816fbfe75bdb2183aac54",
      sub_archives=(
        SubArchiveSpec(
          urls=(
            "https://github.com/ARM-software/Cortex_DFP/archive/c2c70a97a20fb355815e2ead3d4a40e35a4a3cdf.zip",
          ),
          sha256="fee71b073e3805ebbfed76b8e14128ae3b963a05acfefa3735529d00c3429a55",
          subdir="Cortex_DFP",
        ),
      ),
    ),
  ),
  "cmsis_nn": DependencySpec(
    name="cmsis_nn",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/ARM-software/CMSIS-NN/archive/18bb0fd4ecea4b17a5a3c444c8d3a8cf28b1e68f.zip",
      ),
      sha256="5212f41b7c2789816fb42f15e53ade4f0729d3780c18e7eaa52c8a00cf00fb8b",
    ),
  ),
  "ethos_u_core_driver": DependencySpec(
    name="ethos_u_core_driver",
    archives=ArchiveSpec(
      urls=(
        "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver/-/archive/9622608a5cc318c0933bcce720b59737d03bfb6f/ethos-u-core-driver-9622608a5cc318c0933bcce720b59737d03bfb6f.tar.gz",
      ),
      sha256="fa062fd4cca1c32c60db47a3f711935a3ed35c8a0e4cd7bbb24369246bc77c58",
    ),
  ),
  "ethos_u_core_platform": DependencySpec(
    name="ethos_u_core_platform",
    archives=ArchiveSpec(
      urls=(
        "https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-platform/-/archive/e25a89dec1cf990f3168dbd6c565e3b0d51cb151/ethos-u-core-platform-e25a89dec1cf990f3168dbd6c565e3b0d51cb151.tar.gz",
      ),
      sha256="37eca66103086990729faec0dba8c6d020639f8e35c0fc9ec8f5c8b801069cff",
    ),
    patch="tensorflow/lite/micro/tools/make/ethos_u_core_platform.patch",
    hook="ethos_u_core_platform",
  ),
  "xa_nnlib_hifi4": DependencySpec(
    name="xa_nnlib_hifi4",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-xtensa/nnlib-hifi4/raw/master/archive/xa_nnlib_hifi4_09_03_2026.zip",
      ),
      sha256="8b490e8a2f0e28e36dc5f244b4a256b5bcba393c629bf73d3e6e50817fe719af",
    ),
  ),
  "xa_nnlib_hifi5": DependencySpec(
    name="xa_nnlib_hifi5",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-xtensa/nnlib-hifi5/raw/master/archive/xa_nnlib_hifi5_09_03_2026.zip",
      ),
      sha256="b96d396577813163b0271bd611aa674d5a834666986e2ace59d244d8c97544ee",
    ),
  ),
  "xi_tflmlib_vision_p6": DependencySpec(
    name="xi_tflmlib_vision_p6",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-xtensa/tflmlib_vision/raw/main/archive/xi_tflmlib_vision_p6_22_06_29.zip",
      ),
      sha256="b39f6d25305afec534fff806e7531151a5b15a23f4313392801f2754a30cc461",
    ),
    patch="tensorflow/lite/micro/tools/make/ext_libs/xi_tflmlib_vision_p6.patch",
  ),
  "ndsplib-hifi3": DependencySpec(
    name="ndsplib-hifi3",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-xtensa/ndsplib-hifi3/archive/d17bf205dc530a9e1a1d979249520f4401529db1.zip",
      ),
      sha256="f183f77ca77d730697f95a023ba12ebb695e571e8423219080535821dbceaad7",
    ),
    patch="tensorflow/lite/micro/tools/make/ext_libs/ndsplib-hifi3.patch",
    hook="ndsplib",
    ndsp_core_name="HiFi3",
  ),
  "ndsplib-hifi4": DependencySpec(
    name="ndsplib-hifi4",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-xtensa/ndsplib-hifi4/archive/aba2485ba12d9851fa398bcb5c18c05cc3731a17.zip",
      ),
      sha256="a6230204104b6dab423f246b41930c94f85d2977fc207e8882fcece74b280dac",
    ),
    patch="tensorflow/lite/micro/tools/make/ext_libs/ndsplib-hifi4.patch",
    hook="ndsplib",
    ndsp_core_name="HiFi4",
  ),
  "ndsplib-hifi5": DependencySpec(
    name="ndsplib-hifi5",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-xtensa/ndsplib-hifi5/archive/01c92ceb26cc0a598c6d83d17c3d88363bd8f7fc.zip",
      ),
      sha256="efd6976c64f104657239d87adf5d6a58d5cc17947596ad54e4948c405bc232f9",
    ),
    patch="tensorflow/lite/micro/tools/make/ext_libs/ndsplib-hifi5.patch",
    hook="ndsplib",
    ndsp_core_name="HiFi5",
  ),
  "embarc_mli_1_1": DependencySpec(
    name="embarc_mli_1_1",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/archive/refs/tags/Release_1.1.zip",
      ),
      sha256="743a1e86304799824d233565955d89d3bf236df91ece7bd243ce198ddcac997f",
    ),
    hook="arc_source",
  ),
  "embarc_mli_2_0": DependencySpec(
    name="embarc_mli_2_0",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/archive/refs/tags/Release_2.0.zip",
      ),
      sha256="846b6c0c764d120923e7536be22ff2ac4a2054ce67be0d76bccea83ee3bf5c52",
    ),
    hook="arc_source",
  ),
  "arc_mli_package": DependencySpec(
    name="arc_mli_package",
    archives=ArchiveSpec(
      urls=(
        "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/releases/download/Release_1.1/embARC_MLI_package.zip",
      ),
      sha256="550ecee6e9d031642bce42f0821b48f4c87454cbe1f1706acecc912ba30ef7a6",
    ),
  ),
  "gcc_embedded": DependencySpec(
    name="gcc_embedded",
    archives={
      ("linux", "x86_64"): ArchiveSpec(
        urls=(
          f"{_ARM_GNU_PRIMARY}/arm-gnu-toolchain-{_ARM_GNU_REL}-x86_64-arm-none-eabi.tar.xz",
          f"{_ARM_GNU_MIRROR}/arm-gnu-toolchain-{_ARM_GNU_REL}-x86_64-arm-none-eabi.tar.xz",
        ),
        sha256="8f6903f8ceb084d9227b9ef991490413014d991874a1e34074443c2a72b14dbd",
      ),
      ("linux", "aarch64"): ArchiveSpec(
        urls=(
          f"{_ARM_GNU_PRIMARY}/arm-gnu-toolchain-{_ARM_GNU_REL}-aarch64-arm-none-eabi.tar.xz",
          f"{_ARM_GNU_MIRROR}/arm-gnu-toolchain-{_ARM_GNU_REL}-aarch64-arm-none-eabi.tar.xz",
        ),
        sha256="2d465847eb1d05f876270494f51034de9ace9abe87a4222d079f3360240184d3",
      ),
      ("osx", "arm64"): ArchiveSpec(
        urls=(
          f"{_ARM_GNU_PRIMARY}/arm-gnu-toolchain-{_ARM_GNU_REL}-darwin-arm64-arm-none-eabi.tar.xz",
          f"{_ARM_GNU_MIRROR}/arm-gnu-toolchain-{_ARM_GNU_REL}-darwin-arm64-arm-none-eabi.tar.xz",
        ),
        sha256="30f4d08b219190a37cded6aa796f4549504902c53cfc3c7e044a8490b6eba1f7",
      ),
      ("windows", "*"): ArchiveSpec(
        urls=(
          f"{_ARM_GNU_PRIMARY}/arm-gnu-toolchain-{_ARM_GNU_REL}-mingw-w64-i686-arm-none-eabi.zip",
          f"{_ARM_GNU_MIRROR}/arm-gnu-toolchain-{_ARM_GNU_REL}-mingw-w64-i686-arm-none-eabi.zip",
        ),
        sha256="836ebe51fd71b6542dd7884c8fb2011192464b16c28e4b38fddc9350daba5ee8",
      ),
    },
  ),
  "corstone300": DependencySpec(
    name="corstone300",
    archives={
      ("linux", "x86_64"): ArchiveSpec(
        urls=(f"{_CORSTONE_BASE}/FVP_Corstone_SSE-300_11.24_13_Linux64.tgz",),
        sha256="6ea4096ecf8a8c06d6e76e21cae494f0c7139374cb33f6bc3964d189b84539a9",
      ),
      ("linux", "aarch64"): ArchiveSpec(
        urls=(
          f"{_CORSTONE_BASE}/FVP_Corstone_SSE-300_11.24_13_Linux64_armv8l.tgz",
        ),
        sha256="9b43da6a688220c707cd1801baf9cf4f5fb37d6dc77587b9071347411a64fd56",
      ),
    },
    hook="corstone300",
  ),
  "riscv_toolchain": DependencySpec(
    name="riscv_toolchain",
    archives={
      ("linux", "*"): ArchiveSpec(
        urls=(
          f"{_TF_GCS_MIRROR}/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14.tar.gz",
        ),
        sha256="a4965d855d4e06eecc17b0305e0cb60c4fc7c4fc4c89a3514e28eccfc5b06d07",
      ),
      ("osx", "*"): ArchiveSpec(
        urls=(
          f"{_TF_GCS_MIRROR}/static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-8.1.0-2019.01.0-x86_64-apple-darwin.tar.gz",
        ),
        sha256="af3fc151c8fedc684581db5bda28bdb267b4639dd9897cf00644853174b452d4",
      ),
    },
  ),
}

DEFAULT_DEPENDENCIES = [
  "flatbuffers",
  "kissfft",
]


def normalize_host_platform(host_os="", host_arch=""):
  raw_os = (host_os or "").strip().lower()
  if not raw_os:
    sp = sys.platform.lower()
    if sp.startswith("linux"):
      raw_os = "linux"
    elif sp.startswith("darwin"):
      raw_os = "osx"
    elif sp in ("win32", "cygwin", "msys") or os.name == "nt":
      raw_os = "windows"
    else:
      raw_os = sp
  elif raw_os in ("darwin", "macos"):
    raw_os = "osx"
  elif raw_os in ("win32", "nt") or raw_os.startswith(
    ("mingw", "msys", "cygwin")
  ):
    raw_os = "windows"

  raw_arch = (host_arch or "").strip().lower()
  if not raw_arch:
    raw_arch = platform.machine().strip().lower()
  if raw_arch in ("amd64", "x64"):
    raw_arch = "x86_64"
  elif raw_arch in ("i386", "i486", "i586", "i686", "x86"):
    raw_arch = "x86_32"
  elif raw_arch in ("aarch64", "arm64"):
    raw_arch = "arm64" if raw_os == "osx" else "aarch64"

  return raw_os, raw_arch


def validate_dest_name(dest_name):
  if (
    not dest_name
    or dest_name in (".", "..")
    or dest_name.startswith(".")
    or "/" in dest_name
    or "\\" in dest_name
    or Path(dest_name).is_absolute()
    or Path(dest_name).parts != (dest_name,)
  ):
    raise ValueError(f"Invalid --dest_name: {dest_name!r}")
  return dest_name


def compute_file_sha256(filepath):
  hasher = hashlib.sha256()
  with open(str(filepath), "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def compute_tree_sha256(directory):
  root = Path(directory).resolve()
  hasher = hashlib.sha256()
  for dirpath, dirnames, filenames in os.walk(str(root)):
    dirnames[:] = sorted(d for d in dirnames if d != ".git")
    for fname in sorted(filenames):
      fpath = Path(dirpath) / fname
      rel = fpath.relative_to(root).as_posix()
      hasher.update(rel.encode("utf-8") + b"\0")
      if fpath.is_symlink():
        hasher.update(b"L" + os.readlink(str(fpath)).encode("utf-8") + b"\0")
      else:
        hasher.update(b"F" + compute_file_sha256(fpath).encode("ascii") + b"\0")
  return hasher.hexdigest()


def _get_ethos_u_local_src():
  env_src = os.environ.get("ETHOS_U_CORE_PLATFORM_SRC", "").strip()
  if env_src and Path(env_src).is_dir():
    return Path(env_src).resolve()
  return None


def compute_stamp_metadata(dep, tensorflow_root, host_os, host_arch):
  local_src = None
  archive_sha256 = None
  sub_archives = []

  if dep.hook == "ethos_u_core_platform":
    local_dir = _get_ethos_u_local_src()
    if local_dir is not None:
      local_src = {
        "path": str(local_dir),
        "tree_sha256": compute_tree_sha256(local_dir),
      }

  if local_src is None:
    archive = dep.resolve_archive(host_os, host_arch)
    archive_sha256 = archive.sha256
    sub_archives = [
      {"subdir": sub.subdir, "sha256": sub.sha256}
      for sub in archive.sub_archives
    ]

  patch_sha256 = None
  if dep.patch:
    patch_path = tensorflow_root / dep.patch
    if patch_path.exists():
      patch_sha256 = compute_file_sha256(patch_path)

  return {
    "schema": STAMP_SCHEMA_VERSION,
    "name": dep.name,
    "sha256": archive_sha256,
    "sub_archives": sub_archives,
    "patch_sha256": patch_sha256,
    "local_src": local_src,
    "hook_version": dep.hook_version,
  }


def is_already_downloaded(dep, dest_dir, tensorflow_root, host_os, host_arch):
  stamp_path = dest_dir / STAMP_FILENAME
  if not dest_dir.is_dir() or not stamp_path.is_file():
    return False
  if dep.hook == "ethos_u_core_platform":
    parsed_ld = dest_dir / "targets" / "corstone-300" / "platform_parsed.ld"
    if not parsed_ld.is_file():
      return False
  try:
    with open(str(stamp_path), "r", encoding="utf-8") as f:
      existing = json.load(f)
    expected = compute_stamp_metadata(dep, tensorflow_root, host_os, host_arch)
    return existing == expected
  except Exception:
    return False


def _lock_file_path(downloads_dir, dest_name):
  dir_hash = hashlib.sha256(
    str(downloads_dir.resolve()).encode("utf-8")
  ).hexdigest()[:16]
  safe_dest = re.sub(r"[^A-Za-z0-9_.-]", "_", dest_name)
  return Path(tempfile.gettempdir()) / f"tflm_dl_{dir_hash}_{safe_dest}.lock"


def _get_thread_lock(lock_path_str):
  with _THREAD_LOCKS_GUARD:
    lock = _THREAD_LOCKS.get(lock_path_str)
    if lock is None:
      lock = threading.RLock()
      _THREAD_LOCKS[lock_path_str] = lock
    return lock


@contextlib.contextmanager
def destination_lock(downloads_dir, dest_name, shared=False):
  lock_path = _lock_file_path(downloads_dir, dest_name)
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  tlock = _get_thread_lock(str(lock_path))
  with tlock:
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    try:
      if fcntl is not None:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(fd, mode)
      elif msvcrt is not None:
        while True:
          try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            break
          except OSError:
            time.sleep(0.05)
      yield
    finally:
      try:
        if fcntl is not None:
          fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
          os.lseek(fd, 0, os.SEEK_SET)
          msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
      except OSError:
        pass
      os.close(fd)


def sweep_stale_siblings(downloads_dir, dest_name):
  if not downloads_dir.is_dir():
    return
  prefixes = (
    f".tmp_{dest_name}_",
    f".dl_{dest_name}_",
    f".old_{dest_name}_",
    f".unpack_{dest_name}_",
    f".installer_{dest_name}_",
  )
  for entry in downloads_dir.iterdir():
    if entry.name.startswith(prefixes):
      if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(str(entry), ignore_errors=True)
      else:
        try:
          entry.unlink()
        except OSError:
          pass


def delete_build_files(directory):
  for root, _, files in os.walk(str(directory)):
    for file in files:
      if file in ("BUILD", "BUILD.bazel"):
        try:
          os.remove(os.path.join(root, file))
        except OSError:
          pass


def download_urls_with_retry(urls, output_path, expected_sha256, max_retries=5):
  last_err = None
  permanent_failures = 0

  for attempt in range(1, max_retries + 1):
    permanent_failures = 0
    for url in urls:
      try:
        req = urllib.request.Request(url)
        hasher = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=60) as response:
          total_header = response.headers.get("Content-Length")
          total_bytes = (
            int(total_header) if total_header and total_header.isdigit() else 0
          )
          report_progress = total_bytes > 10 * 1024 * 1024
          next_milestone = 25
          downloaded = 0

          with open(str(output_path), "wb") as out_file:
            while True:
              chunk = response.read(65536)
              if not chunk:
                break
              out_file.write(chunk)
              hasher.update(chunk)
              downloaded += len(chunk)
              if report_progress and total_bytes > 0:
                pct = (downloaded * 100) // total_bytes
                if pct >= next_milestone:
                  sys.stderr.write(
                    f"  Downloaded {pct}% ({downloaded // (1024 * 1024)} MiB)...\n"
                  )
                  while next_milestone <= pct:
                    next_milestone += 25

        actual_sha256 = hasher.hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
          raise ValueError(
            f"SHA256 mismatch for {url}: expected {expected_sha256}, "
            f"got {actual_sha256}"
          )
        return
      except ValueError:
        raise
      except urllib.error.HTTPError as e:
        last_err = e
        if 400 <= e.code < 500 and e.code not in (408, 429):
          permanent_failures += 1
          continue
      except Exception as e:
        last_err = e

    if permanent_failures == len(urls):
      break
    if attempt < max_retries:
      time.sleep(min(2**attempt, 8))

  raise RuntimeError(
    f"Failed to download {urls[0]} after {max_retries} attempts: {last_err}"
  )


def _ensure_within_directory(base_dir, target_path):
  resolved_base = base_dir.resolve()
  resolved_target = target_path.resolve()
  if (
    resolved_target != resolved_base
    and resolved_base not in resolved_target.parents
  ):
    raise ValueError(f"Archive member escapes target directory: {target_path}")


def _extract_zip_safely(archive_path, temp_extract):
  resolved_root = temp_extract.resolve()
  with zipfile.ZipFile(str(archive_path), "r") as zf:
    for info in zf.infolist():
      target = temp_extract / info.filename
      _ensure_within_directory(resolved_root, target)
      extracted_str = zf.extract(info, str(temp_extract))
      if os.name != "nt":
        extracted = Path(extracted_str)
        unix_mode = (
          ((info.external_attr >> 16) & 0o777) if info.create_system == 3 else 0
        )
        if info.is_dir() or extracted.is_dir():
          mode = (
            ((unix_mode | 0o700) & ~UMASK) if unix_mode else (0o777 & ~UMASK)
          )
        else:
          mode = (
            ((unix_mode | 0o200) & ~UMASK) if unix_mode else (0o666 & ~UMASK)
          )
        try:
          os.chmod(str(extracted), mode)
        except OSError:
          pass


def _materialize_link_fallback(member, temp_extract):
  dest_path = temp_extract / member.name
  _ensure_within_directory(temp_extract, dest_path)
  if member.issym():
    link_target = (dest_path.parent / member.linkname).resolve()
  else:
    link_target = (temp_extract / member.linkname).resolve()
  _ensure_within_directory(temp_extract, link_target)

  if dest_path.exists() or dest_path.is_symlink():
    if dest_path.is_dir() and not dest_path.is_symlink():
      shutil.rmtree(str(dest_path))
    else:
      dest_path.unlink()

  if link_target.is_dir():
    shutil.copytree(str(link_target), str(dest_path), dirs_exist_ok=True)
  elif link_target.is_file():
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(link_target), str(dest_path))


def _extract_tar_safely(archive_path, temp_extract):
  has_data_filter = hasattr(tarfile, "data_filter")
  resolved_root = temp_extract.resolve()

  with tarfile.open(str(archive_path), "r:*") as tf:
    regular_members = []
    link_members = []

    for member in tf.getmembers():
      if has_data_filter:
        filtered = tarfile.data_filter(member, str(temp_extract))
        if filtered is None:
          continue
        member = filtered
      else:
        if member.isdev():
          raise ValueError(f"Refusing device node in tar: {member.name}")
        target = temp_extract / member.name
        _ensure_within_directory(resolved_root, target)
        if member.issym():
          _ensure_within_directory(
            resolved_root, target.parent / member.linkname
          )
        elif member.islnk():
          _ensure_within_directory(
            resolved_root, temp_extract / member.linkname
          )

      if member.issym() or member.islnk():
        link_members.append(member)
      else:
        regular_members.append(member)

    for member in regular_members:
      if has_data_filter:
        tf.extract(member, str(temp_extract), filter="fully_trusted")
      else:
        tf.extract(member, str(temp_extract))
      if os.name != "nt":
        extracted = temp_extract / member.name
        if extracted.exists() and not extracted.is_symlink():
          try:
            cur_mode = extracted.stat().st_mode & 0o777
            min_owner = 0o700 if extracted.is_dir() else 0o200
            os.chmod(str(extracted), (cur_mode | min_owner) & ~UMASK)
          except OSError:
            pass

    for member in link_members:
      if _FORCE_SYMLINK_COPY_FALLBACK:
        _materialize_link_fallback(member, temp_extract)
        continue
      try:
        if has_data_filter:
          tf.extract(member, str(temp_extract), filter="fully_trusted")
        else:
          tf.extract(member, str(temp_extract))
      except OSError:
        if os.name == "nt":
          _materialize_link_fallback(member, temp_extract)
        else:
          raise


def extract_archive(archive_path, extract_dir, strip_single_root=True):
  extract_dir.parent.mkdir(parents=True, exist_ok=True)
  temp_extract = Path(
    tempfile.mkdtemp(
      prefix=f".unpack_{extract_dir.name}_", dir=str(extract_dir.parent)
    )
  )
  try:
    if zipfile.is_zipfile(str(archive_path)):
      _extract_zip_safely(archive_path, temp_extract)
    elif tarfile.is_tarfile(str(archive_path)):
      _extract_tar_safely(archive_path, temp_extract)
    else:
      raise ValueError(f"Unsupported archive format: {archive_path}")

    entries = [p for p in temp_extract.iterdir() if p.name != "__MACOSX"]
    if strip_single_root and len(entries) == 1 and entries[0].is_dir():
      src_dir = entries[0]
    else:
      src_dir = temp_extract

    extract_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
      shutil.move(str(item), str(extract_dir / item.name))
    if os.name != "nt":
      os.chmod(str(extract_dir), 0o777 & ~UMASK)
  finally:
    shutil.rmtree(str(temp_extract), ignore_errors=True)


def normalize_text_newlines_in_tree(directory):
  for root, _, files in os.walk(str(directory)):
    for fname in files:
      fpath = Path(root) / fname
      if fpath.is_symlink():
        continue
      try:
        cur_mode = fpath.stat().st_mode & 0o777
        if os.name != "nt" and not (cur_mode & 0o200):
          os.chmod(str(fpath), (cur_mode | 0o200) & ~UMASK)
        data = fpath.read_bytes()
        if b"\x00" not in data and b"\r" in data:
          normalized = re.sub(b"\r+\n", b"\n", data)
          if normalized != data:
            fpath.write_bytes(normalized)
      except OSError as e:
        raise RuntimeError(
          f"Failed normalizing newlines in {fpath}: {e}"
        ) from e


def _sweep_patch_artifacts(directory):
  for root, _, files in os.walk(str(directory)):
    for fname in files:
      if fname.endswith((".rej", ".orig")):
        try:
          os.remove(os.path.join(root, fname))
        except OSError:
          pass


def apply_patch(patch_path, target_dir, normalize_crlf=False):
  if not patch_path.is_file():
    raise FileNotFoundError(f"Patch file not found: {patch_path}")

  patch_bytes = patch_path.read_bytes()
  if normalize_crlf:
    patch_bytes = re.sub(b"\r+\n", b"\n", patch_bytes)

  patch_err = ""
  try:
    res = subprocess.run(
      ["patch", "-p1", "-f", "-d", str(target_dir)],
      input=patch_bytes,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
    )
    if res.returncode == 0:
      return
    _sweep_patch_artifacts(target_dir)
    out_msg = res.stdout.decode("utf-8", errors="replace")
    err_msg = res.stderr.decode("utf-8", errors="replace")
    patch_err = f"{out_msg}\n{err_msg}".strip()
  except FileNotFoundError:
    patch_err = "'patch' command not found"

  git_cmd = [
    "git",
    "apply",
    "--ignore-space-change",
    "--ignore-whitespace",
    "-",
  ]
  try:
    res2 = subprocess.run(
      git_cmd,
      input=patch_bytes,
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
  except FileNotFoundError as exc:
    raise RuntimeError(
      f"Failed to apply patch {patch_path}: neither 'patch' nor 'git' commands were found."
    ) from exc


def _download_archive_to_dir(
  archive_spec, staging_dir, downloads_dir, temp_tag, strip_single_root=True
):
  temp_archive = (
    downloads_dir / f".dl_{temp_tag}_{os.getpid()}_{threading.get_ident()}"
  )
  try:
    download_urls_with_retry(
      archive_spec.urls, temp_archive, archive_spec.sha256
    )
    extract_archive(
      temp_archive, staging_dir, strip_single_root=strip_single_root
    )
  finally:
    if temp_archive.exists():
      try:
        temp_archive.unlink()
      except OSError:
        pass


def _prepare_ndsplib(dep, archive_spec, staging_dir, downloads_dir):
  outer_dir = Path(
    tempfile.mkdtemp(
      prefix=f".unpack_{dep.name}_outer_", dir=str(downloads_dir)
    )
  )
  try:
    _download_archive_to_dir(archive_spec, outer_dir, downloads_dir, dep.name)
    core_dir = outer_dir / f"NDSP_{dep.ndsp_core_name}"
    inner_zips = sorted(core_dir.glob(f"NDSP_{dep.ndsp_core_name}*.zip"))
    if not inner_zips:
      raise FileNotFoundError(f"Inner NDSP zip not found in {core_dir}")
    inner_unpack = outer_dir / "inner_unpacked"
    extract_archive(inner_zips[0], inner_unpack, strip_single_root=False)
    subdirs = sorted(
      p for p in inner_unpack.iterdir() if p.is_dir() and p.name != "__MACOSX"
    )
    src_root = subdirs[0] if len(subdirs) == 1 else inner_unpack
    for item in src_root.iterdir():
      shutil.move(str(item), str(staging_dir / item.name))
  finally:
    shutil.rmtree(str(outer_dir), ignore_errors=True)

  normalize_text_newlines_in_tree(staging_dir)


def _copy_local_source_tree(src_dir, staging_dir):
  shutil.copytree(
    str(src_dir),
    str(staging_dir),
    dirs_exist_ok=True,
    ignore=shutil.ignore_patterns(".git"),
  )
  if os.name != "nt":
    for root, dirs, files in os.walk(str(staging_dir)):
      for dname in dirs:
        dpath = os.path.join(root, dname)
        if not os.path.islink(dpath):
          cur = os.stat(dpath).st_mode & 0o777
          os.chmod(dpath, (cur | 0o700) & ~UMASK)
      for fname in files:
        fpath = os.path.join(root, fname)
        if not os.path.islink(fpath):
          cur = os.stat(fpath).st_mode & 0o777
          os.chmod(fpath, (cur | 0o200) & ~UMASK)


def _preprocess_ethos_linker_script(
  staging_dir, downloads_dir, tensorflow_root, host_os, host_arch, cc_override
):
  linker_dir = staging_dir / "targets" / "corstone-300"
  input_ld = linker_dir / "platform.ld"
  output_ld = linker_dir / "platform_parsed.ld"

  compiler = None
  needs_gcc_lock = False
  gcc_dir = (downloads_dir / "gcc_embedded").resolve()

  if cc_override:
    candidate = cc_override.strip()
    if os.path.isfile(candidate) or shutil.which(candidate):
      compiler = candidate
      resolved_cc = Path(shutil.which(candidate) or candidate).resolve()
      if resolved_cc == gcc_dir or gcc_dir in resolved_cc.parents:
        needs_gcc_lock = True

  if compiler is None:
    found = shutil.which("arm-none-eabi-gcc")
    if found:
      compiler = found
      resolved_cc = Path(found).resolve()
      if resolved_cc == gcc_dir or gcc_dir in resolved_cc.parents:
        needs_gcc_lock = True

  if compiler is None:
    download_and_prepare(
      DEPENDENCIES["gcc_embedded"],
      downloads_dir,
      tensorflow_root,
      host_os=host_os,
      host_arch=host_arch,
    )
    exe_name = (
      "arm-none-eabi-gcc.exe" if host_os == "windows" else "arm-none-eabi-gcc"
    )
    compiler = str(downloads_dir / "gcc_embedded" / "bin" / exe_name)
    needs_gcc_lock = True

  cmd = [
    compiler,
    "-E",
    "-x",
    "c",
    "-P",
    "-o",
    str(output_ld),
    str(input_ld),
  ]
  lock_ctx = (
    destination_lock(downloads_dir, "gcc_embedded", shared=True)
    if needs_gcc_lock
    else contextlib.nullcontext()
  )
  with lock_ctx:
    res = subprocess.run(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
    )
  if res.returncode != 0:
    raise RuntimeError(
      "Failed to preprocess {} with {}:\n{}".format(
        input_ld,
        compiler,
        res.stderr.decode("utf-8", errors="replace"),
      )
    )


def _prepare_corstone300(archive_spec, staging_dir, downloads_dir, dest_name):
  installer_dir = Path(
    tempfile.mkdtemp(prefix=f".installer_{dest_name}_", dir=str(downloads_dir))
  )
  try:
    _download_archive_to_dir(
      archive_spec,
      installer_dir,
      downloads_dir,
      dest_name,
      strip_single_root=False,
    )
    scripts = sorted(installer_dir.rglob("FVP_Corstone_SSE-300.sh"))
    if not scripts:
      raise FileNotFoundError(
        f"FVP_Corstone_SSE-300.sh not found in {installer_dir}"
      )
    installer_script = scripts[0]
    if os.name != "nt":
      os.chmod(str(installer_script), 0o755)
    res = subprocess.run(
      [
        str(installer_script),
        "--i-agree-to-the-contained-eula",
        "--no-interactive",
        "-d",
        str(staging_dir),
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=False,
    )
    if res.returncode != 0:
      out_msg = res.stdout.decode("utf-8", errors="replace")
      err_msg = res.stderr.decode("utf-8", errors="replace")
      raise RuntimeError(
        f"Corstone-300 FVP installer failed (rc={res.returncode}):\n"
        f"{out_msg}\n{err_msg}"
      )
  finally:
    shutil.rmtree(str(installer_dir), ignore_errors=True)


def _has_unstamped_arc_build_artifact(dep, dest_dir):
  if dep.hook != "arc_source":
    return False
  if not dest_dir.is_dir() or (dest_dir / STAMP_FILENAME).exists():
    return False
  bin_dir = dest_dir / "bin"
  return bin_dir.is_dir() and any(bin_dir.rglob("libmli.a"))


def download_and_prepare(
  dep,
  downloads_dir,
  tensorflow_root,
  dest_name=None,
  host_os="linux",
  host_arch="x86_64",
  cc_override=None,
):
  target_name = validate_dest_name(dest_name) if dest_name else dep.name
  dest_dir = downloads_dir / target_name

  with destination_lock(downloads_dir, target_name, shared=False):
    sweep_stale_siblings(downloads_dir, target_name)

    if is_already_downloaded(
      dep, dest_dir, tensorflow_root, host_os, host_arch
    ):
      return

    if _has_unstamped_arc_build_artifact(dep, dest_dir):
      sys.stderr.write(
        f"Preserving existing unstamped {dest_dir} containing compiled libmli.a.\n"
      )
      return

    if dep.hook == "ethos_u_core_platform" and host_os != "linux":
      raise RuntimeError(
        f"OS type '{host_os}' is not supported for {dep.name}."
      )

    sys.stderr.write(f"Downloading and setting up {target_name}...\n")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
      tempfile.mkdtemp(prefix=f".tmp_{target_name}_", dir=str(downloads_dir))
    )

    try:
      local_ethos_src = (
        _get_ethos_u_local_src()
        if dep.hook == "ethos_u_core_platform"
        else None
      )

      if local_ethos_src is not None:
        sys.stderr.write(f"Using pre-baked {dep.name} from {local_ethos_src}\n")
        _copy_local_source_tree(local_ethos_src, staging_dir)
      else:
        archive_spec = dep.resolve_archive(host_os, host_arch)
        if dep.hook == "ndsplib":
          _prepare_ndsplib(dep, archive_spec, staging_dir, downloads_dir)
        elif dep.hook == "corstone300":
          _prepare_corstone300(
            archive_spec, staging_dir, downloads_dir, target_name
          )
        else:
          _download_archive_to_dir(
            archive_spec, staging_dir, downloads_dir, target_name
          )
          for sub in archive_spec.sub_archives:
            sub_dest = staging_dir / sub.subdir
            _download_archive_to_dir(
              sub,
              sub_dest,
              downloads_dir,
              f"{target_name}_{sub.subdir}",
            )

      delete_build_files(staging_dir)

      if dep.patch:
        patch_path = tensorflow_root / dep.patch
        apply_patch(
          patch_path,
          staging_dir,
          normalize_crlf=(dep.hook == "ndsplib"),
        )

      if dep.hook == "ndsplib":
        renaming_h = (
          staging_dir / "library" / "include_private" / "__renaming__.h"
        )
        if not renaming_h.is_file():
          raise FileNotFoundError(f"Expected header missing: {renaming_h}")
        raw_bytes = renaming_h.read_bytes()
        renamed = b"\n".join(
          line.replace(b"NatureDSP_", b"NatureDSP_TFLM_", 1)
          for line in raw_bytes.split(b"\n")
        )
        renaming_h.write_bytes(renamed)
      elif dep.hook == "ethos_u_core_platform":
        _preprocess_ethos_linker_script(
          staging_dir,
          downloads_dir,
          tensorflow_root,
          host_os,
          host_arch,
          cc_override,
        )

      metadata = compute_stamp_metadata(
        dep, tensorflow_root, host_os, host_arch
      )
      stamp_file = staging_dir / STAMP_FILENAME
      stamp_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
      )
      if os.name != "nt":
        os.chmod(str(stamp_file), 0o666 & ~UMASK)
        os.chmod(str(staging_dir), 0o777 & ~UMASK)

      old_dir = None
      if dest_dir.exists() or dest_dir.is_symlink():
        old_dir = downloads_dir / f".old_{target_name}_{os.getpid()}"
        if old_dir.exists():
          shutil.rmtree(str(old_dir), ignore_errors=True)
        dest_dir.rename(old_dir)
      try:
        staging_dir.rename(dest_dir)
      except BaseException:
        if old_dir is not None and old_dir.exists() and not dest_dir.exists():
          old_dir.rename(dest_dir)
        raise
      finally:
        if old_dir is not None and old_dir.exists():
          shutil.rmtree(str(old_dir), ignore_errors=True)

      sys.stderr.write(f"Successfully installed {target_name}.\n")
    except BaseException:
      shutil.rmtree(str(staging_dir), ignore_errors=True)
      raise


def main(argv=None):
  parser = argparse.ArgumentParser(
    description="Download third-party dependencies for TFLM."
  )
  parser.add_argument(
    "--downloads_dir",
    default="tensorflow/lite/micro/tools/make/downloads",
    help="Directory to unpack dependencies into.",
  )
  parser.add_argument(
    "--tensorflow_root",
    default="",
    help="Root directory of TensorFlow / TFLM tree.",
  )
  parser.add_argument(
    "--dest_name",
    default="",
    help="Override destination directory name (single dependency only).",
  )
  parser.add_argument(
    "--host_os",
    default="",
    help="Optional host OS override (linux, osx, windows).",
  )
  parser.add_argument(
    "--host_arch",
    default="",
    help="Optional host architecture override (x86_64, aarch64, arm64).",
  )
  parser.add_argument(
    "--cc",
    default="",
    help="Optional C compiler path for preprocessing linker scripts.",
  )
  parser.add_argument(
    "--print_metadata",
    action="store_true",
    help="Print LIBRARY_URL and LIBRARY_SHA256 without downloading.",
  )
  parser.add_argument(
    "dependencies",
    nargs="*",
    help="Specific dependencies to download (defaults to core host deps).",
  )
  args = parser.parse_args(argv)

  host_os, host_arch = normalize_host_platform(args.host_os, args.host_arch)
  target_deps = args.dependencies if args.dependencies else DEFAULT_DEPENDENCIES

  for name in target_deps:
    if name not in DEPENDENCIES:
      sys.stderr.write(
        "Unknown dependency: {}. Available: {}\n".format(
          name, sorted(DEPENDENCIES.keys())
        )
      )
      return 1

  if args.dest_name:
    if len(target_deps) != 1:
      sys.stderr.write(
        "--dest_name can only be used when downloading a single dependency.\n"
      )
      return 1
    try:
      validate_dest_name(args.dest_name)
    except ValueError as e:
      sys.stderr.write(f"{e}\n")
      return 1

  if args.print_metadata:
    for name in target_deps:
      try:
        archive = DEPENDENCIES[name].resolve_archive(host_os, host_arch)
      except Exception as e:
        sys.stderr.write(f"Error resolving metadata for {name}: {e}\n")
        return 1
      print(f"LIBRARY_URL={archive.urls[0]}")
      print(f"LIBRARY_SHA256={archive.sha256}")
    return 0

  downloads_dir = Path(args.downloads_dir).resolve()
  tensorflow_root = Path(args.tensorflow_root).resolve()

  seen = set()
  ordered_deps = []
  for name in target_deps:
    if name not in seen:
      seen.add(name)
      ordered_deps.append(name)

  # Ensure gcc_embedded precedes ethos_u_core_platform when requested together.
  if (
    "gcc_embedded" in seen
    and "ethos_u_core_platform" in seen
    and ordered_deps.index("gcc_embedded")
    > ordered_deps.index("ethos_u_core_platform")
  ):
    ordered_deps.remove("gcc_embedded")
    ordered_deps.insert(
      ordered_deps.index("ethos_u_core_platform"), "gcc_embedded"
    )

  for name in ordered_deps:
    try:
      download_and_prepare(
        DEPENDENCIES[name],
        downloads_dir,
        tensorflow_root,
        dest_name=args.dest_name or None,
        host_os=host_os,
        host_arch=host_arch,
        cc_override=args.cc or None,
      )
    except Exception as e:
      sys.stderr.write(f"Error downloading {name}: {e}\n")
      return 1

  print("SUCCESS")
  return 0


if __name__ == "__main__":
  sys.exit(main())
