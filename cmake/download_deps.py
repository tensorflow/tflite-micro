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
"""Standalone helper to download FlatBuffers & KissFFT for CMake testing/CI.

CMake never invokes this script or accesses the network; downstream builds
provide dependencies via CMake targets or `-DTFLM_*_DIR` / `-DTFLM_DEPS_DIR`.
"""

import argparse
import ast
import dataclasses
import hashlib
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

_DEFAULT_DEPS = ("flatbuffers", "kissfft")


@dataclasses.dataclass(frozen=True)
class DepSpec:
  """Metadata for an external source archive pinned in MODULE.bazel."""

  name: str
  url: str
  sha256: str
  strip_prefix: str = ""


def _load_dep_specs(repo_root: pathlib.Path) -> dict[str, DepSpec]:
  """Extracts FlatBuffers and KissFFT http_archive specs from MODULE.bazel."""
  mod = repo_root / "MODULE.bazel"
  specs: dict[str, DepSpec] = {}
  for node in ast.parse(mod.read_text(encoding="utf-8"), str(mod)).body:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
      continue
    call = node.value
    if getattr(call.func, "id", None) != "http_archive":
      continue
    kw: dict[str, object] = {}
    for k in call.keywords:
      if k.arg:
        try:
          kw[k.arg] = ast.literal_eval(k.value)
        except ValueError:
          pass
    name = kw.get("name")
    urls = kw.get("urls")
    sha = kw.get("sha256")
    if (
      isinstance(name, str)
      and name in _DEFAULT_DEPS
      and isinstance(urls, list)
      and urls
      and isinstance(urls[0], str)
      and isinstance(sha, str)
      and sha
    ):
      if kw.get("patches"):
        raise RuntimeError(
          f"Unsupported patches attribute on {name} in MODULE.bazel"
        )
      strip_prefix = kw.get("strip_prefix")
      specs[name] = DepSpec(
        name=name,
        url=urls[0],
        sha256=sha,
        strip_prefix=strip_prefix if isinstance(strip_prefix, str) else "",
      )
  return specs


def _sha256_file(path: pathlib.Path) -> str:
  """Computes the hex SHA-256 digest of a file."""
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _extract_archive(archive: pathlib.Path, dest: pathlib.Path) -> None:
  """Extracts a .zip or .tar.* archive into dest."""
  lower = archive.name.lower()
  if lower.endswith(".zip"):
    with zipfile.ZipFile(archive) as zf:
      zf.extractall(dest)
  elif lower.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
    with tarfile.open(archive) as tf:
      try:
        tf.extractall(dest, filter="data")
      except TypeError:
        tf.extractall(dest)
  else:
    raise RuntimeError(f"Unsupported archive format: {archive}")


def _install_dep(spec: DepSpec, dest_root: pathlib.Path, force: bool) -> None:
  """Downloads, verifies, and unpacks a single dependency if needed."""
  target_dir = dest_root / spec.name
  stamp_file = target_dir / ".tflm_dep_stamp"
  expected_stamp = f"sha256:{spec.sha256}\n"
  if (
    not force
    and stamp_file.exists()
    and stamp_file.read_text("utf-8") == expected_stamp
  ):
    print(f"[up-to-date] {spec.name} -> {target_dir}")
    return

  print(f"[fetching]   {spec.name} <- {spec.url}")
  with tempfile.TemporaryDirectory(prefix=f"tflm_{spec.name}_") as tmp:
    tmp_path = pathlib.Path(tmp)
    archive_path = tmp_path / spec.url.rsplit("/", 1)[-1]
    with (
      urllib.request.urlopen(spec.url, timeout=60) as resp,
      archive_path.open("wb") as out,
    ):
      shutil.copyfileobj(resp, out)
    actual_digest = _sha256_file(archive_path)
    if actual_digest.lower() != spec.sha256.lower():
      raise RuntimeError(
        f"SHA-256 mismatch for {spec.name}: "
        f"expected {spec.sha256}, got {actual_digest}"
      )
    unpack_dir = tmp_path / "unpacked"
    unpack_dir.mkdir()
    _extract_archive(archive_path, unpack_dir)
    src_root = (
      unpack_dir / spec.strip_prefix if spec.strip_prefix else unpack_dir
    )
    if not src_root.exists():
      subdirs = [p for p in unpack_dir.iterdir() if p.is_dir()]
      if len(subdirs) == 1:
        src_root = subdirs[0]
    if target_dir.exists():
      shutil.rmtree(target_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_root), str(target_dir))
    stamp_file.write_text(expected_stamp, encoding="utf-8")
    print(f"[installed]  {spec.name} -> {target_dir}")


def main(argv: list[str] | None = None) -> int:
  repo_root = pathlib.Path(__file__).resolve().parent.parent
  specs = _load_dep_specs(repo_root)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dest_dir", nargs="?", help="Destination root directory.")
  parser.add_argument("--deps", default=",".join(_DEFAULT_DEPS))
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args(argv)

  requested = [d.strip() for d in args.deps.split(",") if d.strip()]
  if unknown := [d for d in requested if d not in specs]:
    parser.error(
      f"Unknown dep(s): {', '.join(unknown)}. "
      f"Available: {', '.join(sorted(specs))}"
    )
  dest_root = (
    pathlib.Path(args.dest_dir) if args.dest_dir else repo_root / "build/deps"
  ).resolve()
  for dep_name in requested:
    _install_dep(specs[dep_name], dest_root, force=args.force)
  print(
    f"\nDependencies ready in: {dest_root}\n"
    f'Configure CMake with:  -DTFLM_DEPS_DIR="{dest_root}"'
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
