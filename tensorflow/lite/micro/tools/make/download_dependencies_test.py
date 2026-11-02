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
"""Hermetic unit and concurrency tests for download_dependencies.py."""

from __future__ import annotations

import ast
import hashlib
import http.server
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import download_dependencies as dl  # noqa: E402


def _sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _build_zip_bytes(
  entries: list[tuple[str, bytes, int, int]],
) -> bytes:
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name, content, create_system, mode in entries:
      info = zipfile.ZipInfo(name)
      info.create_system = create_system
      info.external_attr = (mode & 0xFFFF) << 16
      zf.writestr(info, content)
  return buf.getvalue()


def _build_tar_gz_bytes(
  files: dict[str, bytes] | None = None,
  symlinks: dict[str, str] | None = None,
  modes: dict[str, int] | None = None,
) -> bytes:
  buf = io.BytesIO()
  files = files or {}
  symlinks = symlinks or {}
  modes = modes or {}
  with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for name, content in files.items():
      info = tarfile.TarInfo(name=name)
      info.size = len(content)
      info.mode = modes.get(name, 0o644)
      tf.addfile(info, io.BytesIO(content))
    for link_name, target in symlinks.items():
      info = tarfile.TarInfo(name=link_name)
      info.type = tarfile.SYMTYPE
      info.linkname = target
      info.mode = 0o777
      tf.addfile(info)
  return buf.getvalue()


class _LocalArchiveServer:
  def __init__(self, routes: dict[str, bytes]):
    self.routes = dict(routes)
    self.request_counts: dict[str, int] = {}
    self._lock = threading.Lock()
    outer = self

    class Handler(http.server.BaseHTTPRequestHandler):
      def do_GET(self):  # noqa: N802
        with outer._lock:
          outer.request_counts[self.path] = (
            outer.request_counts.get(self.path, 0) + 1
          )
        payload = outer.routes.get(self.path)
        if payload is None:
          self.send_response(404)
          self.end_headers()
          return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

      def log_message(self, fmt, *args):
        return

    self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.port = self._server.server_port
    self._thread = threading.Thread(
      target=self._server.serve_forever, daemon=True
    )

  def url(self, path: str) -> str:
    return f"http://127.0.0.1:{self.port}{path}"

  def __enter__(self):
    self._thread.start()
    return self

  def __exit__(self, exc_type, exc, tb):
    self._server.shutdown()
    self._server.server_close()
    self._thread.join(timeout=5)


def _race_worker(args):
  url, sha256, downloads_dir, tf_root = args
  spec = dl.DependencySpec(
    name="race_pkg",
    archives=dl.ArchiveSpec(urls=(url,), sha256=sha256),
  )
  dl.download_and_prepare(
    spec,
    Path(downloads_dir),
    Path(tf_root),
    host_os="linux",
    host_arch="x86_64",
  )
  return True


def _parse_module_bazel_http_archives(module_path: Path) -> dict[str, dict]:
  tree = ast.parse(module_path.read_text(encoding="utf-8"))
  archives = {}
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "http_archive"
    ):
      kwargs = {}
      for kw in node.keywords:
        if kw.arg in ("name", "sha256", "urls"):
          kwargs[kw.arg] = ast.literal_eval(kw.value)
      if "name" in kwargs:
        archives[kwargs["name"]] = kwargs
  return archives


def _parse_repo_bzl_download(bzl_path: Path) -> tuple[list[str], str]:
  tree = ast.parse(bzl_path.read_text(encoding="utf-8"))
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "download_and_extract"
    ):
      kwargs = {
        kw.arg: ast.literal_eval(kw.value)
        for kw in node.keywords
        if kw.arg in ("url", "sha256")
      }
      return kwargs["url"], kwargs["sha256"]
  raise AssertionError(f"No rctx.download_and_extract found in {bzl_path}")


class DownloadDependenciesTest(unittest.TestCase):
  def test_registry_integrity(self):
    hex64 = re.compile(r"^[0-9a-f]{64}$")
    self.assertEqual(dl.DEFAULT_DEPENDENCIES, ["flatbuffers", "kissfft"])

    def check_spec(spec: dl.ArchiveSpec, label: str):
      self.assertTrue(spec.urls, f"Empty urls for {label}")
      for u in spec.urls:
        self.assertTrue(
          u.startswith("https://"), f"Non-HTTPS URL for {label}: {u}"
        )
      self.assertRegex(spec.sha256, hex64, f"Invalid sha256 for {label}")
      for sub in spec.sub_archives:
        for u in sub.urls:
          self.assertTrue(
            u.startswith("https://"), f"Non-HTTPS sub URL for {label}: {u}"
          )
        self.assertRegex(
          sub.sha256, hex64, f"Invalid sub sha256 for {label}:{sub.subdir}"
        )

    for name, dep in dl.DEPENDENCIES.items():
      if isinstance(dep.archives, dl.ArchiveSpec):
        check_spec(dep.archives, name)
      else:
        for plat, spec in dep.archives.items():
          check_spec(spec, f"{name}{plat}")
      if dep.patch:
        self.assertTrue(
          (REPO_ROOT / dep.patch).is_file(),
          f"Missing patch file for {name}: {dep.patch}",
        )

  def test_bazel_url_and_sha256_parity(self):
    module_bazel = REPO_ROOT / "MODULE.bazel"
    bazel_archives = _parse_module_bazel_http_archives(module_bazel)

    parity_map = {
      "flatbuffers": "flatbuffers",
      "kissfft": "kissfft",
      "cmsis_nn": "cmsis_nn",
      "cmsis": "cmsis_core",
      "ethos_u_core_driver": "ethos_u_core_driver",
      "xa_nnlib_hifi4": "nnlib_hifi4",
      "xa_nnlib_hifi5": "nnlib_hifi5",
      "xi_tflmlib_vision_p6": "xi_tflmlib_vision_p6",
    }
    for make_name, bazel_name in parity_map.items():
      self.assertIn(bazel_name, bazel_archives)
      make_spec = dl.DEPENDENCIES[make_name].resolve_archive("linux", "x86_64")
      bazel_spec = bazel_archives[bazel_name]
      self.assertEqual(
        list(make_spec.urls),
        bazel_spec["urls"],
        f"URL mismatch between Make ({make_name}) and MODULE.bazel ({bazel_name})",
      )
      self.assertEqual(
        make_spec.sha256,
        bazel_spec["sha256"],
        f"SHA256 mismatch between Make ({make_name}) and MODULE.bazel ({bazel_name})",
      )

    arm_bzl = REPO_ROOT / "tensorflow/lite/micro/tools/toolchains/arm/repo.bzl"
    arm_urls, arm_sha = _parse_repo_bzl_download(arm_bzl)
    gcc_spec = dl.DEPENDENCIES["gcc_embedded"].resolve_archive(
      "linux", "x86_64"
    )
    self.assertEqual(list(gcc_spec.urls), arm_urls)
    self.assertEqual(gcc_spec.sha256, arm_sha)

    riscv_bzl = (
      REPO_ROOT / "tensorflow/lite/micro/tools/toolchains/riscv/repo.bzl"
    )
    riscv_urls, riscv_sha = _parse_repo_bzl_download(riscv_bzl)
    riscv_spec = dl.DEPENDENCIES["riscv_toolchain"].resolve_archive(
      "linux", "x86_64"
    )
    self.assertEqual(list(riscv_spec.urls), riscv_urls)
    self.assertEqual(riscv_spec.sha256, riscv_sha)

  def test_sha256_verification_permissions_and_stamp_invalidation(self):
    pkg_tar = _build_tar_gz_bytes(
      files={
        "demo-1.0/include/demo.h": b"#define DEMO 1\n",
        "demo-1.0/BUILD": b"# should be deleted\n",
        "demo-1.0/sub/BUILD.bazel": b"# should be deleted\n",
      }
    )
    pkg_tar_v2 = _build_tar_gz_bytes(
      files={"demo-1.0/include/demo.h": b"#define DEMO 2\n"}
    )

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      tf_root = tmp_path / "root"
      tf_root.mkdir()

      with _LocalArchiveServer(
        {"/demo_v1.tar.gz": pkg_tar, "/demo_v2.tar.gz": pkg_tar_v2}
      ) as srv:
        bad_dep = dl.DependencySpec(
          name="demo",
          archives=dl.ArchiveSpec(
            urls=(srv.url("/demo_v1.tar.gz"),), sha256="0" * 64
          ),
        )
        with self.assertRaises(ValueError):
          dl.download_and_prepare(bad_dep, dl_dir, tf_root)
        self.assertFalse((dl_dir / "demo").exists())
        self.assertEqual(list(dl_dir.iterdir()), [])

        dep_v1 = dl.DependencySpec(
          name="demo",
          archives=dl.ArchiveSpec(
            urls=(srv.url("/demo_v1.tar.gz"),),
            sha256=_sha256_bytes(pkg_tar),
          ),
        )
        dl.download_and_prepare(dep_v1, dl_dir, tf_root)
        dest = dl_dir / "demo"
        self.assertTrue((dest / "include/demo.h").is_file())
        self.assertFalse((dest / "BUILD").exists())
        self.assertFalse((dest / "sub/BUILD.bazel").exists())

        if os.name != "nt":
          dir_mode = stat.S_IMODE(dest.stat().st_mode)
          self.assertEqual(dir_mode, 0o777 & ~dl.UMASK)

        count_after_first = srv.request_counts["/demo_v1.tar.gz"]
        dl.download_and_prepare(dep_v1, dl_dir, tf_root)
        self.assertEqual(
          srv.request_counts["/demo_v1.tar.gz"], count_after_first
        )

        # Updating archive SHA-256 invalidates stamp and swaps atomically.
        dep_v2 = dl.DependencySpec(
          name="demo",
          archives=dl.ArchiveSpec(
            urls=(srv.url("/demo_v2.tar.gz"),),
            sha256=_sha256_bytes(pkg_tar_v2),
          ),
        )
        dl.download_and_prepare(dep_v2, dl_dir, tf_root)
        self.assertEqual(
          (dest / "include/demo.h").read_bytes(), b"#define DEMO 2\n"
        )

  def test_zip_permissions_unix_vs_non_unix_and_traversal(self):
    if os.name == "nt":
      self.skipTest("POSIX permission bits test")

    # create_system=3 (Unix) vs create_system=11 (NTFS with 0o666/0o777)
    unix_zip = _build_zip_bytes(
      [
        ("pkg/run.sh", b"#!/bin/sh\n", 3, 0o755),
        ("pkg/read.h", b"int x;\n", 3, 0o444),
      ]
    )
    ntfs_zip = _build_zip_bytes(
      [
        ("ntpkg/dir/", b"", 11, 0o777),
        ("ntpkg/dir/file.h", b"int y;\n", 11, 0o666),
      ]
    )
    evil_zip = _build_zip_bytes([("../escaped.txt", b"bad", 3, 0o644)])

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      z1 = tmp_path / "unix.zip"
      z1.write_bytes(unix_zip)
      out1 = tmp_path / "out1"
      dl.extract_archive(z1, out1)
      self.assertEqual(
        stat.S_IMODE((out1 / "run.sh").stat().st_mode), 0o755 & ~dl.UMASK
      )
      self.assertEqual(
        stat.S_IMODE((out1 / "read.h").stat().st_mode), 0o644 & ~dl.UMASK
      )

      z2 = tmp_path / "ntfs.zip"
      z2.write_bytes(ntfs_zip)
      out2 = tmp_path / "out2"
      dl.extract_archive(z2, out2)
      self.assertEqual(
        stat.S_IMODE((out2 / "dir").stat().st_mode), 0o777 & ~dl.UMASK
      )
      self.assertEqual(
        stat.S_IMODE((out2 / "dir/file.h").stat().st_mode), 0o666 & ~dl.UMASK
      )
      self.assertEqual(
        stat.S_IMODE((out2 / "dir/file.h").stat().st_mode) & 0o002, 0
      )

      z3 = tmp_path / "evil.zip"
      z3.write_bytes(evil_zip)
      with self.assertRaises(ValueError):
        dl.extract_archive(z3, tmp_path / "out3")

  def test_tar_symlink_copy_fallback_and_escape_rejection(self):
    tar_with_links = _build_tar_gz_bytes(
      files={"fb-1.0/src/real.txt": b"hello symlink"},
      symlinks={"fb-1.0/java/link.txt": "../src/real.txt"},
    )
    evil_tar = _build_tar_gz_bytes(
      files={"fb-1.0/src/real.txt": b"x"},
      symlinks={"fb-1.0/escape": "../../../etc/passwd"},
    )

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      t1 = tmp_path / "links.tar.gz"
      t1.write_bytes(tar_with_links)
      out1 = tmp_path / "out1"

      prev = dl._FORCE_SYMLINK_COPY_FALLBACK
      dl._FORCE_SYMLINK_COPY_FALLBACK = True
      try:
        dl.extract_archive(t1, out1)
      finally:
        dl._FORCE_SYMLINK_COPY_FALLBACK = prev

      copied = out1 / "java/link.txt"
      self.assertFalse(copied.is_symlink())
      self.assertEqual(copied.read_bytes(), b"hello symlink")

      t2 = tmp_path / "evil.tar.gz"
      t2.write_bytes(evil_tar)
      with self.assertRaises(Exception):
        dl.extract_archive(t2, tmp_path / "out2")

  def test_sub_archives_cmsis_layout(self):
    primary = _build_zip_bytes(
      [("CMSIS_6-abc/CMSIS/Core/Include/cmsis.h", b"// core\n", 3, 0o644)]
    )
    dfp = _build_zip_bytes(
      [
        (
          "Cortex_DFP-def/Device/ARMCM55/Include/ARMCM55.h",
          b"// dfp\n",
          3,
          0o644,
        )
      ]
    )
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      with _LocalArchiveServer({"/cmsis.zip": primary, "/dfp.zip": dfp}) as srv:
        dep = dl.DependencySpec(
          name="cmsis",
          archives=dl.ArchiveSpec(
            urls=(srv.url("/cmsis.zip"),),
            sha256=_sha256_bytes(primary),
            sub_archives=(
              dl.SubArchiveSpec(
                urls=(srv.url("/dfp.zip"),),
                sha256=_sha256_bytes(dfp),
                subdir="Cortex_DFP",
              ),
            ),
          ),
        )
        dl.download_and_prepare(dep, dl_dir, tmp_path)
        self.assertTrue((dl_dir / "cmsis/CMSIS/Core/Include/cmsis.h").is_file())
        self.assertTrue(
          (
            dl_dir / "cmsis/Cortex_DFP/Device/ARMCM55/Include/ARMCM55.h"
          ).is_file()
        )
        stamp = json.loads(
          (dl_dir / "cmsis/.download_complete").read_text(encoding="utf-8")
        )
        self.assertEqual(
          stamp["sub_archives"],
          [{"subdir": "Cortex_DFP", "sha256": _sha256_bytes(dfp)}],
        )

  def test_ndsplib_nested_zip_crlf_latin1_patch_and_rename(self):
    # Inner zip contains \r\r\n + Latin-1 0xa9 byte + binary file with NUL.
    src_bytes = (
      b"/* Copyright \xa9 Vendor */\r\r\nint old_fn(void) { return 0; }\r\r\n"
    )
    renaming_bytes = (
      b"#define fft16 NatureDSP_fft16\r\n"
      b"#define NatureDSP_keep NatureDSP_second\r\n"
    )
    bin_bytes = b"\x00\x01\r\r\n\xff"
    inner_zip = _build_zip_bytes(
      [
        ("NDSP_HiFi3_rel/library/fft/src.c", src_bytes, 3, 0o444),
        (
          "NDSP_HiFi3_rel/library/include_private/__renaming__.h",
          renaming_bytes,
          3,
          0o444,
        ),
        ("NDSP_HiFi3_rel/library/blob.bin", bin_bytes, 3, 0o644),
      ]
    )
    outer_zip = _build_zip_bytes(
      [
        (
          "ndsplib-hifi3-commit/NDSP_HiFi3/NDSP_HiFi3_v1.zip",
          inner_zip,
          3,
          0o644,
        )
      ]
    )
    patch_bytes = (
      b"--- a/library/fft/src.c\r\n"
      b"+++ b/library/fft/src.c\r\n"
      b"@@ -1,2 +1,2 @@\r\n"
      b" /* Copyright \xa9 Vendor */\r\n"
      b"-int old_fn(void) { return 0; }\r\n"
      b"+int new_fn(void) { return 42; }\r\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      tf_root = tmp_path / "tf"
      patch_rel = "ext_libs/ndsplib-hifi3.patch"
      patch_file = tf_root / patch_rel
      patch_file.parent.mkdir(parents=True)
      patch_file.write_bytes(patch_bytes)

      with _LocalArchiveServer({"/ndsp.zip": outer_zip}) as srv:
        dep = dl.DependencySpec(
          name="ndsplib-hifi3",
          archives=dl.ArchiveSpec(
            urls=(srv.url("/ndsp.zip"),),
            sha256=_sha256_bytes(outer_zip),
          ),
          patch=patch_rel,
          hook="ndsplib",
          ndsp_core_name="HiFi3",
        )
        dl.download_and_prepare(dep, dl_dir, tf_root)

      patched = (dl_dir / "ndsplib-hifi3/library/fft/src.c").read_bytes()
      self.assertIn(b"int new_fn(void) { return 42; }\n", patched)
      self.assertNotIn(b"\r", patched)

      # Binary file containing NUL must not be modified.
      self.assertEqual(
        (dl_dir / "ndsplib-hifi3/library/blob.bin").read_bytes(), bin_bytes
      )

      # sed 's/NatureDSP_/NatureDSP_TFLM_/' replaces first match per line.
      renamed = (
        dl_dir / "ndsplib-hifi3/library/include_private/__renaming__.h"
      ).read_bytes()
      self.assertEqual(
        renamed,
        b"#define fft16 NatureDSP_TFLM_fft16\n"
        b"#define NatureDSP_TFLM_keep NatureDSP_second\n",
      )

      # Zero .bak/.rej/.orig files.
      leftovers = [
        p
        for p in (dl_dir / "ndsplib-hifi3").rglob("*")
        if p.suffix in (".bak", ".rej", ".orig")
      ]
      self.assertEqual(leftovers, [])

  def test_corstone300_installer_and_clean_stdout(self):
    if os.name == "nt":
      self.skipTest("POSIX shell installer test")

    installer_sh = (
      b"#!/usr/bin/env bash\n"
      b"set -e\n"
      b"echo 'noisy installer stdout that must be captured'\n"
      b'while [[ $# -gt 0 ]]; do\n'
      b'  if [[ "$1" == "-d" ]]; then DEST="$2"; shift 2; else shift; fi\n'
      b"done\n"
      b'mkdir -p "$DEST/models/Linux64_GCC-9.3"\n'
      b'echo "#!/bin/sh" > "$DEST/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55"\n'
      b'chmod 755 "$DEST/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55"\n'
    )
    corstone_tgz = _build_tar_gz_bytes(
      files={
        "FVP_Corstone_SSE-300.sh": installer_sh,
        "license_terms/eula.txt": b"license\n",
      },
      modes={"FVP_Corstone_SSE-300.sh": 0o755},
    )

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      with _LocalArchiveServer({"/fvp.tgz": corstone_tgz}) as srv:
        dep = dl.DependencySpec(
          name="corstone300",
          archives={
            ("linux", "x86_64"): dl.ArchiveSpec(
              urls=(srv.url("/fvp.tgz"),),
              sha256=_sha256_bytes(corstone_tgz),
            )
          },
          hook="corstone300",
        )
        orig = dl.DEPENDENCIES.get("corstone300")
        dl.DEPENDENCIES["corstone300"] = dep
        try:
          stdout_buf = io.StringIO()
          old_stdout = sys.stdout
          sys.stdout = stdout_buf
          try:
            rc = dl.main(
              [
                "--downloads_dir",
                str(dl_dir),
                "--tensorflow_root",
                str(tmp_path),
                "--host_os",
                "linux",
                "--host_arch",
                "x86_64",
                "corstone300",
              ]
            )
          finally:
            sys.stdout = old_stdout
        finally:
          if orig is not None:
            dl.DEPENDENCIES["corstone300"] = orig

      self.assertEqual(rc, 0)
      self.assertEqual(stdout_buf.getvalue(), "SUCCESS\n")
      self.assertTrue(
        (
          dl_dir
          / "corstone300/models/Linux64_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55"
        ).is_file()
      )
      self.assertFalse(
        (dl_dir / "corstone300/FVP_Corstone_SSE-300.sh").exists()
      )
      self.assertFalse((dl_dir / "corstone300/license_terms").exists())

  def test_ethos_u_core_platform_local_src_and_cc_override(self):
    if os.name == "nt":
      self.skipTest("POSIX mock compiler test")

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      tf_root = tmp_path / "tf"
      local_src = tmp_path / "prebaked_ethos"
      (local_src / "targets/corstone-300").mkdir(parents=True)
      (local_src / "targets/corstone-300/platform.ld").write_text(
        "#define MEM_SIZE 0x1000\nMEMORY { RAM : LENGTH = MEM_SIZE }\n",
        encoding="utf-8",
      )
      patch_rel = "ethos.patch"
      (tf_root / patch_rel).parent.mkdir(parents=True)
      (tf_root / patch_rel).write_text(
        "--- a/targets/corstone-300/platform.ld\n"
        "+++ b/targets/corstone-300/platform.ld\n"
        "@@ -1,2 +1,2 @@\n"
        "-#define MEM_SIZE 0x1000\n"
        "+#define MEM_SIZE 0x2000\n"
        " MEMORY { RAM : LENGTH = MEM_SIZE }\n",
        encoding="utf-8",
      )

      mock_cc = tmp_path / "mock-gcc.sh"
      mock_cc.write_text(
        "#!/usr/bin/env bash\nexec gcc \"$@\"\n",
        encoding="utf-8",
      )
      os.chmod(str(mock_cc), 0o755)

      dep = dl.DependencySpec(
        name="ethos_u_core_platform",
        archives=dl.ArchiveSpec(
          urls=("https://example.invalid/unused.tar.gz",),
          sha256="a" * 64,
        ),
        patch=patch_rel,
        hook="ethos_u_core_platform",
      )

      old_env = os.environ.get("ETHOS_U_CORE_PLATFORM_SRC")
      os.environ["ETHOS_U_CORE_PLATFORM_SRC"] = str(local_src)
      try:
        dl.download_and_prepare(
          dep,
          dl_dir,
          tf_root,
          host_os="linux",
          host_arch="x86_64",
          cc_override=str(mock_cc),
        )
        parsed = (
          dl_dir
          / "ethos_u_core_platform/targets/corstone-300/platform_parsed.ld"
        ).read_text(encoding="utf-8")
        self.assertIn("0x2000", parsed)

        # Mutating pre-baked source invalidates tree_sha256 in stamp.
        (local_src / "extra.txt").write_text("changed\n", encoding="utf-8")
        self.assertFalse(
          dl.is_already_downloaded(
            dep,
            dl_dir / "ethos_u_core_platform",
            tf_root,
            "linux",
            "x86_64",
          )
        )
      finally:
        if old_env is None:
          os.environ.pop("ETHOS_U_CORE_PLATFORM_SRC", None)
        else:
          os.environ["ETHOS_U_CORE_PLATFORM_SRC"] = old_env

  def test_unstamped_arc_build_artifact_preserved(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      lib_path = dl_dir / "arc_mli_custom/bin/arc/libmli.a"
      lib_path.parent.mkdir(parents=True)
      lib_path.write_bytes(b"hand-built-archive")

      dep = dl.DEPENDENCIES["embarc_mli_1_1"]
      dl.download_and_prepare(
        dep,
        dl_dir,
        tmp_path,
        dest_name="arc_mli_custom",
        host_os="linux",
        host_arch="x86_64",
      )
      self.assertEqual(lib_path.read_bytes(), b"hand-built-archive")
      self.assertFalse((dl_dir / "arc_mli_custom/.download_complete").exists())

  def test_dest_name_validation_and_print_metadata(self):
    for invalid in ("../evil", "a/b", "a\\b", ".hidden", "..", "/abs"):
      with self.assertRaises(ValueError):
        dl.validate_dest_name(invalid)

    with tempfile.TemporaryDirectory() as tmp:
      non_existent_dl = Path(tmp) / "should_not_be_created"
      out_buf = io.StringIO()
      old_stdout = sys.stdout
      sys.stdout = out_buf
      try:
        rc = dl.main(
          [
            "--downloads_dir",
            str(non_existent_dl),
            "--print_metadata",
            "xa_nnlib_hifi4",
            "ndsplib-hifi3",
          ]
        )
      finally:
        sys.stdout = old_stdout

      self.assertEqual(rc, 0)
      self.assertFalse(non_existent_dl.exists())
      lines = out_buf.getvalue().strip().splitlines()
      self.assertEqual(len(lines), 4)
      self.assertTrue(lines[0].startswith("LIBRARY_URL=https://"))
      self.assertTrue(lines[1].startswith("LIBRARY_SHA256="))
      self.assertTrue(lines[2].startswith("LIBRARY_URL=https://"))
      self.assertTrue(lines[3].startswith("LIBRARY_SHA256="))

      # --dest_name with >1 dependency must fail cleanly.
      rc_bad = dl.main(
        [
          "--downloads_dir",
          str(non_existent_dl),
          "--dest_name",
          "custom_dir",
          "flatbuffers",
          "kissfft",
        ]
      )
      self.assertNotEqual(rc_bad, 0)

  def test_multiprocess_concurrent_download_race(self):
    archive_bytes = _build_tar_gz_bytes(
      files={"race-1.0/header.h": b"#pragma once\n" * 200}
    )
    sha256 = _sha256_bytes(archive_bytes)

    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      dl_dir = tmp_path / "downloads"
      tf_root = tmp_path / "tf"
      tf_root.mkdir()

      with _LocalArchiveServer({"/race.tar.gz": archive_bytes}) as srv:
        worker_args = [
          (srv.url("/race.tar.gz"), sha256, str(dl_dir), str(tf_root))
          for _ in range(8)
        ]
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=8) as pool:
          results = pool.map(_race_worker, worker_args)

        self.assertEqual(results, [True] * 8)
        self.assertEqual(srv.request_counts.get("/race.tar.gz"), 1)

      self.assertTrue((dl_dir / "race_pkg/header.h").is_file())
      self.assertTrue((dl_dir / "race_pkg/.download_complete").is_file())
      entries = [p.name for p in dl_dir.iterdir()]
      self.assertEqual(entries, ["race_pkg"])


def run_tests() -> int:
  suite = unittest.TestLoader().loadTestsFromTestCase(DownloadDependenciesTest)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  sys.exit(run_tests())
