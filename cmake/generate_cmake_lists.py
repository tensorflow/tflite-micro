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
"""Generates and verifies CMake source and test manifests from Bazel queries."""

import argparse
import ast
from collections import deque
import dataclasses
import difflib
import os
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

GENERATED_CMAKE_HEADER = """\
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
# AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
# Regenerate with: bazel run //cmake:update_cmake_lists
# Verified by:     bazel test //cmake:cmake_lists_staleness_test"""

SOURCE_EXTENSIONS = (".cc", ".c", ".S", ".s")
HEADER_EXTENSIONS = (".h",)
TEST_SUFFIXES = ("_test.cc", "_test.c")
ASSET_EXTENSIONS = (".tflite", ".wav", ".bmp", ".csv", ".npy")

GENERATED_HEADER_LABEL_MAP: dict[str, str] = {
  "//tensorflow/lite/micro/compression:metadata_cc_srcs": (
    "tensorflow/lite/micro/compression/metadata_generated.h"
  ),
  "//tensorflow/lite/micro/schema:schema_fbs_srcs": (
    "tensorflow/lite/micro/schema/schema_generated.h"
  ),
}

_SIGNAL_OP_USAGE_RE = re.compile(
  r"\b(?:tflm_signal::|Add(?:Delay|Energy|FftAutoScale|FilterBank"
  r"|FilterBankLog|FilterBankSquareRoot|FilterBankSpectralSubtraction"
  r"|Framer|Irfft|OverlapAdd|PCAN|Rfft|Stacker|Window)\s*\()"
)

_CATEGORY_CONFIGS: tuple[tuple[str, str, str | None, bool], ...] = (
  ("core", "# Core unit tests", None, False),
  ("kernel", "# Kernel unit tests", None, False),
  ("signal", "# Signal kernel unit tests", "TFLM_ENABLE_SIGNAL", True),
  ("example", "# Example tests", "TFLM_BUILD_EXAMPLES", False),
  (
    "seanet",
    "# SeaNet integration tests",
    "TFLM_BUILD_INTEGRATION_TESTS AND TFLM_ENABLE_SIGNAL",
    True,
  ),
)

_EXTRA_TEST_GUARDS: dict[str, str] = {
  "tflm_memory_arena_threshold_test": (
    "NOT CMAKE_CROSSCOMPILING AND NOT TFLM_OPTIMIZED_KERNELS"
    " AND NOT TFLM_CO_PROCESSOR"
  ),
}


def normalize_bazel_label(label: str) -> str:
  """Normalizes Bzlmod canonical main-repo labels ('@@//...') to '//...'."""
  label = label.strip()
  return label[2:] if label.startswith("@@//") else label


def bazel_label_to_path(label: str) -> str:
  """Converts '//pkg/subpkg:sub/file.cc' to 'pkg/subpkg/sub/file.cc'."""
  label = normalize_bazel_label(label)
  if label.startswith("@"):
    raise ValueError(
      f"External repo label cannot map to workspace path: {label!r}"
    )
  if not label.startswith("//"):
    return label
  pkg, sep, target = label[2:].partition(":")
  return (f"{pkg}/{target}" if pkg else target) if sep else pkg


def read_source_labels(query_file: pathlib.Path) -> list[str]:
  """Reads non-test source paths from a `bazel query` output file."""
  paths = {
    bazel_label_to_path(line)
    for line in query_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
  }
  return sorted(
    p
    for p in paths
    if p.endswith(SOURCE_EXTENSIONS) and not p.endswith(TEST_SUFFIXES)
  )


def read_header_labels(query_file: pathlib.Path) -> list[str]:
  """Reads header paths (mapping FlatBuffers schema rules) from a query file."""
  paths: set[str] = set()
  for raw_line in query_file.read_text(encoding="utf-8").splitlines():
    if not raw_line.strip():
      continue
    norm = normalize_bazel_label(raw_line)
    if norm in GENERATED_HEADER_LABEL_MAP:
      paths.add(GENERATED_HEADER_LABEL_MAP[norm])
      continue
    rel_path = bazel_label_to_path(norm)
    if not rel_path.endswith(HEADER_EXTENSIONS):
      raise ValueError(
        f"Unexpected non-header label in {query_file}: {raw_line.strip()!r}"
      )
    paths.add(rel_path)
  return sorted(paths)


def format_cmake_source_list(var_name: str, items: list[str]) -> str:
  """Formats a CMake set() source/header list anchored at ${PROJECT_SOURCE_DIR}."""
  body = "\n".join(f'  "${{PROJECT_SOURCE_DIR}}/{item}"' for item in items)
  return f"set({var_name}\n{body}\n)" if body else f"set({var_name}\n)"


def check_file_matches(
  actual_path: pathlib.Path, expected_content: str, label: str
) -> bool:
  """Verifies a checked-in CMake file matches expected generated content."""
  actual_content = actual_path.read_text(encoding="utf-8")
  if actual_content == expected_content:
    return True
  diff = "".join(
    difflib.unified_diff(
      actual_content.splitlines(keepends=True),
      expected_content.splitlines(keepends=True),
      fromfile=f"checked-in/{label}",
      tofile=f"generated/{label}",
    )
  )
  sys.stderr.write(
    f"ERROR: {label} is out of date with Bazel BUILD definitions.\n"
    f"Run `bazel run //cmake:update_cmake_lists` to update it.\n\n{diff}\n"
  )
  return False


def _specialize_by_basename(base: list[str], overrides: list[str]) -> list[str]:
  ovr_names = {pathlib.Path(p).name for p in overrides}
  return sorted(
    overrides + [s for s in base if pathlib.Path(s).name not in ovr_names]
  )


def _duplicate_basenames(paths: list[str]) -> dict[str, list[str]]:
  by_name: dict[str, list[str]] = {}
  for path in paths:
    by_name.setdefault(pathlib.Path(path).name, []).append(path)
  return {k: v for k, v in sorted(by_name.items()) if len(v) > 1}


def _assert_no_duplicate_basenames(paths: list[str], context: str) -> None:
  if dups := _duplicate_basenames(paths):
    details = "; ".join(f"{n}: {items}" for n, items in dups.items())
    raise ValueError(f"Duplicate source basenames in {context}: {details}")


def _assert_valid_signal_basenames(paths: list[str], context: str) -> None:
  """Allows signal/src/<op>.cc + signal/micro/kernels/<op>.cc pairs only."""
  bad = {
    n: items
    for n, items in _duplicate_basenames(paths).items()
    if len(items) != 2
    or set(items) != {f"signal/src/{n}", f"signal/micro/kernels/{n}"}
  }
  if bad:
    details = "; ".join(f"{n}: {items}" for n, items in bad.items())
    raise ValueError(
      f"Invalid duplicate signal basenames in {context}: {details}"
    )


def collect_sources_data(
  core_query: pathlib.Path,
  system_query: pathlib.Path,
  kernel_query: pathlib.Path,
  signal_query: pathlib.Path,
  micro_hdrs_query: pathlib.Path,
  signal_hdrs_query: pathlib.Path,
  test_support_query: pathlib.Path,
) -> tuple[str, set[str]]:
  """Renders cmake/generated/tflm_sources.cmake and returns library sources."""
  core_srcs = read_source_labels(core_query)
  system_srcs = read_source_labels(system_query)
  all_kernel_srcs = read_source_labels(kernel_query)
  all_signal_srcs = read_source_labels(signal_query)
  all_micro_hdrs = read_header_labels(micro_hdrs_query)
  all_signal_hdrs = read_header_labels(signal_hdrs_query)
  test_support_srcs = read_source_labels(test_support_query)

  kernel_prefix = "tensorflow/lite/micro/kernels/"
  ref_kernel_srcs: list[str] = []
  backend_kernel_srcs: dict[str, list[str]] = {}
  for path in all_kernel_srcs:
    if path.startswith("third_party/hexagon/"):
      backend_kernel_srcs.setdefault("hexagon", []).append(path)
    elif path.startswith(kernel_prefix) and "/" in path[len(kernel_prefix) :]:
      backend = path[len(kernel_prefix) :].split("/", 1)[0]
      backend_kernel_srcs.setdefault(backend, []).append(path)
    else:
      ref_kernel_srcs.append(path)

  kissfft_wrapper_prefix = "signal/src/kiss_fft_wrappers/"
  micro_public_hdrs: list[str] = []
  backend_kernel_hdrs: dict[str, list[str]] = {}
  for hdr in all_micro_hdrs:
    if hdr.startswith(kissfft_wrapper_prefix):
      continue
    if hdr.startswith(kernel_prefix) and "/" in hdr[len(kernel_prefix) :]:
      subdir = hdr[len(kernel_prefix) :].split("/", 1)[0]
      if subdir != "internal":
        backend_kernel_hdrs.setdefault(subdir, []).append(hdr)
        continue
    micro_public_hdrs.append(hdr)

  signal_kernel_prefix = "signal/micro/kernels/"
  signal_srcs = [
    p
    for p in all_signal_srcs
    if not (
      p.startswith(signal_kernel_prefix)
      and "/" in p[len(signal_kernel_prefix) :]
    )
  ]

  # Exclude private KissFFT wrappers and headers already owned by tflm_micro.
  signal_public_hdrs = sorted(
    hdr
    for hdr in set(all_signal_hdrs) - set(micro_public_hdrs)
    if not hdr.startswith(kissfft_wrapper_prefix)
  )

  _assert_no_duplicate_basenames(
    core_srcs + system_srcs + ref_kernel_srcs,
    "TFLM_MICRO_CORE_SRCS + TFLM_SYSTEM_DEFAULT_SRCS + TFLM_MICRO_KERNEL_SRCS",
  )
  for backend, ovr_srcs in sorted(backend_kernel_srcs.items()):
    _assert_no_duplicate_basenames(
      ovr_srcs, f"TFLM_MICRO_KERNEL_SRCS_{backend.upper()}"
    )
    specialized_micro = _specialize_by_basename(ref_kernel_srcs, ovr_srcs)
    _assert_no_duplicate_basenames(
      core_srcs + system_srcs + specialized_micro,
      f"specialized tflm_micro ({backend})",
    )

  _assert_valid_signal_basenames(signal_srcs, "TFLM_SIGNAL_SRCS")
  _assert_no_duplicate_basenames(
    core_srcs + system_srcs + test_support_srcs,
    "TFLM_MICRO_CORE_SRCS + TFLM_SYSTEM_DEFAULT_SRCS + TFLM_TEST_SUPPORT_SRCS",
  )

  sections = [
    GENERATED_CMAKE_HEADER,
    format_cmake_source_list("TFLM_MICRO_PUBLIC_HDRS", micro_public_hdrs),
    *(
      format_cmake_source_list(
        f"TFLM_MICRO_KERNEL_HDRS_{b.upper()}", backend_kernel_hdrs[b]
      )
      for b in sorted(backend_kernel_hdrs)
    ),
    format_cmake_source_list("TFLM_MICRO_CORE_SRCS", core_srcs),
    format_cmake_source_list("TFLM_SYSTEM_DEFAULT_SRCS", system_srcs),
    format_cmake_source_list("TFLM_MICRO_KERNEL_SRCS", ref_kernel_srcs),
    *(
      format_cmake_source_list(
        f"TFLM_MICRO_KERNEL_SRCS_{b.upper()}", backend_kernel_srcs[b]
      )
      for b in sorted(backend_kernel_srcs)
    ),
    format_cmake_source_list("TFLM_SIGNAL_PUBLIC_HDRS", signal_public_hdrs),
    format_cmake_source_list("TFLM_SIGNAL_SRCS", signal_srcs),
    format_cmake_source_list("TFLM_TEST_SUPPORT_SRCS", test_support_srcs),
  ]

  compiled_library_srcs = set(
    core_srcs
    + system_srcs
    + all_kernel_srcs
    + all_signal_srcs
    + test_support_srcs
  )
  return "\n\n".join(sections) + "\n", compiled_library_srcs


@dataclasses.dataclass(frozen=True)
class _BazelRuleNode:
  name: str
  rule_class: str
  srcs: tuple[str, ...]
  hdrs: tuple[str, ...]
  deps: tuple[str, ...]
  outputs: tuple[str, ...]
  cmd: str = ""


@dataclasses.dataclass(frozen=True)
class _CmakeTestSpec:
  bazel_label: str
  cmake_name: str
  category: str
  labels: tuple[str, ...]
  srcs: tuple[str, ...]
  assets: tuple[str, ...]
  deps: tuple[str, ...]
  requires_signal: bool
  requires_compression: bool


def _parse_compression_gated_tests(
  test_defs_build_path: pathlib.Path,
) -> set[str]:
  """Extracts cc_test labels gated on with_compression_enabled in BUILD defs.

  Uses `--output=build` AST because `tflm_defines()` injects
  `//:with_compression_enabled` into `<rule-input>` on all tests in XML.
  """
  tree = ast.parse(
    test_defs_build_path.read_text(encoding="utf-8"),
    filename=str(test_defs_build_path),
  )
  gated: set[str] = set()
  for node in tree.body:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
      continue
    call = node.value
    if not (isinstance(call.func, ast.Name) and call.func.id == "cc_test"):
      continue
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    if (
      not {"name", "generator_location", "target_compatible_with"} <= kw.keys()
    ):
      continue
    if "with_compression_enabled" not in ast.unparse(
      kw["target_compatible_with"]
    ):
      continue
    name = ast.literal_eval(kw["name"])
    loc = ast.literal_eval(kw["generator_location"])
    if not loc.startswith("BUILD:") and "/BUILD:" not in loc:
      raise ValueError(
        f"Unexpected generator_location={loc!r} for cc_test({name!r})"
      )
    pkg = "" if loc.startswith("BUILD:") else loc.partition("/BUILD:")[0]
    gated.add(f"//{pkg}:{name}" if pkg else f"//:{name}")
  return gated


def _classify_cc_test(bazel_label: str) -> tuple[str, str, tuple[str, ...]]:
  """Maps a Bazel cc_test label to (cmake_target_name, category, labels)."""
  if not bazel_label.startswith("//") or ":" not in bazel_label:
    raise ValueError(
      f"Expected canonical '//pkg:name' cc_test label, got {bazel_label!r}"
    )
  pkg, _, name = bazel_label[2:].partition(":")
  if pkg == "tensorflow/lite/micro/kernels" or pkg.startswith(
    "tensorflow/lite/micro/kernels/"
  ):
    return f"tflm_kernel_{name}", "kernel", ("unit", "kernel")
  if pkg == "signal/micro/kernels" or pkg.startswith("signal/"):
    return f"tflm_signal_{name}", "signal", ("unit", "signal")
  if pkg.startswith("tensorflow/lite/micro/examples/"):
    return f"tflm_{name}", "example", ("example",)
  seanet_prefix = "tensorflow/lite/micro/integration_tests/seanet/"
  if pkg.startswith(seanet_prefix):
    op = pkg[len(seanet_prefix) :]
    if not op or "/" in op:
      raise ValueError(
        f"Unexpected SeaNet integration test package: {bazel_label!r}"
      )
    return f"tflm_seanet_{op}_{name}", "seanet", ("integration", "seanet")
  if pkg == "tensorflow/lite/micro" or pkg.startswith("tensorflow/lite/micro/"):
    return f"tflm_{name}", "core", ("unit", "core")
  raise ValueError(f"Unclassified cc_test package for {bazel_label!r}")


def _parse_test_deps_xml(
  test_deps_xml_path: pathlib.Path,
) -> tuple[dict[str, _BazelRuleNode], set[str]]:
  """Parses `bazel query --output=xml` into rule nodes and generated labels."""
  raw_xml = test_deps_xml_path.read_text(encoding="utf-8")
  if raw_xml.startswith('<?xml version="1.1"'):
    raw_xml = '<?xml version="1.0"' + raw_xml[len('<?xml version="1.1"') :]

  root = ET.fromstring(raw_xml)
  generated_labels = {
    normalize_bazel_label(e.attrib["name"])
    for e in root.findall("generated-file")
    if e.attrib.get("name")
  }
  rules: dict[str, _BazelRuleNode] = {}
  for rule_elem in root.findall("rule"):
    rule_name = normalize_bazel_label(rule_elem.attrib["name"])
    lists_by_attr = {
      le.attrib.get("name", ""): tuple(
        normalize_bazel_label(lbl.attrib["value"])
        for lbl in le.findall("label")
        if "value" in lbl.attrib
      )
      for le in rule_elem.findall("list")
    }
    cmd_el = rule_elem.find("string[@name='cmd']")
    outputs = tuple(
      normalize_bazel_label(oe.attrib["name"])
      for oe in rule_elem.findall("rule-output")
      if "name" in oe.attrib
    )
    generated_labels.update(outputs)
    rules[rule_name] = _BazelRuleNode(
      name=rule_name,
      rule_class=rule_elem.attrib["class"],
      srcs=lists_by_attr.get("srcs", ()),
      hdrs=lists_by_attr.get("hdrs", ()),
      deps=lists_by_attr.get("deps", ()),
      outputs=outputs,
      cmd=cmd_el.attrib.get("value", "") if cmd_el is not None else "",
    )
  return rules, generated_labels


def _sources_invoke_signal_ops(
  rel_srcs: list[str], workspace_root: pathlib.Path | None
) -> bool:
  if workspace_root is None:
    return False
  return any(
    (p := workspace_root / rel).is_file()
    and _SIGNAL_OP_USAGE_RE.search(p.read_text(encoding="utf-8"))
    for rel in rel_srcs
  )


def _format_cc_test_block(spec: _CmakeTestSpec, indent: int = 0) -> str:
  guard = _EXTRA_TEST_GUARDS.get(spec.cmake_name)
  body_pad = " " * (indent + (2 if guard else 0))
  lines = [
    f"{body_pad}tflm_add_cc_test(",
    f"{body_pad}  NAME {spec.cmake_name}",
    f"{body_pad}  SRCS",
    *(f'{body_pad}    "${{PROJECT_SOURCE_DIR}}/{src}"' for src in spec.srcs),
  ]
  if spec.assets:
    lines.append(f"{body_pad}  ASSETS")
    lines.extend(
      f'{body_pad}    "${{PROJECT_SOURCE_DIR}}/{asset}"'
      for asset in spec.assets
    )
  if spec.deps:
    lines.append(f"{body_pad}  DEPS {' '.join(spec.deps)}")
  if spec.labels:
    lines.append(f"{body_pad}  LABELS {' '.join(spec.labels)}")
  lines.append(f"{body_pad})")
  body = "\n".join(lines)
  if not guard:
    return body
  pad = " " * indent
  return f"{pad}if({guard})\n{body}\n{pad}endif()"


def _render_category_blocks(
  specs: list[_CmakeTestSpec],
  base_indent: int = 0,
  signal_implied: bool = False,
) -> list[str]:
  """Renders test blocks for a category with compression/signal guards."""
  pad = " " * base_indent
  buckets: dict[tuple[bool, bool], list[_CmakeTestSpec]] = {}
  for s in specs:
    buckets.setdefault(
      (s.requires_compression, s.requires_signal and not signal_implied), []
    ).append(s)
  blocks = [
    _format_cc_test_block(s, indent=base_indent)
    for s in buckets.get((False, False), ())
  ]
  for key, cond in (
    ((True, False), "TFLM_ENABLE_COMPRESSION"),
    ((False, True), "TFLM_ENABLE_SIGNAL"),
    ((True, True), "TFLM_ENABLE_COMPRESSION AND TFLM_ENABLE_SIGNAL"),
  ):
    if group := buckets.get(key):
      inner = "\n\n".join(
        _format_cc_test_block(s, indent=base_indent + 2) for s in group
      )
      blocks.append(f"{pad}if({cond})\n{inner}\n{pad}endif()")
  return blocks


def render_tests_cmake(
  test_deps_xml_path: pathlib.Path,
  test_defs_build_path: pathlib.Path,
  compiled_library_srcs: set[str],
  workspace_root: pathlib.Path | None = None,
) -> str:
  """Renders cmake/generated/tflm_generated_tests.cmake from Bazel queries."""
  compression_gated_labels = _parse_compression_gated_tests(
    test_defs_build_path
  )
  rules, generated_labels = _parse_test_deps_xml(test_deps_xml_path)

  assets_by_label: dict[str, tuple[str, ...]] = {}
  for rule in rules.values():
    if rule.rule_class != "genrule" or "generate_cc_arrays" not in rule.cmd:
      continue
    raw_assets = tuple(
      sorted(
        {
          bazel_label_to_path(s)
          for s in rule.srcs
          if not s.startswith("@")
          and bazel_label_to_path(s).endswith(ASSET_EXTENSIONS)
        }
      )
    )
    if raw_assets:
      for lbl in (rule.name, *rule.outputs):
        assets_by_label[lbl] = raw_assets

  # Umbrella resolver targets pull //signal/... into Bazel deps even though
  # MicroMutableOpResolver is header-only and links only explicitly added ops.
  umbrella_resolver_targets = {
    "//tensorflow/lite/micro:op_resolvers",
    "//tensorflow/lite/micro/kernels:micro_ops",
  }
  known_signal_resolver_tests = {
    "//tensorflow/lite/micro/examples/micro_speech:micro_speech_test",
  }

  cc_test_rules = sorted(
    (r for r in rules.values() if r.rule_class == "cc_test"),
    key=lambda r: r.name,
  )
  if not cc_test_rules:
    raise ValueError(f"No cc_test rules found in {test_deps_xml_path}")

  def _is_test_src(lbl: str) -> str | None:
    if lbl.startswith("@") or lbl in generated_labels or lbl in rules:
      return None
    rel = bazel_label_to_path(lbl)
    return (
      rel
      if rel.endswith(SOURCE_EXTENSIONS) and rel not in compiled_library_srcs
      else None
    )

  specs_by_name: dict[str, _CmakeTestSpec] = {}
  for test_rule in cc_test_rules:
    cmake_name, category, ctest_labels = _classify_cc_test(test_rule.name)
    if cmake_name in specs_by_name:
      prev = specs_by_name[cmake_name].bazel_label
      raise ValueError(
        f"Duplicate CMake test target name {cmake_name!r} from "
        f"{prev!r} and {test_rule.name!r}"
      )

    direct_srcs: set[str] = set()
    helper_srcs: set[str] = set()
    test_assets: set[str] = set()
    has_non_umbrella_signal_dep = False
    requires_kissfft_hdrs = test_rule.name.startswith(
      "//signal/src/kiss_fft_wrappers:"
    )

    visited: set[str] = {test_rule.name}
    queue: deque[str] = deque([test_rule.name])
    while queue:
      cur_lbl = queue.popleft()
      cur_rule = rules[cur_lbl]
      if cur_rule.rule_class != "genrule":
        target_srcs = direct_srcs if cur_lbl == test_rule.name else helper_srcs
        for src_lbl in cur_rule.srcs:
          if (rel := _is_test_src(src_lbl)) is not None:
            target_srcs.add(rel)

      ignore_signal_edges = cur_lbl in umbrella_resolver_targets
      for next_lbl in cur_rule.srcs + cur_rule.hdrs + cur_rule.deps:
        if ignore_signal_edges and next_lbl.startswith("//signal/"):
          continue
        if cur_lbl == test_rule.name and next_lbl.startswith(
          "//signal/src/kiss_fft_wrappers:"
        ):
          requires_kissfft_hdrs = True
        if next_lbl in assets_by_label:
          test_assets.update(assets_by_label[next_lbl])
        if next_lbl.startswith("//signal/"):
          has_non_umbrella_signal_dep = True
        if next_lbl in rules and next_lbl not in visited:
          visited.add(next_lbl)
          queue.append(next_lbl)

    all_test_srcs = sorted(direct_srcs) + sorted(helper_srcs - direct_srcs)
    if not all_test_srcs:
      raise ValueError(
        f"cc_test {test_rule.name!r} resolved to zero C/C++ sources"
      )

    requires_signal = (
      category in ("signal", "seanet")
      or test_rule.name in known_signal_resolver_tests
      or has_non_umbrella_signal_dep
    )
    if not requires_signal and _sources_invoke_signal_ops(
      all_test_srcs, workspace_root
    ):
      raise ValueError(
        f"cc_test {test_rule.name!r} invokes signal ops via an umbrella "
        "op_resolver target without a direct //signal/... dep; add it to "
        "`known_signal_resolver_tests` in cmake/generate_cmake_lists.py so "
        "sandboxed staleness checks stay deterministic."
      )

    test_deps = (["tflm::signal"] if requires_signal else []) + (
      ["tflm::kissfft"] if requires_kissfft_hdrs else []
    )
    specs_by_name[cmake_name] = _CmakeTestSpec(
      bazel_label=test_rule.name,
      cmake_name=cmake_name,
      category=category,
      labels=ctest_labels,
      srcs=tuple(all_test_srcs),
      assets=tuple(sorted(test_assets)),
      deps=tuple(test_deps),
      requires_signal=requires_signal,
      requires_compression=test_rule.name in compression_gated_labels,
    )

  all_specs = sorted(specs_by_name.values(), key=lambda s: s.cmake_name)
  sections: list[str] = [GENERATED_CMAKE_HEADER]
  for category, heading, outer_cond, signal_implied in _CATEGORY_CONFIGS:
    cat_specs = [s for s in all_specs if s.category == category]
    if not cat_specs:
      continue
    if outer_cond is None:
      sections.append(heading)
      sections.extend(
        _render_category_blocks(cat_specs, 0, signal_implied=signal_implied)
      )
    else:
      inner = "\n\n".join(
        _render_category_blocks(cat_specs, 2, signal_implied=signal_implied)
      )
      sections.append(f"{heading}\nif({outer_cond})\n{inner}\nendif()")
  return "\n\n".join(sections) + "\n"


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  for flag in (
    "--core_srcs",
    "--system_srcs",
    "--kernel_srcs",
    "--signal_srcs",
    "--micro_hdrs",
    "--signal_hdrs",
    "--test_support_srcs",
    "--test_deps_xml",
    "--test_defs_build",
  ):
    parser.add_argument(flag, type=pathlib.Path, required=True)
  parser.add_argument("--sources_cmake", type=pathlib.Path)
  parser.add_argument("--tests_cmake", type=pathlib.Path)
  mode_group = parser.add_mutually_exclusive_group(required=True)
  mode_group.add_argument("--update_workspace", action="store_true")
  mode_group.add_argument("--check", action="store_true")
  args = parser.parse_args()
  if args.check and (args.sources_cmake is None or args.tests_cmake is None):
    parser.error("--check requires both --sources_cmake and --tests_cmake.")
  return args


def main() -> int:
  args = _parse_args()
  env_ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  ws_root = pathlib.Path(env_ws).resolve() if env_ws else None
  try:
    srcs_content, compiled_lib_srcs = collect_sources_data(
      args.core_srcs,
      args.system_srcs,
      args.kernel_srcs,
      args.signal_srcs,
      args.micro_hdrs,
      args.signal_hdrs,
      args.test_support_srcs,
    )
    tests_content = render_tests_cmake(
      args.test_deps_xml, args.test_defs_build, compiled_lib_srcs, ws_root
    )
  except (ValueError, TypeError, SyntaxError, OSError) as exc:
    sys.stderr.write(f"ERROR: {exc}\n")
    return 1

  if args.check:
    srcs_ok = check_file_matches(
      args.sources_cmake, srcs_content, "cmake/generated/tflm_sources.cmake"
    )
    tests_ok = check_file_matches(
      args.tests_cmake,
      tests_content,
      "cmake/generated/tflm_generated_tests.cmake",
    )
    return 0 if (srcs_ok and tests_ok) else 1

  if ws_root is None:
    sys.stderr.write(
      "ERROR: BUILD_WORKSPACE_DIRECTORY not set "
      "(use `bazel run //cmake:update_cmake_lists`).\n"
    )
    return 1
  gen_dir = ws_root / "cmake" / "generated"
  gen_dir.mkdir(parents=True, exist_ok=True)
  outputs = (
    (gen_dir / "tflm_sources.cmake", srcs_content),
    (gen_dir / "tflm_generated_tests.cmake", tests_content),
  )
  for out_path, content in outputs:
    out_path.write_text(content, encoding="utf-8")
  print(f"Updated {', '.join(str(p) for p, _ in outputs)}.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
