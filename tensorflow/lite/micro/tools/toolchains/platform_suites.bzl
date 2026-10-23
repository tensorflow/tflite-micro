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
"""Starlark helpers for generating platform-transitioned TFLM test suites."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_platform//platform_data:defs.bzl", "platform_data")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load(
    "//tensorflow/lite/micro/tools/toolchains:env.bzl",
    "TFLM_PLATFORM_CONFIGS",
)

_BAREMETAL_TAGS = [
    "manual",
    "noasan",
    "nomsan",
    "notsan",
    "noubsan",
]

_FORWARDED_HOST_TAGS = [
    "noasan",
    "nomsan",
    "notsan",
    "noubsan",
]

def _short_name(target):
    if ":" in target:
        return target.split(":", 1)[1]
    return target.rsplit("/", 1)[-1]

def _local_rule_dict(target):
    if target.startswith("//"):
        return None
    rule_name = target[1:] if target.startswith(":") else target
    return native.existing_rules().get(rule_name)

def tflm_platform_target(name, target, profile, visibility = None):
    """Wraps a C++ binary/test target for execution or build verification on a TFLM platform.

    Args:
      name: Name of the generated wrapper target (e.g. `"smoke_test_cortex_m55"`).
      target: Label of the underlying `cc_test` or `cc_binary` target.
      profile: Key in `TFLM_PLATFORM_CONFIGS` (e.g. `"cortex_m55"`, `"compression"`).
      visibility: Optional visibility list for the generated wrapper target.

    Returns:
      The label `":<name>"` if the profile is active, or `None` if disabled.
    """
    cfg = TFLM_PLATFORM_CONFIGS.get(profile)
    if not cfg or cfg.get("mode") == "off":
        return None

    wrapper_name = name
    if native.existing_rule(wrapper_name) != None:
        return ":%s" % wrapper_name

    bin_name = "%s_bin" % wrapper_name
    platform_label = "//tensorflow/lite/micro/tools/toolchains:%s_test_platform" % profile

    platform_data(
        name = bin_name,
        platform = platform_label,
        tags = ["manual"],
        target = target,
        testonly = True,
        visibility = ["//visibility:private"],
    )

    mode = cfg.get("mode", "test")
    baremetal = cfg.get("baremetal", True)

    if mode == "build":
        build_test(
            name = wrapper_name,
            tags = list(_BAREMETAL_TAGS),
            targets = [":%s" % bin_name],
            visibility = visibility,
        )
        return ":%s" % wrapper_name

    rule_info = _local_rule_dict(target)
    orig_tags = []
    if rule_info:
        raw_tags = rule_info.get("tags")
        if type(raw_tags) == "list" or type(raw_tags) == "tuple":
            orig_tags = list(raw_tags)

    if baremetal:
        tags = list(_BAREMETAL_TAGS)
    else:
        tags = ["manual"]
        for t in _FORWARDED_HOST_TAGS:
            if t in orig_tags and t not in tags:
                tags.append(t)

    runner = cfg.get("runner")
    bin_label = ":%s" % bin_name
    if runner:
        args = [
            "$(rootpath %s)" % runner,
            "$(rootpath %s)" % bin_label,
        ]
        data = [runner, bin_label]
    else:
        args = ["$(rootpath %s)" % bin_label]
        data = [bin_label]

    test_kwargs = {}
    default_timeout = cfg.get("default_timeout")
    if default_timeout:
        test_kwargs["timeout"] = default_timeout
    elif rule_info:
        if rule_info.get("size"):
            test_kwargs["size"] = rule_info.get("size")
        if rule_info.get("timeout"):
            test_kwargs["timeout"] = rule_info.get("timeout")

    if rule_info and rule_info.get("flaky"):
        test_kwargs["flaky"] = True

    sh_test(
        name = wrapper_name,
        srcs = ["//tensorflow/lite/micro/tools/toolchains:run_target.sh"],
        args = args,
        data = data,
        tags = tags,
        visibility = visibility,
        **test_kwargs
    )
    return ":%s" % wrapper_name

def tflm_platform_test_suite(
        name,
        profiles,
        targets = None,
        exclude_targets = None,
        extra_targets_by_profile = None,
        smoke_targets = None,
        smoke_profiles = None):
    """Generates per-platform wrapper targets and `test_suite` rules.

    Args:
      name: Prefix for generated suites (`<name>_<profile>_tests`).
      profiles: List of profile names for full suites.
      targets: Explicit list of test labels; if `None`, discovers all non-manual
        `cc_test` rules in the current package via `native.existing_rules()`.
      exclude_targets: Optional list of test labels to omit from `targets`.
      extra_targets_by_profile: Optional dict mapping profile -> additional test labels.
      smoke_targets: Optional list of test labels for `<name>_<profile>_smoke_tests`.
      smoke_profiles: Optional list of profile names for smoke suites.
    """
    if targets == None:
        resolved_targets = []
        for rule_name, rule_info in native.existing_rules().items():
            if rule_info.get("kind") != "cc_test":
                continue
            raw_tags = rule_info.get("tags") or []
            if (type(raw_tags) == "list" or type(raw_tags) == "tuple") and "manual" in raw_tags:
                continue
            resolved_targets.append(":%s" % rule_name)
    else:
        resolved_targets = list(targets)

    if exclude_targets:
        resolved_targets = [t for t in resolved_targets if t not in exclude_targets]

    extra_map = extra_targets_by_profile or {}

    for profile in profiles:
        cfg = TFLM_PLATFORM_CONFIGS.get(profile)
        if not cfg or cfg.get("mode") == "off":
            continue
        suite_tests = []
        for target in resolved_targets + extra_map.get(profile, []):
            wrapped = tflm_platform_target(
                name = "%s_%s" % (_short_name(target), profile),
                profile = profile,
                target = target,
                visibility = ["//visibility:public"],
            )
            if wrapped:
                suite_tests.append(wrapped)
        native.test_suite(
            name = "%s_%s_tests" % (name, profile),
            tags = ["manual"],
            tests = suite_tests,
            visibility = ["//visibility:public"],
        )

    if smoke_targets and smoke_profiles:
        for profile in smoke_profiles:
            cfg = TFLM_PLATFORM_CONFIGS.get(profile)
            if not cfg or cfg.get("mode") == "off":
                continue
            smoke_suite_tests = []
            for target in smoke_targets:
                wrapped = tflm_platform_target(
                    name = "%s_%s" % (_short_name(target), profile),
                    profile = profile,
                    target = target,
                    visibility = ["//visibility:public"],
                )
                if wrapped:
                    smoke_suite_tests.append(wrapped)
            native.test_suite(
                name = "%s_%s_smoke_tests" % (name, profile),
                tags = ["manual"],
                tests = smoke_suite_tests,
                visibility = ["//visibility:public"],
            )
