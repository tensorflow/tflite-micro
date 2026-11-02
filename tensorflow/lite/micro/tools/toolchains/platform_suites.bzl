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
"""Starlark test rule and symbolic macros for TFLM platform test suites."""

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

def _rlocationpath(ctx, file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return ctx.workspace_name + "/" + file.short_path

def _tflm_platform_test_impl(ctx):
    guest_bin = ctx.executable.binary
    runner_bin = ctx.executable.runner

    runfiles = ctx.runfiles(
        files = [guest_bin] + ([runner_bin] if runner_bin else []),
    )
    runfiles = runfiles.merge(ctx.attr.binary[DefaultInfo].default_runfiles)

    if runner_bin:
        launcher = ctx.actions.declare_file(ctx.label.name + ".sh")
        ctx.actions.expand_template(
            template = ctx.file._launcher_template,
            output = launcher,
            substitutions = {
                "{{GUEST_RLOCATION}}": _rlocationpath(ctx, guest_bin),
                "{{RUNNER_RLOCATION}}": _rlocationpath(ctx, runner_bin),
                "{{TARGET_NAME}}": str(ctx.label),
            },
            is_executable = True,
        )
        runfiles = runfiles.merge(ctx.attr.runner[DefaultInfo].default_runfiles)
    else:
        ext = ".exe" if guest_bin.extension == "exe" else ".bin"
        launcher = ctx.actions.declare_file(ctx.label.name + ext)
        ctx.actions.symlink(
            output = launcher,
            target_file = guest_bin,
            is_executable = True,
        )

    return [
        DefaultInfo(
            executable = launcher,
            runfiles = runfiles,
        ),
    ]

_tflm_platform_test = rule(
    doc = "Executes a platform-transitioned test binary directly or under a host simulator.",
    implementation = _tflm_platform_test_impl,
    test = True,
    attrs = {
        "binary": attr.label(
            cfg = "target",
            executable = True,
            mandatory = True,
        ),
        "runner": attr.label(
            cfg = "target",
            executable = True,
        ),
        "_launcher_template": attr.label(
            allow_single_file = True,
            default = Label(
                "//tensorflow/lite/micro/tools/toolchains:platform_launcher.sh.tpl",
            ),
        ),
    },
)

def _emit_platform_target(
        name,
        target,
        profile,
        visibility,
        flaky = False,
        orig_tags = None,
        size = None,
        timeout = None):
    cfg = TFLM_PLATFORM_CONFIGS.get(profile)
    if not cfg or cfg.get("mode") == "off":
        return None

    bin_name = "%s_bin" % name
    platform_label = Label(
        "//tensorflow/lite/micro/tools/toolchains:%s_test_platform" % profile,
    )

    platform_data(
        name = bin_name,
        platform = platform_label,
        tags = ["manual"],
        target = target,
        testonly = True,
    )

    mode = cfg.get("mode", "test")
    baremetal = cfg.get("baremetal", True)

    if mode == "build":
        build_test(
            name = name,
            tags = list(_BAREMETAL_TAGS),
            targets = [":%s" % bin_name],
            visibility = visibility,
        )
        return ":%s" % name

    if baremetal:
        tags = list(_BAREMETAL_TAGS)
    else:
        tags = ["manual"]
        for t in _FORWARDED_HOST_TAGS:
            if t in (orig_tags or []) and t not in tags:
                tags.append(t)

    runner = cfg.get("runner")
    test_kwargs = {}
    if runner:
        test_kwargs["runner"] = Label(runner)

    default_timeout = cfg.get("default_timeout")
    if default_timeout:
        test_kwargs["timeout"] = default_timeout
    else:
        if size:
            test_kwargs["size"] = size
        if timeout:
            test_kwargs["timeout"] = timeout

    if flaky:
        test_kwargs["flaky"] = True

    _tflm_platform_test(
        name = name,
        binary = ":%s" % bin_name,
        tags = tags,
        visibility = visibility,
        **test_kwargs
    )
    return ":%s" % name

def _tflm_platform_target_macro_impl(
        name,
        visibility,
        target,
        profile,
        flaky,
        size,
        tags,
        timeout):
    _emit_platform_target(
        name = name,
        flaky = flaky,
        orig_tags = tags,
        profile = profile,
        size = size,
        target = target,
        timeout = timeout,
        visibility = visibility,
    )

tflm_platform_target = macro(
    doc = "Wraps a C++ binary or test target for execution or build verification on a TFLM platform.",
    implementation = _tflm_platform_target_macro_impl,
    attrs = {
        "flaky": attr.bool(
            configurable = False,
            default = False,
        ),
        "profile": attr.string(
            configurable = False,
            mandatory = True,
        ),
        "size": attr.string(
            configurable = False,
        ),
        "tags": attr.string_list(
            configurable = False,
        ),
        "target": attr.label(
            configurable = False,
            mandatory = True,
        ),
        "timeout": attr.string(
            configurable = False,
        ),
    },
)

def tflm_platform_sh_test(
        name,
        profile,
        srcs,
        args = None,
        data = None,
        size = None,
        tags = None,
        timeout = None,
        visibility = None):
    """Declares a shell test gated on a TFLM platform's execution mode."""
    cfg = TFLM_PLATFORM_CONFIGS.get(profile)
    if not cfg or cfg.get("mode") == "off":
        return

    resolved_tags = tags if tags != None else list(_BAREMETAL_TAGS)
    if cfg.get("mode", "test") == "build":
        build_test(
            name = name,
            tags = resolved_tags,
            targets = data or [],
            visibility = visibility,
        )
        return

    sh_kwargs = {}
    if size:
        sh_kwargs["size"] = size
    if timeout:
        sh_kwargs["timeout"] = timeout
    sh_test(
        name = name,
        srcs = srcs,
        args = args or [],
        data = data or [],
        tags = resolved_tags,
        visibility = visibility,
        **sh_kwargs
    )

def _rule_metadata(existing, curr_pkg, target_label):
    if target_label.package != curr_pkg:
        return [], None, None, False
    rule_info = existing.get(target_label.name)
    if not rule_info:
        return [], None, None, False
    raw_tags = rule_info.get("tags")
    orig_tags = (
        list(raw_tags) if type(raw_tags) == "list" or type(raw_tags) == "tuple" else []
    )
    return (
        orig_tags,
        rule_info.get("size"),
        rule_info.get("timeout"),
        bool(rule_info.get("flaky")),
    )

def _tflm_platform_test_suite_macro_impl(
        name,
        visibility,
        exclude_targets,
        extra_target_profiles,
        profiles,
        smoke_profiles,
        smoke_targets,
        targets):
    existing = native.existing_rules()
    curr_pkg = native.package_name()

    if targets:
        resolved_targets = list(targets)
    else:
        resolved_targets = []
        for rule_name, rule_info in existing.items():
            if rule_info.get("kind") != "cc_test":
                continue
            raw_tags = rule_info.get("tags") or []
            if (type(raw_tags) == "list" or type(raw_tags) == "tuple") and "manual" in raw_tags:
                continue
            resolved_targets.append(native.package_relative_label(":" + rule_name))

    if exclude_targets:
        excluded = {t: True for t in exclude_targets}
        resolved_targets = [t for t in resolved_targets if t not in excluded]

    extra_by_profile = {}
    for target_label, profile_csv in (extra_target_profiles or {}).items():
        for prof in profile_csv.split(","):
            prof = prof.strip()
            if prof:
                extra_by_profile.setdefault(prof, []).append(target_label)

    emitted = {}

    for profile in profiles:
        cfg = TFLM_PLATFORM_CONFIGS.get(profile)
        if not cfg or cfg.get("mode") == "off":
            continue
        suite_tests = []
        for target_label in resolved_targets + extra_by_profile.get(profile, []):
            wrapper_name = "%s_%s_%s" % (name, target_label.name, profile)
            if wrapper_name in emitted:
                wrapped = emitted[wrapper_name]
            else:
                orig_tags, size, timeout, flaky = _rule_metadata(
                    existing,
                    curr_pkg,
                    target_label,
                )
                wrapped = _emit_platform_target(
                    name = wrapper_name,
                    flaky = flaky,
                    orig_tags = orig_tags,
                    profile = profile,
                    size = size,
                    target = target_label,
                    timeout = timeout,
                    visibility = visibility,
                )
                emitted[wrapper_name] = wrapped
            if wrapped:
                suite_tests.append(wrapped)
        native.test_suite(
            name = "%s_%s_tests" % (name, profile),
            tags = ["manual"],
            tests = suite_tests,
            visibility = visibility,
        )

    if smoke_targets and smoke_profiles:
        for profile in smoke_profiles:
            cfg = TFLM_PLATFORM_CONFIGS.get(profile)
            if not cfg or cfg.get("mode") == "off":
                continue
            smoke_list = list(smoke_targets)
            for extra_label in extra_by_profile.get(profile, []):
                if extra_label not in smoke_list:
                    smoke_list.append(extra_label)
            smoke_suite_tests = []
            for target_label in smoke_list:
                wrapper_name = "%s_%s_%s" % (name, target_label.name, profile)
                if wrapper_name in emitted:
                    wrapped = emitted[wrapper_name]
                else:
                    orig_tags, size, timeout, flaky = _rule_metadata(
                        existing,
                        curr_pkg,
                        target_label,
                    )
                    wrapped = _emit_platform_target(
                        name = wrapper_name,
                        flaky = flaky,
                        orig_tags = orig_tags,
                        profile = profile,
                        size = size,
                        target = target_label,
                        timeout = timeout,
                        visibility = visibility,
                    )
                    emitted[wrapper_name] = wrapped
                if wrapped:
                    smoke_suite_tests.append(wrapped)
            native.test_suite(
                name = "%s_%s_smoke_tests" % (name, profile),
                tags = ["manual"],
                tests = smoke_suite_tests,
                visibility = visibility,
            )

tflm_platform_test_suite = macro(
    doc = "Generates per-platform wrapper targets and test_suite rules as a package finalizer.",
    finalizer = True,
    implementation = _tflm_platform_test_suite_macro_impl,
    attrs = {
        "exclude_targets": attr.label_list(
            configurable = False,
            default = [],
        ),
        "extra_target_profiles": attr.label_keyed_string_dict(
            configurable = False,
            default = {},
        ),
        "profiles": attr.string_list(
            configurable = False,
            mandatory = True,
        ),
        "smoke_profiles": attr.string_list(
            configurable = False,
            default = [],
        ),
        "smoke_targets": attr.label_list(
            configurable = False,
            default = [],
        ),
        "targets": attr.label_list(
            configurable = False,
            default = [],
        ),
    },
)
