#!/bin/sh

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# ---

# Output key-value pairs with which to stamp build outputs. This script is
# called by the bazel option --workspace_status_command, which is likely to be
# embedded in .bazelrc. Bazel generates some keys, such as BUILD_EMBED_LABEL,
# on its own, outside of this script. Search for "Bazel workspace status" for
# more, including the differences between STABLE_ and volatile keys.


# Unambiguous identification of the source tree
echo STABLE_GIT_HASH $(git describe --always --long --dirty)

# Human-readable timestamp of git HEAD's commit date. Use dates derived from
# git for stability across multiple invocations of the `bazel` command. Use UTC
# rather than committer or local timezones for consistency across build
# environments. Use commit date instead of author date, the default date shown
# by `git log` and GitHub, because amending, rebasing, merging, etc. can cause
# the author date of descendent commits to be earlier than those of their
# ancestors.
#
# Comparable commit dates can be produced via:
#     `TZ=UTC0 git log --pretty=fuller --date=local`.
#
echo STABLE_GIT_COMMIT_TIME $(TZ=UTC0 git show \
    --no-patch \
    --format=format:%cd \
    --date=format-local:%Y%m%d%H%M%S)
