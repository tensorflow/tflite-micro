#!/usr/bin/env bash
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
#
# Installs the exact native Bazel binary specified by .bazelversion into
# /usr/local/bin/bazel-real with an offline version-checking wrapper at
# /usr/local/bin/bazel (and /usr/local/bin/bazelisk symlink), ensuring CI
# containers never fetch Bazel over the network at runtime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.bazelversion" ]]; then
  VERSION_FILE="${SCRIPT_DIR}/.bazelversion"
elif [[ -f "${SCRIPT_DIR}/../.bazelversion" ]]; then
  VERSION_FILE="${SCRIPT_DIR}/../.bazelversion"
elif [[ -f ".bazelversion" ]]; then
  VERSION_FILE=".bazelversion"
else
  echo "ERROR: .bazelversion not found" >&2
  exit 1
fi

BAZEL_VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}")"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) BAZEL_ARCH="x86_64" ;;
  aarch64|arm64) BAZEL_ARCH="arm64" ;;
  *)
    echo "ERROR: Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

BAZEL_ARTIFACT="bazel-${BAZEL_VERSION}-linux-${BAZEL_ARCH}"
BASE_URL="https://releases.bazel.build/${BAZEL_VERSION}/release"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

pushd "${TMP_DIR}" > /dev/null
wget -q "${BASE_URL}/${BAZEL_ARTIFACT}"
wget -q "${BASE_URL}/${BAZEL_ARTIFACT}.sha256"
sha256sum -c "${BAZEL_ARTIFACT}.sha256"

chmod 0755 "${BAZEL_ARTIFACT}"
mv "${BAZEL_ARTIFACT}" /usr/local/bin/bazel-real
popd > /dev/null

cat > /usr/local/bin/bazel <<EOF
#!/bin/sh
set -e
BAKED_VERSION="${BAZEL_VERSION}"
if [ -f .bazelversion ]; then
  WANT_VERSION="\$(tr -d '[:space:]' < .bazelversion)"
  if [ -n "\${WANT_VERSION}" ] && [ "\${WANT_VERSION}" != "\${BAKED_VERSION}" ]; then
    echo "ERROR: .bazelversion (\${WANT_VERSION}) != container Bazel (\${BAKED_VERSION}). Rebuild CI Docker images." >&2
    exit 1
  fi
fi
exec /usr/local/bin/bazel-real "\$@"
EOF

chmod 0755 /usr/local/bin/bazel
ln -sf /usr/local/bin/bazel /usr/local/bin/bazelisk

/usr/local/bin/bazel --version
