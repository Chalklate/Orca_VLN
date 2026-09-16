#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${NAVILA_AWS_INSTANCE_ID:-i-066515f762428ba55}"
AWS_PROFILE_NAME="${NAVILA_AWS_PROFILE:-navila}"
AWS_REGION_NAME="${NAVILA_AWS_REGION:-ap-northeast-1}"
LOCAL_PORT="${NAVVLM_PORT:-54321}"
REMOTE_PORT="${NAVILA_AWS_REMOTE_PORT:-54321}"

for command_name in aws session-manager-plugin; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is not installed: ${command_name}" >&2
    exit 2
  fi
done

for port in "${LOCAL_PORT}" "${REMOTE_PORT}"; do
  if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
    echo "Invalid TCP port: ${port}" >&2
    exit 2
  fi
done

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "NAVILA_AWS_INSTANCE_ID must not be empty" >&2
  exit 2
fi

echo "Checking AWS identity for profile ${AWS_PROFILE_NAME} in ${AWS_REGION_NAME}..."
if ! aws sts get-caller-identity \
  --profile "${AWS_PROFILE_NAME}" \
  --region "${AWS_REGION_NAME}" \
  >/dev/null; then
  echo "AWS profile ${AWS_PROFILE_NAME} is not authenticated." >&2
  echo "Authenticate it using your organisation's approved AWS login method, then retry." >&2
  exit 2
fi

echo "Opening 127.0.0.1:${LOCAL_PORT} -> ${INSTANCE_ID}:${REMOTE_PORT}"
echo "Keep this terminal open. Press Ctrl-C to close the tunnel."

exec aws ssm start-session \
  --target "${INSTANCE_ID}" \
  --profile "${AWS_PROFILE_NAME}" \
  --region "${AWS_REGION_NAME}" \
  --document-name AWS-StartPortForwardingSession \
  --parameters \
  "{\"portNumber\":[\"${REMOTE_PORT}\"],\"localPortNumber\":[\"${LOCAL_PORT}\"]}"
